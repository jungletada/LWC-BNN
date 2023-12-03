import torch
from torch.distributions import constraints
from torch.distributions.transforms import AbsTransform, Transform

from pyro.distributions.torch import TransformedDistribution
from pyro.distributions import Normal
from pyro.distributions import TorchDistribution

class FoldedDistribution_New(TransformedDistribution):
    """
    Equivalent to ``TransformedDistribution(base_dist, AbsTransform())``,
    but additionally supports :meth:`log_prob` .

    :param ~torch.distributions.Distribution base_dist: The distribution to
        reflect.
    """

    support = constraints.nonnegative

    def __init__(self, base_dist, validate_args=None):
        if base_dist.event_shape:
            raise ValueError("Only univariate distributions can be folded.")
        super().__init__(base_dist, AbsTransform(), validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(type(self), _instance)
        return super().expand(batch_shape, _instance=new)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        dim = max(len(self.batch_shape), value.dim())
        plus_minus = value.new_tensor([1.0, -1.0]).reshape((2,) + (1,) * dim)
        return self.base_dist.log_prob(plus_minus * value).logsumexp(0)

class ReLUTransform(Transform):
    domain = constraints.real
    codomain = constraints.nonnegative

    def __call__(self, x):
        return torch.nn.functional.relu(x)

    def _inverse(self, y):
        raise NotImplementedError("Inverse is not implemented for ReLUTransform.")

    def log_abs_det_jacobian(self, x, y):
        # The Jacobian for the ReLU function is either 0 (for x < 0) or 1 (for x >= 0)
        # The log absolute determinant of the Jacobian is therefore 0 where x >= 0, and -inf where x < 0.
        return torch.where(x >= 0, torch.zeros_like(x), torch.tensor(-float('inf'), device=x.device))

class RectifiedNormal(TransformedDistribution):
    support = constraints.nonnegative

    def __init__(self, base_dist, validate_args=None):
        super().__init__(base_dist, ReLUTransform(), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(RectifiedNormal, _instance)
        return super().expand(batch_shape, _instance=new)

    def log_prob(self, value):
        # Set log prob to -inf for values less than 0, which should never be sampled
        log_prob = super().log_prob(value)
        log_prob = torch.where(value < 0, torch.tensor(-float('inf'), device=value.device), log_prob)
        return log_prob


class CensoredNormal(TorchDistribution):
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.nonnegative
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        self.loc = loc
        self.scale = scale
        self.base_dist = Normal(loc, scale)
        super().__init__(self.base_dist.batch_shape, self.base_dist.event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(CensoredNormal, _instance)
        new.base_dist = self.base_dist.expand(batch_shape)
        super(CensoredNormal, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        sample = self.base_dist.rsample(sample_shape)
        return torch.nn.functional.relu(sample)

    def log_prob(self, value):
        log_prob = self.base_dist.log_prob(value)
        log_prob = torch.where(value < 0, torch.tensor(-float('inf'), device=value.device), log_prob)
        return log_prob

    def cdf(self, value):
        cdf = self.base_dist.cdf(value)
        cdf = torch.where(value < 0, torch.zeros_like(cdf), cdf)
        return cdf

    def icdf(self, value):
        raise NotImplementedError