import torch
import numpy as np
import argparse
from models.dnn_models import UNet, Mlp
from data_loader import RadarDataset
from data_configs import train_date_selection, test_date_selection
from visualize import visualize_prediction
from utils_read import data_mkdir, de_normalize
from evaluate import eval_all
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn


def train_model(model, train_loader, val_loader, args, save_path):
    model.train()
    best_val_loss = float('inf')
    for epoch in range(args.num_epochs):
        running_loss = 0.0
        # Training loop
        for iteration, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            args.optimizer.zero_grad()
            outputs = model(inputs)
            loss = args.criterion(outputs, targets)
            loss.backward()
            args.optimizer.step()
            args.scheduler.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                outputs = model(inputs)
                loss = args.criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # Print training and validation loss
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        # Save the model if it's the best so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved with Validation Loss: {avg_val_loss:.4f}")

        model.train()  # Switch back to training mode

    print("Training completed.")


def eval_model(model, test_dataset, args, ckpt, save_path):
    model.load_state_dict(torch.load(ckpt))
    model.eval()  # Set the model to evaluation mode
    
    heights = test_dataset.heights * 10000
    _, y_test = test_dataset.get_data_list()
    y_test = y_test.reshape(-1, 1)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    with torch.no_grad():
        test_pred = []
        for inputs, _ in test_loader:
            inputs = inputs.to(args.device)
            outputs = model(inputs)
            test_pred.append(outputs.cpu())

    # Concatenate all the batched outputs
    test_pred = torch.cat(test_pred, dim=0)
    test_pred = torch.squeeze(test_pred, 1)  # Adjust dimensions if necessary

    # Save predictions
    y_pred = test_pred.numpy()
    npy_save_path = f'{save_path}/{args.modelname}_preds.npy'
    np.save(npy_save_path, y_pred)
    
    y_test, y_pred = de_normalize(y_test, y_pred)
    y_pred[y_pred < 0] = 0
    eval_all(y_test, y_pred, save_file=save_path + f"eval_{args.modelname}.log")
    # Visualization
    visualize_prediction(
        y_test, y_pred, 
        img_path=save_path, interval=1)


def get_args_parse():
    parser = argparse.ArgumentParser(description="Liquid Water Content Prediction")
    parser.add_argument('--modelname', type=str, default="mlp", help='Model type')
    parser.add_argument('--num_epochs', type=int, default=100, help='Epochs')
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate for SVI")
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    # arg parameters
    args = get_args_parse()
    # Device settings
    use_cuda = True if torch.cuda.is_available() else False
    print(f"Model: {args.modelname}, use CUDA: {use_cuda}")
    if use_cuda:
        # torch.set_default_device('cuda')
        args.device = 'cuda'
    else:
        args.device = 'cpu'

    # Define train data
    trainset = RadarDataset(dates=train_date_selection, to_tensor=True)
    # Define test data
    testset = RadarDataset(dates=test_date_selection, to_tensor=True)

    # DataLoader
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    
    # Define model
    save_root = f'results/{args.modelname}-{args.num_epochs}/'
    data_mkdir(save_root)
    
    if args.modelname == "mlp":
        model = Mlp(in_dim=7, out_dim=1, hid_dim=96, n_hid_layers=2)
        
    elif args.modelname == "unet":
        model = UNet(in_channels=trainset.num_inputs, out_channels=1, features=[16, 32, 64])
        
    model.to(args.device)

    # Define loss function and optimizer for U-Net
    args.criterion = nn.L1Loss()  # or another appropriate loss function
    args.optimizer = optim.Adam(model.parameters(), lr=args.lr)
    args.scheduler = optim.lr_scheduler.CosineAnnealingLR(
        args.optimizer, T_max=args.num_epochs * len(train_loader))
   
    ckpt_save_path = f'{save_root}/best_{args.modelname}.pth'
    train_model(model, train_loader, test_loader, args=args, save_path=ckpt_save_path)
    eval_model(model, testset, args=args, ckpt=ckpt_save_path, save_path=save_root)
    
    