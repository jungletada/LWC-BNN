import torch
import numpy as np
import argparse
from models.dnn_models import UNet, Mlp
from data_loader_f import RadarDataset
from data_configs import train_date_selection, test_date_selection
from visualize import visualize_prediction
from utils_read import data_mkdir, de_normalize
from evaluate import eval_all, clip
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn



def get_args_parse():
    parser = argparse.ArgumentParser(description="Liquid Water Content Prediction")
    parser.add_argument('--modelname', type=str, default="mlp", help='Model type')
    parser.add_argument('--epoch', type=int, default=200, help='Epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    args = parser.parse_args()
    
    return args


def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, num_epochs, device, save_path):
    model.train()
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        running_loss = 0.0
        # Training loop
        for iteration, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # Print training and validation loss
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

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
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    with torch.no_grad():
        test_predictions = []
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            test_predictions.append(outputs.cpu())

    # Concatenate all the batched outputs
    test_predictions = torch.cat(test_predictions, dim=0)
    test_predictions = torch.squeeze(test_predictions, 1)  # Adjust dimensions if necessary

    # Save predictions
    y_pred = test_predictions.numpy()
    npy_save_path = f'{save_path}/{args.modelname}_preds.npy'
    np.save(npy_save_path, y_pred)
    
    y_test, y_pred = de_normalize(y_test, y_pred)
    if len(y_test.shape) == 1:
        y_test = y_test.reshape(-1, 41)
        y_pred = y_pred.reshape(-1, 41)
    y_test, y_pred = clip(y_test), clip(y_pred)
    eval_all(y_test, y_pred, save_file=save_path + f"eval_{args.modelname}.log")
    # Visualization
    visualize_prediction(
        y_test=y_test,
        y_pred=y_pred,
        test_dates=test_date_selection,
        img_path=save_path,
        height=heights,
        length=test_dataset.num_times,
        name=args.modelname,
        interval=20,
        type="standard")


if __name__ == '__main__':
    # arg parameters
    args = get_args_parse()
    print(f"Model: {args.modelname}")
    if_flatten = False if args.modelname in ["unet"] else True

    # Device settings
    use_cuda = True if torch.cuda.is_available() else False
    print(f"Use CUDA: {use_cuda}")
    if use_cuda:
        # torch.set_default_device('cuda')
        device = 'cuda'
    else:
        device = 'cpu'

    # Define train data
    trainset = RadarDataset(
        dates=train_date_selection, 
        if_flatten=if_flatten, 
        to_tensor=True)
    # Define test data
    testset = RadarDataset(
        dates=test_date_selection, 
        if_flatten=if_flatten, 
        to_tensor=True)

    # DataLoader
    train_loader = DataLoader(trainset, batch_size=args.batch_size, 
                              shuffle=True, drop_last=True)
    test_loader = DataLoader(testset, batch_size=args.batch_size, 
                             shuffle=False, drop_last=True)
    
    # Define model
    save_root = f'results/{args.modelname}-{args.epoch}/'
    data_mkdir(save_root)
    
    if args.modelname == "mlp":
        model = Mlp(in_dim=8, out_dim=1, hid_dim=96, n_hid_layers=2)
        
    elif args.modelname == "unet":
        model = UNet(in_channels=trainset.num_inputs, out_channels=1, features=[16, 32, 64])
        
    model.to(device)

    # Define loss function and optimizer for U-Net
    criterion = nn.L1Loss()  # or another appropriate loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epoch * len(train_loader))
   
    ckpt_save_path = f'{save_root}/best_{args.modelname}.pth'
    
    train_model(model, train_loader, test_loader, optimizer, criterion, scheduler, 
                num_epochs=args.epoch, device=device, save_path=ckpt_save_path)
    
    eval_model(model, testset, args=args, ckpt=ckpt_save_path, save_path=save_root)
    
    