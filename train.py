import numpy as np
from datetime import datetime
from tqdm import tqdm
import torch
from torch.utils import data
import torch.nn as nn
from model import deeplob, ae_deeplob
from data_process import Dataset

def train(model, criterion, optimizer, train_loader, test_loader, epochs):
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    best_test_loss = np.inf
    best_test_epoch = 0

    for it in tqdm(range(epochs)):

        model.train()
        t0 = datetime.now()
        train_loss = []
        for inputs, targets in train_loader:
            # move data to GPU
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)
            # print("inputs.shape:", inputs.shape)
            # zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            # print("about to get model output")
            outputs = model(inputs)
            # print("done getting model output")
            # print("outputs.shape:", outputs.shape, "targets.shape:", targets.shape)
            loss = criterion(outputs, targets)
            # Backward and optimize
            # print("about to optimize")
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        # Get train loss and test loss
        train_loss = np.mean(train_loss)  # a little misleading

        model.eval()
        test_loss = []
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss.append(loss.item())
        test_loss = np.mean(test_loss)

        # Save losses
        train_losses[it] = train_loss
        test_losses[it] = test_loss

        if test_loss < best_test_loss:
            torch.save(model, './best_val_model')
            best_test_loss = test_loss
            best_test_epoch = it
            print('model saved')

        dt = datetime.now() - t0
        print(f'Epoch {it + 1}/{epochs}, Train Loss: {train_loss:.4f}, \
          Validation Loss: {test_loss:.4f}, Duration: {dt}, Best Val Epoch: {best_test_epoch}')

    return train_losses, test_losses

if __name__ == "__main__":
    data_path = 'data/'

    dec_data = np.loadtxt(data_path + 'Train_Dst_NoAuction_DecPre_CF_7.txt')
    dec_train = dec_data[:, :int(np.floor(dec_data.shape[1] * 0.8))]
    dec_val = dec_data[:, int(np.floor(dec_data.shape[1] * 0.8)):]

    batch_size = 16

    dataset_train = Dataset(data=dec_train, k=4, num_classes=3, T=100)
    dataset_val = Dataset(data=dec_val, k=4, num_classes=3, T=100)

    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ae_deeplob(y_len=dataset_train.num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    train_losses, val_losses = train(model, criterion, optimizer,
                                        train_loader, val_loader, epochs=50)