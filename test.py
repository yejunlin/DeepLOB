import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from torch.utils import data
from model import deeplob
from data_process import Dataset

if __name__ == "__main__":
    data_path = 'data/'
    dec_test1 = np.loadtxt(data_path + 'Test_Dst_NoAuction_DecPre_CF_7.txt')
    dec_test2 = np.loadtxt(data_path + 'Test_Dst_NoAuction_DecPre_CF_8.txt')
    dec_test3 = np.loadtxt(data_path + 'Test_Dst_NoAuction_DecPre_CF_9.txt')
    dec_test = np.hstack((dec_test1, dec_test2, dec_test3))
    batch_size = 64
    dataset_test = Dataset(data=dec_test, k=4, num_classes=3, T=100)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load('best_val_model')
    all_targets = []
    all_predictions = []
    for inputs, targets in test_loader:
        # Move to GPU
        inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)

        # Forward pass
        outputs = model(inputs)

        # Get prediction
        # torch.max returns both max and argmax
        _, predictions = torch.max(outputs, 1)

        all_targets.append(targets.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())
    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)
    print('accuracy_score:', accuracy_score(all_targets, all_predictions))
    print(classification_report(all_targets, all_predictions, digits=4))