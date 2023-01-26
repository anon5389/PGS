
import os
import sys
import pickle
import time
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

class NeuralNetwork(nn.Module):
    def __init__(self, input_shape, hidden_size):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        pred = self.linear_relu_stack(x)
        return pred

class EltSampler(torch.utils.data.Dataset):
    def __init__(self, G_list):
        self.feat_all = []
        self.label_all = []
        self.id_all = []
        for i in range(len(G_list)):
            self.feat_all.append(G_list[i]['feat'])
            self.label_all.append(G_list[i]['label'])
            self.id_all.append(G_list[i]['id'])
    
    def __len__(self):
        return len(self.feat_all)
    
    def __getitem__(self, idx):
        return {'feat':self.feat_all[idx],
                'label':self.label_all[idx],
                'id':self.id_all[idx]}

class Data(Dataset):
    def __init__(self, X, Y):
        self.X=torch.from_numpy(X)
        self.Y=torch.from_numpy(Y)
        self.len=self.X.shape[0]
    
    def __getitem__(self,index):      
        return self.X[index], self.Y[index]
    
    def __len__(self):
        return self.len

def prepare_dataloader(train_list, batch_size):
    dataset_sampler = EltSampler(train_list)
    train_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=batch_size, 
            shuffle=False)
    
    return train_dataset_loader

def prepare_dataloaders(train_list,val_list,batch_size):
    dataset_sampler = EltSampler(train_list)
    train_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=batch_size, 
            shuffle=False)
    
    dataset_sampler = EltSampler(val_list)
    val_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=batch_size, 
            shuffle=False)
    
    return train_dataset_loader, val_dataset_loader

def split_dataset(x,y, val_size):
    data_len = x.shape[0]
    idxs = np.arange(x.shape[0])
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    # create a training and validation split
    x = x[indices]
    y = y[indices]
    idxs = idxs[indices]
    
    train_list = []
    val_list = []
    for i in range(x[val_size:].shape[0]):
        train_element = {"feat":   x[i], "label": y[i], "id": idxs[i]}
        train_list.append(train_element)
    
    for i in range(x[:val_size].shape[0]):
        val_element = {"feat":   x[data_len-val_size+i], "label": y[data_len-val_size+i], "id": idxs[data_len-val_size+i]}
        val_list.append(val_element)
    
    return train_list, val_list

def get_data(dataset_name):
    with open('continuous_data/'+ dataset_name +'.pickle', 'rb') as f:
        dataset = pickle.load(f)
    
    return dataset["x"], dataset["y"]

def normalize(x):
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x_std = np.where(x_std == 0.0, 1.0, x_std)
    print("mean: ", x_mean)
    print("std: ", x_std)
    return (x - x_mean) / x_std

def validate_dnn(val_dataset, model_dnn):
    model_dnn.eval()
    preds = []
    labels = []
    
    for batch_idx, data in enumerate(val_dataset):
        feat = Variable(data['feat'].float(), requires_grad=False).to(device)
        label = Variable(data['label'].float()).to(device)
        output = model_dnn(feat)
        preds.extend(output.squeeze().tolist())
        labels.extend(label.squeeze().tolist())
    prediction_on_test = {}
    prediction_on_test['labels'] = preds
    prediction_on_test['preds'] = labels

    return prediction_on_test

def mean_squared_error(y_true, y_pred):
    y_true = np.array(y_true).reshape((-1, 1))
    y_pred = np.array(y_pred).reshape((-1, 1))
    output_errors = np.average((y_true - y_pred) ** 2, axis=0)
    
    return np.average(output_errors)   


if __name__ == '__main__':
    dataset_names = ["AntMorphology", "DKittyMorphology", "HopperController", "Superconductor"]
    # dataset_names = ['TFBind8', 'GFP', 'UTR']
    dataset_ = dataset_names[2] + "_vae_64"
    print(dataset_)
    x, y = get_data(dataset_)
    # print(x.max(), x.min())
    # print(y.max(), y.min())
    # x_normalized = normalize(x)
    # y_normalized = normalize(y)    
    print(x.shape, y.shape)
    # print(np.max(x), np.min(x))
    # print(np.max(y), np.min(y))
    x_normalized = x #.numpy()
    y_normalized = y
    # print(x_normalized.max(), x_normalized.min())
    # print(y_normalized.max(), y_normalized.min())
    # sys.exit("error")
    input_shape = x.shape[1]
    hidden_size = 2048
    model_learning_rate = 0.0003
    val_size = int(y.shape[0]/10)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNetwork(input_shape, hidden_size).to(device)

    torch.manual_seed(42)

    train_list, val_list = split_dataset(x_normalized, y_normalized, val_size=val_size)
    train_dataloader, val_dataloader = prepare_dataloaders(train_list, val_list, batch_size=128)

    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_learning_rate)
    n_epochs = 50

    prediction_on_train = []
    prediction_on_test = []
    for epoch in range(n_epochs): 
        st = time.time()
        current_loss = 0.0
        preds = []
        labels = []
        for i, data in enumerate(train_dataloader, 0):
            inputs = Variable(data['feat'].float(), requires_grad=False).to(device)
            targets = Variable(data['label'].float()).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            current_loss += loss.item()
            if i % 500 == 499:
                print('Loss after mini-batch %5d: %.3f' %
                    (i + 1, current_loss / 500))
                current_loss = 0.0
        

            preds.extend(outputs.squeeze().tolist())
            labels.extend(targets.squeeze().tolist())
        
        prediction_on_epoch = {}
        prediction_on_epoch['labels'] = labels
        prediction_on_epoch['preds'] = preds
        prediction_on_train.append(prediction_on_epoch)
        prediction_on_test = validate_dnn(val_dataloader, model)
        et = time.time()
        print("Epoch "+ str(epoch) 
                + ": time = " + str(round(et-st,4))
                + ", train MSE = " + str(round(mean_squared_error(prediction_on_train[-1]['labels'], prediction_on_train[-1]['preds']), 4))
                + ", test MSE = " + str(round(mean_squared_error(prediction_on_test['labels'], prediction_on_test['preds']), 4))
                )
        
        PATH = dataset_+"_std_dnn.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, PATH)

