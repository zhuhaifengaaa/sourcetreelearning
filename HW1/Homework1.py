# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
print(111)
# For data preprocess
import numpy as np
import csv
import os

# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

myseed = 42069  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

tr_path = 'covid.train.csv'
tt_path = 'covid.test.csv'
#
def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'
# #Dataset
class COVID19Dataset(Dataset):
    def __init__(self, path, mode='train', target_only=False):
        self.mode = mode
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data[1:])[:,1:].astype(float)

        if not target_only:
            feats = list(range(93))
        else:
            pass

        if mode=='test':
            data = data[:,feats]
            self.data=torch.FloatTensor(data)
        else:
            target=data[:,-1]
            data=data[:,feats]
            if mode=='train':
                indices=[i for i in range(len(data)) if i%10!=0]
            elif mode=='dev':
                indices=[i for i in range(len(data)) if i%10==0]
            self.data=torch.FloatTensor(data[indices])
            self.target=torch.FloatTensor(target[indices])

        self.data[:, 40:]=(self.data[:,40:]-self.data[:,40:].mean(dim=0,keepdim=True))\
                             /self.data[:,40:].std(dim=0, keepdim=True)    # 归一化
        self.dim=self.data.shape[1]
        print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))
    def __getitem__(self, index):
        if self.mode in ['train', 'dev']:
            return self.data[index], self.target[index]
        else:
            return self.data[index]

    def __len__(self):
        return len(self.data)

# DataLoader
def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False):
    ''' Generates a dataset, then is put into a dataloader. '''
    dataset = COVID19Dataset(path, mode=mode, target_only=target_only)  # Construct dataset
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)                            # Construct dataloader
    return dataloader

# Deep Neural Network
class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        self.net=nn.Sequential(
            nn.Linear(input_dim,64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.criterion=nn.MSELoss(reduction='mean')
    def forward(self, x):
        return self.net(x).squeeze(1)
    def cal_loss(self, pred, target):
        return self.criterion(pred, target)

# training
def train(tr_set, dv_set, model, config, device):
    n_epochs=config['n_epochs']

    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), **config['optim_hparas'])
    min_mse=1000.
    loss_record={'train':[],'dev':[]}
    early_stop_cnt=0
    epoch=0
    while epoch<n_epochs:
        model.train()
        for x,y in tr_set:
            optimizer.zero_grad()
            x,y = x.to(device),y.to(device)
            pred=model(x)
            mse_loss=model.cal_loss(pred,y)
            mse_loss.backward()
            optimizer.step()
            loss_record['train'].append(mse_loss.detach().cpu().item())

        dev_mse=dev(dv_set,model,device)
        if dev_mse<min_mse:
            min_mse = dev_mse
            print('Saving model (epoch = {:4d}, loss = {:.4f})'.format(epoch + 1, min_mse))
            torch.save(model.state_dict(), config['save_path'])
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > config['early_stop']:
            # Stop training if your model stops improving for "config['early_stop']" epochs.
            break

    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record

def dev(dv_set, model, device):
    model.eval()
    total_loss=0
    for x,y in dv_set:
        x,y=x.to(device),y.to(device)
        with torch.no_grad():
            pred=model(x)
            mse_loss=model.cal_loss(pred,y)
        total_loss+=mse_loss.detach().cpu().item()*len(x)
        total_loss=total_loss/len(dv_set.dataset)
        print('len(x)',len(x))
        print("len(dv_set.dataset)",len(dv_set.dataset))
        print('x',x.shape)

    return total_loss

def test(tt_set, model, device):
    model.eval()                                # set model to evalutation mode
    preds = []
    for x in tt_set:                            # iterate through the dataloader
        x = x.to(device)                        # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            preds.append(pred.detach().cpu())   # collect prediction
    preds = torch.cat(preds, dim=0).numpy()     # concatenate all predictions and convert to a numpy array
    return preds
    
def plot_learning_curve(loss_record, title=''):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]

    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    plt.ylim(0.0, 5.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()

def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None):
    ''' Plot prediction of your DNN '''
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()

def save_pred(preds, file):
    ''' Save predictions to specified file '''
    print('Saving results to {}'.format(file))
    with open(file, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])
def sum_demo(x, y):
    for _ in range(2):
        x += 1
        y += 1
        result = x + y
    return result


if __name__=='__main__':

    result = sum_demo(1, 1)
    print(result)
    device=get_device()
    os.makedirs('model', exist_ok=True)
    target_only=False
    print(1)

    config = {
            'n_epochs': 50,
            'batch_size': 270,
            'optimizer': 'SGD',
            'optim_hparas': {'lr': 0.001, 'momentum': 0.9},
            'early_stop': 200,
            'save_path': './model/model.pth'
            }
    print(1)

    tr_set = prep_dataloader(tr_path, 'train', config['batch_size'], target_only=target_only)
    dv_set = prep_dataloader(tr_path, 'dev', config['batch_size'], target_only=target_only)
    tt_set = prep_dataloader(tt_path, 'test', config['batch_size'], target_only=target_only)
    # print('111',tr_set.dataset.dim)
    model = NeuralNet(tr_set.dataset.dim).to(device)

    model_loss, model_loss_record = train(tr_set, dv_set, model, config, device)
    print('model_loss:{}, model_loss_record:{}'.format(model_loss, model_loss_record))
    plot_learning_curve(model_loss_record, title='deep model')
    plt.show()

    del model
    model = NeuralNet(tr_set.dataset.dim).to(device)
    ckpt = torch.load(config['save_path'], map_location='cpu')  # Load your best model
    model.load_state_dict(ckpt)
    plot_pred(dv_set, model, device)  # Show prediction on the validation set
    plt.show()

    preds = test(tt_set, model, device)  # predict COVID-19 cases with your model
    save_pred(preds, 'pred.csv')         # save prediction file to pred.csv



