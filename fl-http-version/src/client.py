import numpy as np
import time
import json
import pickle
import random
import sys
import base64
import copy

from socketIO_client import SocketIO, LoggingNamespace

import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

num_items = 100 # number of data in each client
#dataset = 'mnist'
dataset = 'cifar'
local_bs = 10
local_ep = 10
optimizer = 'sgd'
lr = 0.01


def to_pickle(m):
    return base64.b64encode(pickle.dumps(m)).decode()

def to_obj(s):
    return pickle.loads(base64.b64decode(s.encode()))

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)

class LocalModel(object):
    def __init__(self, model_config, dataset, idxs):
        self.trainloader, self.validloader, self.testloader = self.train_val_test(dataset, list(idxs))
        self.criterion = nn.NLLLoss()
        self.model_config = model_config

        self.model = to_obj(model_config['model_pickle'])

    def train_val_test(self, dataset, idxs):
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, global_round, optimizer, lr, local_ep):
        self.model.train()
        epoch_loss = []

        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr,
                                        momentum=0.5)
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr,
                                         weight_decay=1e-4)

        for iter in range(local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                self.model.zero_grad()
                log_probs = self.model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                
                if (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return self.model, sum(epoch_loss) / len(epoch_loss)

    def evaluate(self):

        self.model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):

            outputs = self.model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return loss, accuracy

def sample_iid(data):
    all_idxs = [i for i in range(len(data))]
    dict_users = set(np.random.choice(all_idxs, num_items, replace=False))
    return list(dict_users)


def get_data(dataset):
    if dataset == 'cifar':
        data_dir = './data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

    elif dataset == 'mnist':
        data_dir = './data/mnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

    elif dataset == 'fmnist':

        data_dir = './data/fmnist/'

        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.2860,), (0.3529,))])

        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

    user_group = sample_iid(train_dataset)
    return train_dataset, test_dataset, user_group



class FederatedClient(object):
    MAX_DATASET_SIZE_KEPT = 1200

    def __init__(self, server_host, server_port):
        self.local_model = None

        self.sio = SocketIO(server_host, server_port, LoggingNamespace)
        self.register_handles()
        print("sent wakeup")
        self.sio.emit('client_wake_up')
        self.sio.wait()

    
    def on_init(self, *args):
        t1 = time.time()
        print(t1)
        f = open('time.txt', 'a+')
        f.write(str(t1)+'\n')
        f.close()
        model_config = args[0]
        print('on init')
        print('preparing local data based on server model_config')
        train_set, test_set, user_groups = get_data(dataset)
        print('data ready for train')
        self.local_model = LocalModel(model_config=model_config, dataset=train_set, idxs=user_groups)
        print('trainsize:', len(self.local_model.trainloader))
        self.sio.emit('client_ready', {
                'train_size': len(self.local_model.trainloader),
            })


    def register_handles(self):
        def on_connect():
            print('connect')

        def on_disconnect():
            print('disconnect')

        def on_reconnect():
            print('reconnect')

        def on_request_update(*args):
            req = args[0]
            # req:
            #     'model_id'
            #     'round_number'
            #     'current_model'
            #     'model_format'
            #     'run_validation'
            print("update requested")

            if req['model_format'] == 'pickle':
                new_model = to_obj(req['current_model'])

            self.local_model.model = new_model
            my_model, train_loss = self.local_model.update_weights(req['round_number'], optimizer, lr, local_ep) 
            resp = {
                'round_number': req['round_number'],
                'model': to_pickle(my_model),
                'train_size': len(self.local_model.trainloader),
                'train_loss': train_loss,
            }
            print("resp size is ", sys.getsizeof(resp))
            print("model_pickle_string size is ", sys.getsizeof(to_pickle(my_model)))
            self.sio.emit('client_update', resp)


        def on_stop_and_eval(*args):
            req = args[0]
            if req['model_format'] == 'pickle':
                self.local_model.model = to_obj(req['current_model'])
            test_loss, test_accuracy = self.local_model.evaluate()
            resp = {
                'test_size': len(self.local_model.testloader),
                'test_loss': test_loss,
                'test_accuracy': test_accuracy
            }
            self.sio.emit('client_eval', resp)


        self.sio.on('connect', on_connect)
        self.sio.on('disconnect', on_disconnect)
        self.sio.on('reconnect', on_reconnect)
        self.sio.on('init', lambda *args: self.on_init(*args))
        self.sio.on('request_update', on_request_update)
        self.sio.on('stop_and_eval', on_stop_and_eval)



    
    def intermittently_sleep(self, p=.1, low=10, high=100):
        if (random.random() < p):
            time.sleep(random.randint(low, high))



if __name__ == "__main__":
    FederatedClient("130.238.28.175", 5000)
