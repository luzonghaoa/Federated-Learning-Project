import numpy as np
import random
import base64
import grpc
import copy
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset



import FederatedML_pb2
import FederatedML_pb2_grpc

client_id = 3
client_nums = 3
model = 'cnn'
dataset = 'mnist'
if dataset == 'cifar':
    img_size = 3 * 32 * 32
else:
    img_size = 3 * 28 * 28

local_bs = 10
local_ep = 10
optimizer = 'sgd'
lr = 0.01

total_global_round = 5


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, dataset, idxs):
        self.trainloader, self.validloader, self.testloader = self.train_val_test(dataset, list(idxs))
        self.criterion = nn.NLLLoss()

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

    def update_weights(self, model, global_round, optimizer, lr, local_ep):
        model.train()
        epoch_loss = []

        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                        momentum=0.5)
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                         weight_decay=1e-4)

        for iter in range(local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                model.zero_grad()
                log_probs = model(images)
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

        return model, sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss

def sample_iid(data, client_nums):
    num_items = 100
    #num_items = int(len(data)/client_nums)
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

    user_group = sample_iid(train_dataset, client_nums)
    return train_dataset, test_dataset, user_group


def get_model(stub, op, global_round):
    feature = stub.GetModel(FederatedML_pb2.Option(op=op, gr=global_round))
    model = base64.b64decode(feature.model)
    return model

def send_model(stub, modelString, client_id, global_round):
    feature = stub.SendModel(FederatedML_pb2.Model(model=modelString, id=client_id, gr=global_round))
    if feature.value == 1:
        print("Model Sent")
    else:
        print("An error occurs")




def run():
    print("FL start!")
    print("-------------- Generate Data --------------")
    train_set, test_set, user_groups = get_data(dataset)
    local_progress = LocalUpdate(dataset=train_set, idxs=user_groups)
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = FederatedML_pb2_grpc.FederatedMLStub(channel)
        print("-------------- Initialize Model --------------")
        local_model = get_model(stub, 0, 0)
        with open('local_model.pkl',"wb") as file:
            file.write(local_model)
            print("Successfully saved model 0")
        print("-------------- Start Train --------------")
        for i in range(1, total_global_round + 1):
            print("Global round: ", i)
            local_model = torch.load('local_model.pkl')
            print("Load Model Successfully!")
            new_model, loss = local_progress.update_weights(model=copy.deepcopy(local_model), global_round=i, optimizer=optimizer, lr=lr, local_ep=local_ep)
            torch.save(new_model, 'tmp_model.pkl')
            with open('tmp_model.pkl', 'rb') as file:
                encoing_string = base64.b64encode(file.read())
                send_model(stub, encoing_string, client_id, i)
            print("-------------- Get New Global Model--------------")
            local_model = get_model(stub, 1, i)
            with open('local_model.pkl',"wb") as file:
                file.write(local_model)
                print("Successfully saved model" + str(i))


if __name__ == '__main__':
    run()
