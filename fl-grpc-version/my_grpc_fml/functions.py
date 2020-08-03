import time
import numpy as np
import pandas as pd
import base64
import pickle
from selfsave import save
import fcntl
import os
import copy
import math
import torch
from models import MLP, CNNMnist, CNNCifar

client_nums = 20
total_global_round = 20
#model = 'cnn'
model = 'densenet'
#model = 'vgg19'
dataset = 'mnist'
#dataset = 'cifar'
if dataset == 'cifar':
    img_size = 3 * 32 * 32
else:
    img_size = 3 * 28 * 28

clients_arr = []
clients_limit = 2

loss_arr = []
global_loss_per_epoch = []
pre_loss = None
stop_round = None



def Average_weight(w):
    global loss_arr
    global pre_loss
    print("-----Doing Averaging-----")
    w_avg = copy.deepcopy(w[0])
    w_para = w_avg.state_dict()
    for key in w_para.keys():
        for i in range(1, len(w)):
            w_para[key] += w[i].state_dict()[key]
        w_para[key] = torch.div(w_para[key], len(w))
    print("-----Averaging loss-----")
    print(loss_arr)
    loss_avg = sum(loss_arr) / len(loss_arr)
    loss_arr = []
    global_loss_per_epoch.append(loss_avg)
    print("length of lossarr after average", len(loss_arr))
    flag = 0
    if pre_loss is not None:
        if (pre_loss - loss_avg) / pre_loss < 0.1:
            flag = 1
    pre_loss = loss_avg
    return w_avg, flag

def Evaluate_model(m):
    pass




def GetFileNum(global_round):
    path = "./Models/LocalModels/" + str(global_round) + '/'
    count = 0
    res = []
    for files in os.listdir(path):    
        count += 1
        res.append(files)
    return count, res


def GetModel(op, client_id, global_round):
    # initial
    global stop_round
    if op == 0:
        if model == 'cnn':
            if dataset == 'mnist' or dataset == 'fmnist':
                global_model = CNNMnist()
            elif dataset == 'cifar':
                global_model = CNNCifar()
        elif model == 'mlp':
            global_model = MLP(dim_in=img_size, dim_hidden=64,
                                   dim_out=10)
        elif model == 'densenet':
            global_model = torch.hub.load('pytorch/vision:v0.5.0', 'densenet121', pretrained=False)
        elif model == 'vgg19':
            global_model = torch.hub.load('pytorch/vision:v0.5.0', 'vgg19', pretrained=False)
        else:
            exit('Error: unrecognized model')
        encoing_string = base64.b64encode(pickle.dumps(global_model)).decode()
        clients_arr.remove(client_id)
        return encoing_string
    else:
        f_path_global = 'Models/global_model' + str(global_round) + '.pkl'
        if client_id == 1:
            while 1:
                filenums, files = GetFileNum(global_round)
                if filenums == client_nums:
                    weights = []
                    for f in files:
                        f_path = "Models/LocalModels/" +str(global_round) + '/' + f
                        weights.append(torch.load(f_path))
                    new_global_model, cvg_flag  = Average_weight(weights)
                    if cvg_flag and stop_round is None:
                        stop_round = global_round
                    if global_round == total_global_round:
                        print(global_loss_per_epoch, stop_round)
                    print(f_path_global)
                    save(new_global_model, f_path_global)
                    #torch.save(new_global_model, f_path_global)
                    with open(f_path_global, 'rb') as file:
                        encoing_string = base64.b64encode(file.read())
                        clients_arr.remove(client_id)
                        return encoing_string
                time.sleep(2)
        else:
            while 1:
                if os.path.isfile(f_path_global):
                    with open(f_path_global, 'rb') as file:
                        encoing_string = base64.b64encode(file.read())
                        clients_arr.remove(client_id)
                        return encoing_string
                time.sleep(2)


def SendModel(modelString, client_id, global_round, local_loss):
    try:
        filename = "model" + str(client_id) + ".pkl"
        #print(filename)
        filepath = "Models/LocalModels/" + str(global_round) + "/"
        #print(filepath)
        folder = os.path.exists(filepath)
        if not folder:
            os.makedirs(filepath)
        filepath += filename
        print(filepath)
        file = open(filepath, 'wb')
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        file.write(base64.b64decode(modelString))
        file.close()
        loss_arr.append(local_loss)
        print("Successfully saved model...")
        clients_arr.remove(client_id)
        return 1
    except:
        print("An error occured!")
        clients_arr.remove(client_id)
        return 0


def SendModel_stream(req_it):
    try:
        tmp = next(req_it)
        models = tmp.model
        client_id = tmp.id
        global_round = tmp.gr
        for _ in req_it:
            models += _.model
        filename = "model" + str(client_id) + ".pkl"
        filepath = "Models/LocalModels/" + str(global_round) + "/"
        print(filepath)
        folder = os.path.exists(filepath)
        if not folder:
            os.makedirs(filepath)
        filepath += filename
        print(filepath)
        file = open(filepath, 'wb')
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        file.write(base64.b64decode(models))
        file.close()
        print("Successfully saved model...")
        clients_arr.remove(client_id)
        return 1
    except:
        print("An error occured!")
        clients_arr.remove(client_id)
        return 0

def Regis(client_id):
    if len(clients_arr) >= clients_limit:
        print(str(client_id) + "full")
        return 1
    else:
        clients_arr.append(client_id)
        print(str(client_id) + "empty")
        return 0

def GetReady(client_id, global_round):
    if client_id == 1:
        filenums, files = GetFileNum(global_round)
        if filenums == client_nums:
            return 0
        else:
            #print("not all come")
            return 1
    else:
        f_path_global = 'Models/global_model' + str(global_round) + '.pkl'
        if os.path.isfile(f_path_global):
            return 0
        else:
            #print(str(client_id) + "no gb model")
            return 1
