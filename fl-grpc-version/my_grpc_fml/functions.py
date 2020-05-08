import time
import numpy as np
import pandas as pd
import base64
import os
import copy
import torch
from models import MLP, CNNMnist, CNNCifar

client_nums = 2
model = 'cnn'
dataset = 'mnist'
if dataset == 'cifar':
    img_size = 3 * 32 * 32
else:
    img_size = 3 * 28 * 28


def Average_weight(w):
    w_avg = copy.deepcopy(w[0])
    w_para = w_avg.state_dict()
    for key in w_para.keys():
        for i in range(1, len(w)):
            w_para[key] += w[i].state_dict()[key]
        w_para[key] = torch.div(w_para[key], len(w))
    return w_avg


def GetFileNum(global_round):
    path = "./Models/LocalModels/" + str(global_round) + '/'
    count = 0
    res = []
    for files in os.listdir(path):    
        count += 1
        res.append(files)
    return count, res


def GetModel(op, global_round):
    # initial
    if op == 0:
        if model == 'cnn':
            if dataset == 'mnist' or dataset == 'fmnist':
                global_model = CNNMnist()
            elif dataset == 'cifar':
                global_model = CNNCifar()
        elif args.model == 'mlp':
            global_model = MLP(dim_in=img_size, dim_hidden=64,
                                   dim_out=10)
        else:
            exit('Error: unrecognized model')
        torch.save(global_model, 'Models/global_model.pkl')
        with open('Models/global_model.pkl', 'rb') as file:
            encoing_string = base64.b64encode(file.read())
            return encoing_string
    else:
        timelimit = 100
        t = 0
        if os.path.isfile('Models/global_model.pkl'):
            os.remove('Models/global_model.pkl')
        while t * 10 <= timelimit:
            filenums, files = GetFileNum(global_round)
            print(filenums)
            print(files)
            if filenums == client_nums or os.path.isfile('Models/global_model.pkl'):
                if os.path.isfile('Models/global_model.pkl'):
                    pass
                else:
                    weights = []
                    for f in files:
                        print(f)
                        print(type(f))
                        f_path = "Models/LocalModels/" +str(global_round) + '/' + f
                        print(f_path)
                        #weights.append(torch.load(f_path).state_dict())
                        weights.append(torch.load(f_path))
                    new_global_model = Average_weight(weights)
                    print(type(new_global_model))
                    torch.save(new_global_model, 'Models/global_model.pkl')
                with open('Models/global_model.pkl', 'rb') as file:
                    encoing_string = base64.b64encode(file.read())
                    return encoing_string
            time.sleep(10)
        print("Error")



def SendModel(modelString, client_id, global_round):
    try:
        filename = "model" + str(client_id) + ".pkl"
        filepath = "Models/LocalModels/" + str(global_round) + "/"
        print(filepath)
        folder = os.path.exists(filepath)
        if not folder:
            os.makedirs(filepath)
        filepath += filename
        print(filepath)
        with open(filepath,"wb") as file:
            file.write(base64.b64decode(modelString))
            print("Successfully saved model...")
            return 1
    except:
        print("An error occured!")
        return 0
