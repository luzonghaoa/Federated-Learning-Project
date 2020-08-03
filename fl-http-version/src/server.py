import pickle
import uuid

import msgpack
import random
import base64
import numpy as np
import json
import msgpack_numpy
import copy
import sys
import time

from flask import *
from flask_socketio import SocketIO
from flask_socketio import *
import logging


import torch
from models import MLP, CNNMnist, CNNCifar

#model = 'cnn'
model = 'densenet'
#model = 'vgg19'
#dataset = 'mnist'
dataset = 'cifar'
if dataset == 'cifar':
    img_size = 3 * 32 * 32
else:
    img_size = 3 * 28 * 28
epoch_per_round = 10


#todo: test msgpack for numpy
def to_pickle(m):
    return base64.b64encode(pickle.dumps(m)).decode()
    # return msgpack.packb(x, default=msgpack_numpy.encode)

def to_obj(s):
    return pickle.loads(base64.b64decode(s.encode()))
    # return msgpack.unpackb(s, object_hook=msgpack_numpy.decode)

def Average_weight(w):
    w_avg = copy.deepcopy(w[0])
    w_para = w_avg.state_dict()
    for key in w_para.keys():
        for i in range(1, len(w)):
            w_para[key] += w[i].state_dict()[key]
        w_para[key] = torch.div(w_para[key], len(w))
    return w_avg

''' TODO: weighted
def Average_lose(l):
    l_avg = copy.deepcopy(l[0])
    for i in range(1, len(l)):
        l_avg += l[i]
    w_para[key] = torch.div(w_para[key], len(w))
    return w_avg
'''

class FLServer(object):
    
    MIN_NUM_WORKERS = 1
    MAX_NUM_ROUNDS = 1
    NUM_CLIENTS_CONTACTED_PER_ROUND = 1

    def __init__(self, host, port):
        if model == 'cnn':
            if dataset == 'mnist' or dataset == 'fmnist':
                self.global_model = CNNMnist()
            elif dataset == 'cifar':
                self.global_model = CNNCifar()
        elif model == 'mlp':
            self.global_model = MLP(dim_in=img_size, dim_hidden=64,
                                   dim_out=10)
        elif model == 'densenet':
            self.global_model = torch.hub.load('pytorch/vision:v0.5.0', 'densenet121', pretrained=False)
        elif model == 'vgg19':
        	self.global_model = torch.hub.load('pytorch/vision:v0.5.0', 'vgg19', pretrained=False)
        else:
            exit('Error: unrecognized model')


        self.ready_client_sids = set()

        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.host = host
        self.port = port

        self.model_id = str(uuid.uuid4())


        self.current_round = -1  # -1 for not yet started
        self.current_round_client_updates = []
        self.eval_client_updates = []


        self.register_handles()

     
    def register_handles(self):

        @self.socketio.on('connect')
        def handle_connect():
            print(request.sid, "connected")

        @self.socketio.on('reconnect')
        def handle_reconnect():
            print(request.sid, "reconnected")

        @self.socketio.on('disconnect')
        def handle_disconnect():
            print(request.sid, "disconnected")
            if request.sid in self.ready_client_sids:
                self.ready_client_sids.remove(request.sid)

        @self.socketio.on('client_wake_up')
        def handle_wake_up():
            print("client wake_up: ", request.sid)
            print(time.time())
            emit('init', {
                    'model_pickle': to_pickle(self.global_model),
                    'model_id': self.model_id,
                })

        @self.socketio.on('client_ready')
        def handle_client_ready(data):
            print("client ready for training", request.sid, data)
            self.ready_client_sids.add(request.sid)
            if len(self.ready_client_sids) >= FLServer.MIN_NUM_WORKERS and self.current_round == -1:  # probably problem here, this -1
                self.train_next_round()

        @self.socketio.on('client_update')
        def handle_client_update(data):
            print("received client update of bytes: ", sys.getsizeof(data))
            print("handle client_update", request.sid)
            for x in data:
                if x != 'model':
                    print(x, data[x])
            # data:
            #   round_number
            #   model
            #   train_size
            #   train_loss

            # discard outdated update
            if data['round_number'] == self.current_round:
                self.current_round_client_updates += [data]
                self.current_round_client_updates[-1]['model'] = to_obj(data['model'])
                
                # tolerate 30% unresponsive clients
                if len(self.current_round_client_updates) > FLServer.NUM_CLIENTS_CONTACTED_PER_ROUND * .7:
                    updated_model = Average_weight([x['model'] for x in self.current_round_client_updates])
                    self.global_model = updated_model
                    print("model updated")  
                    if self.current_round >= FLServer.MAX_NUM_ROUNDS:
                        self.stop_and_eval()
                    else:
                        self.train_next_round()

        @self.socketio.on('client_eval')
        def handle_client_eval(data):
        	#data
        	#  test_size
            #  test_loss
            #  test_accuracy
            if self.eval_client_updates is None:
                return
            print("handle client_eval", request.sid)
            print("eval_resp", data)
            self.eval_client_updates += [data]

            if len(self.eval_client_updates) > FLServer.NUM_CLIENTS_CONTACTED_PER_ROUND * .7:
                aggr_test_loss = sum([x['test_loss'] for x in self.eval_client_updates]) / len(self.eval_client_updates)
                aggr_test_accuracy = sum([x['test_accuracy'] for x in self.eval_client_updates]) / len(self.eval_client_updates)
                print("\naggr_test_loss", aggr_test_loss)
                print("aggr_test_accuracy", aggr_test_accuracy)
                print("== done ==")
                self.eval_client_updates = None  # special value, forbid evaling again

    
    def train_next_round(self):
        self.current_round += 1
        self.current_round_client_updates = []

        print("### Round ", self.current_round, "###")
        client_sids_selected = random.sample(list(self.ready_client_sids), FLServer.NUM_CLIENTS_CONTACTED_PER_ROUND)
        print("request updates from", client_sids_selected)

        for rid in client_sids_selected:
            emit('request_update', {
                    'model_id': self.model_id,
                    'round_number': self.current_round,
                    'current_model': to_pickle(self.global_model),
                    'model_format': 'pickle',
                }, room=rid)

    
    def stop_and_eval(self):
        self.eval_client_updates = []
        for rid in self.ready_client_sids:
            emit('stop_and_eval', {
                    'model_id': self.model_id,
                    'current_model': to_pickle(self.global_model),
                    'model_format': 'pickle'
                }, room=rid)

    def start(self):
        logging.getLogger('socketio').setLevel(logging.ERROR)
        logging.getLogger('engineio').setLevel(logging.ERROR)
        self.socketio.run(self.app, host=self.host, port=self.port)



if __name__ == '__main__':
    # When the application is in debug mode the Werkzeug development server is still used
    # and configured properly inside socketio.run(). In production mode the eventlet web server
    # is used if available, else the gevent web server is used.
    
    server = FLServer("127.0.0.1", 5000)
    print("listening on 127.0.0.1:5000");
    server.start()