import grpc
from concurrent import futures
import time
import math

import FederatedML_pb2
import FederatedML_pb2_grpc
import functions

M_M_SIZE = 1000 * 1024 * 1024
MESSAGE_BLOCK_SIZE = 1024 * 1024

class FederatedMLServicer(FederatedML_pb2_grpc.FederatedMLServicer):
    def GetModel(self, request, context):
        response = FederatedML_pb2.Model()
        response.model = functions.GetModel(request.op, request.id, request.gr)
        return response

    def SendModel(self, request, context):
        response = FederatedML_pb2.Empty()
        response.value = functions.SendModel(request.model, request.id, request.gr, request.ll)
        return response

    def GetModel_stream(self, request, context):
        response = FederatedML_pb2.Model()
        encoing_string = functions.GetModel(request.op, request.id, request.gr)
        blocks = math.ceil(len(encoing_string) / MESSAGE_BLOCK_SIZE)
        for i in range(blocks):
            response.model = encoing_string[i*MESSAGE_BLOCK_SIZE:(i+1)*MESSAGE_BLOCK_SIZE]
            yield response

    def SendModel_stream(self, request_it, context):
        response = FederatedML_pb2.Empty()
        response.value = functions.SendModel_stream(request_it)
        return response
    
    def Regis(self, request, context):
        response = FederatedML_pb2.Mark()
        response.flag = functions.Regis(request.flag)
        return response

    def GetReady(self, request, context):
        response = FederatedML_pb2.Mark()
        response.flag = functions.GetReady(request.id, request.gr)
        return response

def serve():
    options = [('grpc.max_send_message_length', M_M_SIZE), ('grpc.max_receive_message_length', M_M_SIZE)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=20), options=options)
    FederatedML_pb2_grpc.add_FederatedMLServicer_to_server(
        FederatedMLServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("running now")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()

    '''
    try:
        while True:
            time.sleep(100)
    except KeyboardInterrupt:
        server.stop(0)
    '''
