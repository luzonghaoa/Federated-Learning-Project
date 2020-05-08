import grpc
from concurrent import futures
import time

import FederatedML_pb2
import FederatedML_pb2_grpc
import functions


class FederatedMLServicer(FederatedML_pb2_grpc.FederatedMLServicer):
    def GetModel(self, request, context):
        response = FederatedML_pb2.Model()
        response.model = functions.GetModel(request.op, request.gr)
        return response

    def SendModel(self, request, context):
        response = FederatedML_pb2.Empty()
        response.value = functions.SendModel(request.model, request.id, request.gr)
        return response

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
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
