import asyncore
import socket
import ProcessLandmark as pl
import simplejson as json

class EchoHandler(asyncore.dispatcher_with_send):

    def handle_read(self):
        data = self.recv(20480)
        if data:
            received_data = json.loads(data)
            echo_data = pl.getEmotionPredict(received_data['landmarkChange'])
            self.send(json.dumps(echo_data).encode())

class EchoServer(asyncore.dispatcher):

    def __init__(self, host, port):
        asyncore.dispatcher.__init__(self)
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        self.set_reuse_addr()
        self.bind((host, port))
        self.listen(5)

    def handle_accept(self):
        pair = self.accept()
        if pair is not None:
            sock, addr = pair
            print('Incoming connection from', repr(addr))
            handler = EchoHandler(sock)

host = ''        # Symbolic name meaning all available interfaces
port = 12345     # Arbitrary non-privileged port
server = EchoServer(host, port)
asyncore.loop()