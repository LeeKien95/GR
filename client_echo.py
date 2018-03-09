# echo_client.py
import socket

class ServerData:
	def __init__(self, data):
		self.data = data

host = socket.gethostname()    
port = 12345                   # The same port as used by the server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((host, port))
s.sendall(b'Data from client')
data = s.recv(1024)
s.close()
received_data = ServerData(data)
print('Received', repr(received_data.data))