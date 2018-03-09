# echo_server.py
import socket

class ServerData:
	def __init__(self, data):
		self.data = data

host = ''        # Symbolic name meaning all available interfaces
port = 12345     # Arbitrary non-privileged port
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((host, port))
s.listen(1)
conn, addr = s.accept()
print('Connected by', addr)

while True:
    data = conn.recv(1024)
    if not data: break
    conn.sendall(b'ServerData("data from server")')
conn.close()