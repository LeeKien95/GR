# echo_server.py
import socket
import simplejson as json
import pickle
import ProcessLandmark as pl


host = ''        # Symbolic name meaning all available interfaces
port = 12345     # Arbitrary non-privileged port
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((host, port))
s.listen(1)
conn, addr = s.accept()
print('Connected by', addr)

while True:
    data = conn.recv(20480)
    if not data: break
    received_data = json.loads(data)
    # print(len(received_data['landmarkChange']))
    echo_data = pl.getEmotionPredict(received_data['landmarkChange'])
    conn.sendall(json.dumps(echo_data).encode())
conn.close()