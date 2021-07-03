import socket
import json

ADDRESS = ('127.0.0.1', 8712)
CLIENT_ID = 'client_1'


def send_data(client, cmd, **kv):
    jd = {'COMMAND': cmd, 'client_id': CLIENT_ID, 'data': kv}
    json_str = json.dumps(jd)
    print('send: ' + json_str)
    client.sendall(json_str.encode('utf8'))


if __name__ == '__main__':
    socket_client = socket.socket()
    socket_client.connect(ADDRESS)
    print(socket_client.recv(1024).decode(encoding='utf8'))
    send_data(socket_client, 'CONNECT')

    while True:
        print(socket_client.recv(1024).decode(encoding='utf8'))
