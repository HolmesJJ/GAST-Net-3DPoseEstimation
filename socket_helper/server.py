import socket
from threading import Thread
from queue import Queue
import json
import time
import datetime

from base.singleton import Singleton


def producer(out_q):
    while True:
        # Produce some data
        out_q.put("Test Data " + str(datetime.datetime.utcnow()))
        time.sleep(1)


class SocketServerHelper(object):
    __metaclass__ = Singleton

    ADDRESS = ('127.0.0.1', 8712)  # 绑定地址

    g_socket_server = None  # 负责监听的socket
    g_conn_pool = {}  # 连接池

    def __init__(self):
        """
        初始化服务端
        """
        self.g_socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.g_socket_server.bind(self.ADDRESS)
        self.g_socket_server.listen(5)  # 最大等待数（有很多人理解为最大连接数，其实是错误的）
        print("server start，wait for client connecting...")

    def accept_client(self, queue):
        """
        接收新连接
        """
        while True:
            if self.g_socket_server is not None:
                client, info = self.g_socket_server.accept()  # 阻塞，等待客户端连接
                if client is not None:
                    jd = {'code': 0}
                    json_str = json.dumps(jd)
                    client.sendall(json_str.encode(encoding='utf8'))
                # 等待连接信号
                self.connect(client, info, queue)

    def connect(self, client, info, queue):
        """
        消息处理
        """
        try:
            data = client.recv(1024)
            msg = data.decode(encoding='utf8')
            jd = json.loads(msg)
            cmd = jd['COMMAND']
            client_id = jd['client_id']
            if 'CONNECT' == cmd:
                self.g_conn_pool[client_id] = client
                # 给每个客户端创建两个独立的线程进行管理
                self.receive(client_id)
                self.send(client_id, queue)
                print('client connect: ', info)
            else:
                self.release(client_id)
        except Exception as e:
            print("connect exception: " + str(e))

    def release(self, client_id):
        """
        释放客户端连接
        """
        try:
            client = self.g_conn_pool[client_id]
            if client is not None:
                client.close()
                self.g_conn_pool.pop(client_id)
                print("client disconnect")
        except Exception as e:
            print("release exception: " + str(e))

    def send(self, client_id, queue):
        """
        发送数据线程
        """
        try:
            client = self.g_conn_pool[client_id]
            if client is not None:
                send_thread = Thread(target=self.send_message_handle, args=(client_id, queue))
                # 设置成守护线程
                send_thread.setDaemon(True)
                send_thread.start()
                print("client send")
        except Exception as e:
            self.release(client_id)
            print("send exception: " + str(e))

    def receive(self, client_id):
        """
        接收数据线程
        """
        try:
            client = self.g_conn_pool[client_id]
            if client is not None:
                receive_thread = Thread(target=self.receive_message_handle, args=(client_id,))
                # 设置成守护线程
                receive_thread.setDaemon(True)
                receive_thread.start()
                print("client receive")
        except Exception as e:
            self.release(client_id)
            print("receive exception: " + str(e))

    def send_message_handle(self, client_id, queue):
        """
        发送消息处理
        """
        while True:
            try:
                client = self.g_conn_pool[client_id]
                if client is not None:
                    data = queue.get()
                    jd = {'code': 0, 'data': str(data)}
                    json_str = json.dumps(jd)
                    client.sendall(json_str.encode(encoding='utf8'))
                    print("client send message")
            except Exception as e:
                self.release(client_id)
                print("send message exception: " + str(e))
                break

    def send_all_message_handle(self, queue):
        """
        发送消息处理
        """
        while True:
            for client_id, client in self.g_conn_pool.items():
                try:
                    if client is not None:
                        data = queue.get()
                        jd = {'code': 0, 'data': str(data)}
                        json_str = json.dumps(jd)
                        client.sendall(json_str.encode(encoding='utf8'))
                        print("client send message")
                except Exception as e:
                    self.release(client_id)
                    print("send message exception: " + str(e))

    def receive_message_handle(self, client_id):
        """
        接收消息处理
        """
        while True:
            try:
                client = self.g_conn_pool[client_id]
                if client is not None:
                    data = client.recv(1024)
                    msg = data.decode(encoding='utf8')
                    jd = json.loads(msg)
                    cmd = jd['COMMAND']
                    if 'SEND_DATA' == cmd:
                        print('client message: ', jd['data'])
                    else:
                        self.release(client)
            except Exception as e:
                self.release(client_id)
                print("receive message exception: " + str(e))
                break


if __name__ == '__main__':
    data_queue = Queue(maxsize=1)
    test_thread = Thread(target=producer, args=(data_queue,))
    test_thread.start()
    socket_server = SocketServerHelper()
    # 新开一个线程，用于接收新连接
    socket_server_thread = Thread(target=socket_server.accept_client, args=(data_queue,))
    socket_server_thread.setDaemon(True)
    socket_server_thread.start()
    # 主线程逻辑
    while True:
        time.sleep(0.1)
