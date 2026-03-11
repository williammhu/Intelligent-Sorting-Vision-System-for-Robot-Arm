import socket
import messageQueue


class Client:
    def __init__(self):
        self.port = 5000  
        self.ip = ''
        self.connect_flag = False
        self.data_queue = messageQueue.MessageQueue() 
        self.data_queue.clear()                      

    def connect(self, ip):
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)   
            self.client_socket.settimeout(3.0)                                     
            self.client_socket.connect((ip, int(self.port)))                        
            self.client_socket.settimeout(None)                                      
            self.connect_flag = True                                                 
        except:
            try:
                self.client_socket.close()
            except:
                pass
            self.connect_flag = False                                               
        return self.connect_flag                                                 

    def disconnect(self):
        self.connect_flag = False
        try:
            self.client_socket.shutdown(socket.SHUT_RDWR)
        except:
            pass
        try:
            self.client_socket.close()
        except:
            pass

    def send_messages(self, data):
        try:
            if self.connect_flag:
                self.client_socket.sendall(data.encode('utf-8'))
        except:
            print("Client send data failed.")

    def receive_messages(self, stop_event=None, on_data=None, timeout_s=0.2):
        previous_timeout = None
        try:
            previous_timeout = self.client_socket.gettimeout()
        except:
            previous_timeout = None
        try:
            self.client_socket.settimeout(timeout_s)
        except:
            pass

        while self.connect_flag:
            if stop_event is not None and stop_event.is_set():
                break
            try:
                data = self.client_socket.recv(1024).decode('utf-8')
                if data != '':
                    if on_data is None:
                        self.data_queue.put(data)
                    else:
                        on_data(data)
                else:
                    self.connect_flag = False
                    break
            except socket.timeout:
                continue
            except:
                self.connect_flag = False
                break

        try:
            self.client_socket.settimeout(previous_timeout)
        except:
            pass


if __name__ == '__main__':
    wifi = Client()
    wifi.connect("192.168.1.139")
    wifi.send_messages("Hello world.")
    while True:
        if wifi.receive_messages()!= "":
            pass
        else:
            break
    wifi.disconnect()
    print("Close tcp.")
    pass









































