import socket
import time #used to calc FPS
import threading
import queue
import cv2
import imutils #resizing images
from yoloDet import YoloTRT #predicting
import pycuda.driver as cuda

#preparing cuda GPU
cuda.init() 
device = cuda.Device(0)
ctx = device.make_context()

host_ip = ''  # Accept connections on any interface
port = 5800
backlog = 5
capture_ready=False #Model won't inference until this value is True which is when camera is ready

model = YoloTRT(library="yolov5/build/libmyplugins.so", engine="yolov5/build/yolov5s.engine", conf=0.5, yolo_ver="v5")
class Client:
    """Lightweight class to store client socket and lock for thread-safety."""
    def __init__(self, socket, addr):
        self.socket = socket
        self.addr = addr
        self.lock = threading.Lock()  # Lock to ensure thread-safety of send and recv calls

class Server:
    """Server class to accept connections, handle clients, and broadcast data to clients."""
    def __init__(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((host_ip, port))
        self.server_socket.listen(backlog)
        self.video_capture = cv2.VideoCapture(0)
        print(f"[*] Listening on {host_ip}:{port}")
        self.data_queue = queue.Queue(maxsize=1)  # Only store the latest data
        self.data_available = threading.Condition()  # Condition variable to signal when new data is available
        self.clients=[]
        self.frame_counter=1
    def start(self):
        """Start the server and begin accepting connections."""
        threading.Thread(target=self.accept_connections).start()
        threading.Thread(target=self.capture_images).start()
        threading.Thread(target=self.image_processing).start()
        threading.Thread(target=self.broadcast_data).start()
    def accept_connections(self):
        """Accept incoming connections and spawn a new thread for each client."""
        while True:
            client_socket, addr = self.server_socket.accept()
            print(f"[*] Accepted connection from {addr[0]}:{addr[1]}")
            client = Client(client_socket, addr)
            self.clients.append(client)
            threading.Thread(target=self.handle_client, args=(client,)).start()
            print("CLIENTS:", self.clients)
    def capture_images(self):
        while True:
            global capture_ready
            global latest_image
            capture_ready,latest_image = self.video_capture.read()
            latest_image  = imutils.resize(latest_image, width=640,height=640)
            self.frame_counter+=1
    def image_processing(self):
        """Do image processing and send data."""
        while True:
            if capture_ready:
                ctx.push() #making context
                time1=time.time()
                detections, t = model.Inference(latest_image)
                time2=time.time()
                time_taken=time2-time1
                fps=1/time_taken
                ctx.pop() #clearing the context
                print('fps:',fps)
                bounding_boxes=[]
                if len(detections)>0:
                    for i,v in enumerate(detections):
                        box=v['box']
                        bounding_boxes.append(box)
                bounding_box_str=''
                #message indicating start
                bounding_box_str+= f'''\nF {self.frame_counter} 352 320\n'''

                #bounding box messages

                for index,box in enumerate(bounding_boxes):
                    x1,y1,x2,y2=box
                    x1,y1,x2,y2 = int(round(x1)),int(round(y1)),int(round(x2)),int(round(y2))
                    bounding_box_str+=f'R {x1} {y1} {x2} {y2}\n'
                bounding_box_str+=f'E\n'
                data=bounding_box_str
                print('data:',data)
                self.send_data(data)
            else:
                print('camera not functioning')

    def send_data(self, data):
        """Send data to all connected clients."""
        if self.data_queue.full():
            self.data_queue.get()
        self.data_queue.put(data)
        with self.data_available:
            self.data_available.notify_all()  # Signal that new data is available

    def handle_client(self, client):
        """Handle data received on a client connection."""
        while True:
            with client.lock: # Don't allow other threads to send data while we're receiving
                data = client.socket.recv(1024)
                if not data:
                    print('not receiving any data')
                else:
                    print('data:',data, client.addr)

    def broadcast_data(self):
        """Broadcast data to all connected clients."""
        while True:
            with self.data_available:
                self.data_available.wait()  # Wait for new data to be available
            data = self.data_queue.get()
            for client in self.clients:
                print('broadcasting...', client.addr)
                try:
                    #with client.lock: # Don't allow other threads to receive data while we're sending
                    client.socket.sendall(data.encode()) #changed from send to sendall
                except Exception as e:
                    print("Removing client", client.addr, e)
                    self.clients.remove(client) 

if __name__ == "__main__":
    server = Server()
    server.start()
