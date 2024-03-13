import socket
import time #used to calc FPS
import threading
import queue
import cv2
import imutils #resizing images
from yoloDet import YoloTRT #predicting
import pycuda.driver as cuda
import os

#preparing cuda GPU
cuda.init() 
device = cuda.Device(0)
ctx = device.make_context()

host_ip = ''  # Accept connections on any interface
port = 5800
backlog = 5
capture_ready=False #Model won't inference until this value is True which is when camera is ready
latest_image=None
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
        try:
            self.server_socket.bind((host_ip, port))
        except:
            #TODO: add exception for address already in use
            print('address already in use. Killing all python processes, restart this code in order for it to run again')
            os.system('sudo killall -9 python3')

        self.server_socket.listen(backlog)

        #TODO: CAPTURE IN 352x320
        self.video_capture = cv2.VideoCapture(0)
        #self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH,352)
        #self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT,320)

        #TODO: MAKE IT SO DRIVERSTATION CAN ACCESS CAPTURED IMAGES USING KEY
        #self.key = 0xaa55aa55 #2857740885 as int

        self.capture_ready=False
        self.latest_image=None
        self.latest_image_time=None
        print(f"[*] Listening on {host_ip}:{port}")
        self.data_queue = queue.Queue(maxsize=1)  # Only store the latest data
        self.data_available = threading.Condition()  # Condition variable to signal when new data is available
        self.clients=[]
        self.frame_counter=1

    #TODO: WORK ON THIS FUNCTION LATER
    def send_key_width_height_nbytes_jpg(self):
        while True:
            if self.latest_image is not None:
                latest_image = self.latest_image # locking onto latest image by storing it as variable
                if latest_image.shape==(320,352,3):
                    #convert key to 32 bit first
                    self.send_data(key,352,320,self.latest_image.size,latest_image)  #send as binary. 352 and 320 represent width and height
                    time.sleep(.1) #sending every 100 milliseconds

    
    def start(self):
        """Start the server and begin accepting connections."""
        threading.Thread(target=self.accept_connections).start()
        threading.Thread(target=self.capture_images).start()
        threading.Thread(target=self.image_processing).start()
        threading.Thread(target=self.broadcast_data).start()
        #threading.Thread(target=self.send_key_width_height_nbytes_jpg).start()
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
            self.capture_ready,self.latest_image = self.video_capture.read()
            self.latest_image_time=time.time()
            #print('capture ready:',self.capture_ready)
            if self.capture_ready:
                self.latest_image = cv2.resize(self.latest_image,(352,320))
            else:
                print('not capture ready. FIX CAMERA!')
                self.video_capture = cv2.VideoCapture(0)
    def image_processing(self):
        """Do image processing and send data."""
        while True:
            if self.capture_ready and self.latest_image is not None:
                if self.latest_image.shape==(320,352,3) and (time.time()-self.latest_image_time)<.08:
                    time1=time.time()
                    ctx.push() #making context
                    detections, t = model.Inference(self.latest_image)

                    #displaying image is display is avalible
                    if 'DISPLAY' in os.environ:
                        if self.latest_image.shape==(320,352,3):
                            cv2.imshow("Output", self.latest_image)
                            key = cv2.waitKey(1)
                            if key == ord('q'):
                                break


                    ctx.pop() #clearing the context
                    self.frame_counter+=1
                    time2=time.time()
                    time_taken=time2-time1
                    fps=1/time_taken
                    print('fps:',fps)
                    for i in detections:
                        print(i)
                    bounding_boxes=[i['box'] for i in detections]
                    #print('bounding_boxes:',bounding_boxes)
                    bounding_box_str=''
                    #message indicating start
                    bounding_box_str+= f'''\nF {self.frame_counter} 352 320\n'''

                    #bounding box messages

                    for index,box in enumerate(bounding_boxes):
                        print('box:',box)
                        x1,y1,x2,y2=box
                        bounding_box_str+=f'R {x1} {y1} {x2} {y2}\n'
                    bounding_box_str+=f'E\n'
                    data=bounding_box_str
                    print('data:',data)
                    #print('detections:',detections)
                    self.send_data(data)
                elif not (time.time()-self.latest_image_time)<.08:
                    print('IMAGES TOO OLD, camera not functioning')
                    #print(self.capture_ready)
                    #if self.latest_image is not None:
                        #print('SHAPE:',self.latest_image.shape)
                        #continue

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
