import logging
import os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import socket
import time #used to calc FPS
import threading
import queue
import cv2
import imutils #resizing images

# from yoloDet import YoloTRT #predicting
# import pycuda.driver as cuda
# import pycuda.autoinit

import struct
from datetime import datetime
import numpy as np

# cuda.init() #making context
# device = cuda.Device(0)

#preparing cuda GPU
# cuda.init() 
image_width = 640
image_height = 400
# device = cuda.Device(0)
# ctx = device.make_context()
# ctx.push()
radius = 7
camera_matrix = np.array(((6.2874914053271243e+02, 0.,  3.1950000000000000e+02,),(0.,
     6.2874914053271243e+02, 1.9950000000000000e+02), (0., 0., 1.)))
distortion_coefficients = np.array((-1.5434763501469506e-01, 7.2106771708519934e-01, 0., 0.,
    -9.9172780117959070e-01))
object_points = np.expand_dims(np.array(((0,0,radius),(-radius,0,0),(radius,0,0),(0,0,-radius))),axis=2).astype('float32')
print('object points shape:',np.shape(object_points))

host_ip = ''  # Accept connections on any interface
port = 5800
driver_station_port=5801

backlog = 5
capture_ready=False #Model won't inference until this value is True which is when camera is ready
latest_image=None
# model = YoloTRT(library="yolov5/build/libmyplugins.so", engine="yolov5/build/yolov5s.engine", conf=0.5, yolo_ver="v5")
def solve_pnp(x1,y1,x2,y2):
   image_points = np.expand_dims(np.array([(np.mean((x1,x2)),y1),(x1,np.mean((y1,y2))),(x2,np.mean((y1,y2))),(np.mean((x1,x2)),y2)]),axis=2).astype('float32')
   print('image points shape:',np.shape(image_points))
   rtval,rvec,tvec = cv2.solvePnP(object_points,image_points,camera_matrix,distortion_coefficients)
   if(rtval):
        return [rvec,tvec]
   else:
        return False
 
class Client:
    """Lightweight class to store client socket and lock for thread-safety."""
    def __init__(self, socket, addr):
        self.socket = socket
        self.addr = addr
        self.lock = threading.Lock()  # Lock to ensure thread-safety of send and recv calls
    def __repr__(self) -> str:
        return f"{self.addr}"


class Service:
    def __init__(self, port:int, handler):
        self.port = port
        self.handler = handler
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        binded = False
        while not binded:
            try:
                self.socket.bind((host_ip, self.port))
                break
            except:
                print(f'failed binding port {port}, trying again')


        self.socket.listen(backlog)
        print(f"[*] Listening on {host_ip}:{port}")
        self.clients = []
        self.data_queue = queue.Queue(maxsize=1)  # Only store the latest data
        self.data_available = threading.Condition()  # Condition variable to signal when new data is available

    def start(self):
        threading.Thread(target=self.accept_connections,daemon=True).start()
        threading.Thread(target=self.broadcast_data,daemon=True).start()
        return self

    def accept_connections(self):
        """Accept incoming connections and spawn a new thread for each client."""
        while True:
            client_socket, addr = self.socket.accept()
            print(f"[*] Accepted connection from {addr[0]}:{addr[1]}")
            client = Client(client_socket, addr)
            self.clients.append(client)
            print("CLIENTS:", self.clients)
            threading.Thread(target=self.handler, args=(client,),daemon=True).start()

    def send_data(self, data):
        """Send data to all connected clients."""
        if self.data_queue.full():
            self.data_queue.get()
        self.data_queue.put(data)
        with self.data_available:
            self.data_available.notify_all()  # Signal that new data is available

    def broadcast_data(self):
        """Broadcast data to all connected clients."""
        while True:
            with self.data_available:
                self.data_available.wait()  # Wait for new data to be available
            data = self.data_queue.get()
            if not type(data)==bytes:
                data=data.encode()
            print('number of clients:',len(self.clients),'port:',self.port)
            for client in self.clients:
                print('broadcasting...', client.addr,self.port)
                try:
                    #with client.lock: # Don't allow other threads to receive data while we're sending
                    client.socket.sendall(data) #changed from send to sendall
                except Exception as e:
                    print("Removing client", client.addr, e)
                    self.clients.remove(client)



class Server:
    """Server class to accept connections, handle clients, and broadcast data to clients."""
    def __init__(self):
        self.rio_service = Service(port, self.handle_client).start()
        self.ds_service = Service(driver_station_port, self.handle_client).start()

        self.video_capture = cv2.VideoCapture(0)

        self.key = 0xaa55aa55 #2857740885 as int
        self.latest_predicted_img=None
        self.capture_ready=False
        self.latest_image=None
        self.latest_image_time=time.time()
        self.frame_counter=1
    def send_image(self,key,width:int,height:int,img:bytes):
        #rint(width,height,len(img))
        #encode as a 4-byte integers in network byte order
        img_bytes=cv2.imencode('.jpg',img)[1].tobytes()
        print(type(img_bytes))
        data=struct.pack('!IIII',self.key,width,height,len(img_bytes))+img_bytes
        #print('data:',data)
        self.ds_service.send_data(data)

    def send_images_thread(self):
            print("trying to send image")
            if self.latest_predicted_img is not None:
                self.send_image(self.key,image_width,image_height,self.latest_predicted_img)
                # time.sleep(.1) # send every 100 milliseconds
                #self.send_image(self.key,image_width,image_height,self.latest_predicted_img)
                print("sent image")
            
            else:
                print("no latest image")


    def start(self):
        """Start the server and begin accepting connections."""
        # threading.Thread(target=self.capture_images).start()
        # threading.Thread(target=self.image_processing).start()
        # threading.Thread(target=self.send_images_thread).start()
        
        t = threading.Thread(target=self.worker,daemon=True)
        t.start()
        return t

    def capture_images(self):
        #while True:
            print("waiting for image...")
            self.capture_ready,self.latest_image = self.video_capture.read()
            print("got image",self.capture_ready)
            if self.capture_ready:
                self.latest_image = imutils.resize(self.latest_image,width=image_width)
                global image_height
                image_height = self.latest_image.shape[0]
                print('image height:',image_height)
                self.latest_image_time=time.time()
            else:
                time.sleep(.1)
                print('not capture ready. FIX CAMERA!')
                self.video_capture = cv2.VideoCapture(0)
    def worker(self):
        print("started worker")
        # ctx.push()
        # ctx = device.make_context()
        from yoloDet import YoloTRT #predicting
        global model
        model = YoloTRT(library="yolov5/build/libmyplugins.so", engine="yolov5/build/yolov5s.engine", conf=0.5, yolo_ver="v5")
        while True:
            self.capture_images()
            self.image_processing()
            self.send_images_thread()

    def image_processing(self):
            """Do image processing and send data."""
    
            img = self.latest_image
            tvecs_and_rvecs = []
            latest_image_time=self.latest_image_time

            if self.capture_ready and img is not None:
                if img.shape[1]==image_width and (time.time()-latest_image_time)<1:
                    time1=time.time()
                    clone_img = img.copy()
                    image_with_boxes = img.copy()
                    # print("about to push context")
                    starttimeInference = time.time()
                    print("time to copy images:",starttimeInference-time1)
                    # ctx.push() #making context
                    # print("pushed context")
                    detections, t = model.Inference(clone_img) #clone_img is the image with the bad bounding boxes drawn on it
                    print("detections and t:",detections,t)
                    # print("about to pop context")
                    # ctx.pop() #clearing the context
                    endTimeInference = time.time()
                    print("Time to inference:",endTimeInference-starttimeInference)
                    # print("popped context")
                    image_border_filter = 10 #the number of pixels the note has to be inside of the camera's view in order to be seen
                    bounding_boxes=[i['box'] for i in detections]
                    bounding_boxes = [i for i in bounding_boxes if i[0]>0+image_border_filter and i[1]>0+image_border_filter and i[2]<(image_width-image_border_filter) and i[3]<(image_height-image_border_filter)]
                    time3 = time.time()
                    for box in bounding_boxes:
                        # print(image_with_boxes,box[:2],box[2:])
                        box = [int(i) for i in box]
                        # print("box:",box)
                        
                        image_with_boxes = cv2.rectangle(image_with_boxes, tuple(box[:2]), tuple(box[2:]), (255,255,0), 8)
                    # x1,y1,x2,y2=box
                      #  if detections and x1>0 and y1>0 and x2<(image_width-1) and y2<(image_height-1)
                   
                    t1 = time.time()
                    print("drawing boxes time:",t1-time3)
                    if detections:
                        for box in bounding_boxes:
                            
                            bounding_box_info =  solve_pnp(*box)
                            if bounding_box_info: #checking if solvepnp failed
                                 rvec, tvec = bounding_box_info

                                 tvecs_and_rvecs.append([rvec,tvec])
                                #  print('tvec:',tvec)
                                #  font = cv2.FONT_HERSHEY_SIMPLEX
                                #  for i, value in enumerate(tvec):                     
                                     #cv2.putText(image_with_boxes,str(value), (50,20+i*20),font,1,(0,50,0))
                    t2 = time.time()
                    print("solve pnp time:",t2-t1)
                    self.latest_predicted_img=image_with_boxes
                    #displaying image is display is avalible
                    if 'DISPLAY' in os.environ:
                        # cv2.imshow("Output", img) without bounding boxes
                        cv2.imshow("Output", image_with_boxes) #with bounding boxes
                        key = cv2.waitKey(1)
                        if key == ord('s'): #press s to save latest image
                            timestr = datetime.utcnow().isoformat(timespec='milliseconds')
                            save_path = f'/home/paradox/JetsonYolov5/{timestr}.png'
                            cv2.imwrite(save_path,img)
                    
                    self.frame_counter+=1
                    time2=time.time()
                    time_taken=time2-time1
                    fps=1/time_taken
                    print('fps:',fps)
                    # for i in detections:
                    #     print(i)

                    #print('bounding_boxes:',bounding_boxes)
                    bounding_box_str=''
                    #message indicating start
                    bounding_box_str+= f'''\nF {self.frame_counter} {image_width} {image_height}\n'''

                    #bounding box messages

                    for index,box in enumerate(bounding_boxes):
                        #print('box:',box)
                        x1,y1,x2,y2=box
                        if detections and x1>0 and y1>0 and x2<(image_width-1) and y2<(image_height-1): #checking if bounding box is outside of image, and if there are detections in the first place
                            rvec,tvec = tvecs_and_rvecs[index]
                            tx,ty,tz = tvec[0],tvec[1],tvec[2]
                            bounding_box_str+=f'R {x1} {y1} {x2} {y2} {tx[0]} {ty[0]} {tz[0]}\n'
                    bounding_box_str+=f'E\n'
                    print(bounding_box_str)
                    data=bounding_box_str
                    #print('data:',data)
                    #print('detections:',detections)
                    enddtime = time.time()
                    print("image processing time:",enddtime-time1)
                    print("image processing fps:",1/(enddtime-time1))
                    self.rio_service.send_data(data)
                elif not (time.time()-latest_image_time)<1:
                    print('IMAGES TOO OLD, camera not functioning')

    def handle_client(self, client):
        """Handle data received on a client connection."""
        while True:
            with client.lock: # Don't allow other threads to send data while we're receiving
                data = client.socket.recv(1024)
                if not data:
                         #print('not receiving any data')
                    pass
                else:
                    print('data:',data, client.addr)


if __name__ == "__main__":
    server = Server()
    t = server.start()
    t.join()
