import socket
import time
import sys
import cv2
import imutils
from yoloDet import YoloTRT
import os
import threading
import pycuda.driver as cuda

#MAKE SURE TO IMPLEMENT THIS: nano /home/paradox/JetsonYolov5/yoloDet.py THEN import pycuda.autoinit 
model = YoloTRT(library="yolov5/build/libmyplugins.so", engine="yolov5/build/yolov5s.engine", conf=0.5, yolo_ver="v5")
# [{'class': 'Note', 'conf': 0.6934612, 'box': array([164.14363, 129.19603, 280.63977, 219.     ], dtype=float32)}]

cap = cv2.VideoCapture(0)
conn=None

port = 5800  # initiate port no above 1024
host_ip='10.21.2.86' #optimize so we can get IP from hostname
test_without_connection=True # testing without clients connecting to server
fps_verbose=True #prints out FPS

if test_without_connection:
    print('NOT SENDING DATA, TESTING WITHOUT HOSTS!')
else:
    print('PREPARING TO CONNECT WITH HOSTS!')
def predict():
    ret, frame = cap.read()
    if ret:
        frame = imutils.resize(frame, width=352,height=320)
        detections, t = model.Inference(frame)
        #print('DETECTIONS:',detections)
        # for obj in detections:
        #    print(obj['class'], obj['conf'], obj['box'])
        # Only show output if a display is available
        if 'DISPLAY' in os.environ:
            cv2.imshow("Output", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                #break
                pass
        else:
            print("No display available, skipping visualization.")
        boxes=[]
        if len(detections)>0:
            for i,v in enumerate(detections):
                box=v['box']
                boxes.append(box)
        
        return boxes
    else:
        print('camera not functioning')
        return 'camera not functioning'

print(f'hostname:{socket.gethostbyname(socket.gethostname())}')
server_socket = socket.socket()  # get instance
#host = socket.gethostname()
#host_ip = socket.gethostbyname(host)
if not test_without_connection:
    server_socket.bind((host_ip, port))  # bind host address and port together

def connect():
    #server_socket.bind((host_ip, port))  # bind host address and port together
    #binding is potentially part of connecting?
    print('listening for connetions...')
        
    #listens for up to 5 clients
    server_socket.listen(5)
    #makes conn and address accessible to other parts of the code
    global conn
    global address
    connect.conn, connect.address = server_socket.accept()  # accept new connection
    conn,address=connect.conn, connect.address
    print("Connection from: " + str(connect.address)) 

def send_data(data,conn,delay=0,verbose=False):
    conn.sendall(data.encode())  # send data to the client
    time.sleep(delay)
    if verbose:
        print('sent data')
        print(data)
def server_program():
    frame_counter = 1
    bounding_box_counter=0

    while True:
        if not test_without_connection:
            connect()
        global conn
        global address
        if not test_without_connection:
            conn= connect.conn
            address =  connect.address
        while True:
            try:
                bounding_boxes=predict()
                if bounding_boxes!='camera not functioning':
                    time1=time.time()
                    #preparing the string to be sent over to the robo rio
                    bounding_box_str=''
                        
                    #message indicating start
                    bounding_box_str+= f'''\nF {frame_counter} 352 320\n'''
    
                    #bounding box messages
    
                    for index,box in enumerate(bounding_boxes):
                        x1,y1,x2,y2=box
                        x1,y1,x2,y2 = int(round(x1)),int(round(y1)),int(round(x2)),int(round(y2))
                        bounding_box_str+=f'R {x1} {y1} {x2} {y2}\n'
                    
                    #message indicating end
                    bounding_box_str+=f'E\n'
                    print('bounding box str:',bounding_box_str)
                        
                    #Sending the data to robo rio
                    if not test_without_connection:
                        send_data(bounding_box_str,conn,verbose=True)
                        
                    frame_counter+=1
                    time2=time.time()
                    time_taken=time1-time2
                    if fps_verbose:
                        print('fps:',1/time_taken)

                else:
                    print('predictions not working because camera is not functioning')
            except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError,ConnectionError) as e:
                print('DISCONNECTED. attempting reconnecting...')
                break  
    #conn.close() (closes the connection, I'll potentially use this later)
    
def listen():
    while True:
        if conn and conn is not None:
            data = conn.recv(1024).decode()
            if not data:
                print('not recieving any data')
            else:
                print('data:',data)
prediction_thread = threading.Thread(target=server_program)
listening_thread = threading.Thread(target=listen)

print("TESTING!")
if __name__ == '__main__':
    prediction_thread.start()
    listening_thread.start()
