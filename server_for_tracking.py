print("STARTED!!!!")
import socket
import time
import sys
import cv2
import imutils
from yoloDet import YoloTRT
import os


model = YoloTRT(library="yolov5/build/libmyplugins.so", engine="yolov5/build/yolov5s.engine", conf=0.5, yolo_ver="v5")
# [{'class': 'Note', 'conf': 0.6934612, 'box': array([164.14363, 129.19603, 280.63977, 219.     ], dtype=float32)}]
cap = cv2.VideoCapture(0)
print('captured')

def predict():
        ret, frame = cap.read()
        print('ret:',ret)
        frame = imutils.resize(frame, width=352,height=320)
        detections, t = model.Inference(frame)
        #print('DETECTIONS:',detections)
        # for obj in detections:
        #    print(obj['class'], obj['conf'], obj['box'])
        print("FPS: {} sec".format(1/t))
        print('predicted in',t)
        # Only show output if a display is available
        if 'DISPLAY' in os.environ:
                cv2.imshow("Output", frame)
                key = cv2.waitKey(1)
                if key == ord('q'):
                        #break
                        pass
        else:
                pass
                #print("No display available, skipping visualization.")
        boxes=[]
        if len(detections)>0:
                for i,v in enumerate(detections):
                        box=v['box']
                        boxes.append(box)
        
        return boxes

print(f'hostname:{socket.gethostbyname(socket.gethostname())}')
server_socket = socket.socket()  # get instance
host = socket.gethostname()
print('host:',host)
host_ip = socket.gethostbyname(host)
port = 5800  # initiate port no above 1024
print('host:',host_ip)
server_socket.bind((host_ip, port))  # bind host address and port together
def connect():
    #server_socket.bind((host_ip, port))  # bind host address and port together
    #binding is potentially part of connecting?
    print('listening for connetions...')
    server_socket.listen(5)
    connect.conn, connect.address = server_socket.accept()  # accept new connection
    print("Connection from: " + str(connect.address)) 

def send_data(data,conn,delay=0,verbose=False):
    conn.sendall(data.encode())  # send data to the client
    time.sleep(delay)
    if verbose:
        print('sent data')
        print(data)
def server_program():
    # get the hostname
    # look closely. The bind() function takes tuple as argument

    # configure how many client the server can listen simultaneously
    frame_counter = 1
    bounding_box_counter=0
    while True:
        #connect()
        #conn= connect.conn
        #address =  connect.address
        while True:
            try:
                bounding_boxes=predict()

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
                #send_data(bounding_box_str,conn,delay=3,verbose=True)
                frame_counter+=1
            except BrokenPipeError:
                print('DISCONNECTED. attempting reconnecting...')
                break
        
    conn.close()  # close the connection
    

print("TESTING!")
if __name__ == '__main__':

    server_program()
