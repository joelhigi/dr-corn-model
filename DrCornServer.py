from cv2 import *
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import socket
import struct
import time
from datetime import datetime

listensocket = socket.socket()
Port = 8800
maxConnections = 10
IP = socket.gethostname()

listensocket.bind(('',Port))

listensocket.listen(maxConnections)
print("Server started at " + IP + " on port " + str(Port))

serverLoop = True

while serverLoop:

    (clientsocket, address) = listensocket.accept()
    #clientsocket.sendall("hey".encode())
    print("New connection made!")

    running = True

    while running:
        directory = 'F:\py_mac_learn\plant disease\images\\'
        dt = datetime.now()
        ts = datetime.timestamp(dt)
        ts = str(ts)
        filetype = '.jpg'
        file_dir = directory + ts + filetype
        message = clientsocket.recv(7)
        message = message.decode()
        print(message)
        
        if message=="TakePic":
            cam = VideoCapture(0)
            s, img = cam.read()
            if s:
                imwrite(file_dir,img)
            cam.release()
                 
        else:
            buf=b''
            while len(buf) < 4:
                buf+=clientsocket.recv(4-len(buf))
            size = struct.unpack('!i',buf)[0]
            print(size)
            with open(file_dir, 'wb') as f:
                while size>0:
                    data = clientsocket.recv(1024)
                    if not data:
                        f.close()
                        break
                    f.write(data)
                    size-=len(data)
                                
        print('Image Saved')

        
        model = load_model('model')
        image_result=Image.open(file_dir)
        file_dir=image.load_img(file_dir,target_size=(224,224))
        file_dir=image.img_to_array(file_dir)
        file_dir=file_dir/255
        file_dir=np.expand_dims(file_dir,axis=0 )
        result=model.predict(file_dir)
        print(result)
        
        categories=['Blight','Common_Rust','Gray_Leaf_Spot','Healthy']
        image_result=plt.imshow(image_result)
        status = categories[np.argmax(result)]
        pred = np.amax(result)
        plt.title(status)
        print(status)
        print()
        intpred = int(pred*10000)
        stringpred = str(intpred)
        extra = '\r\n'
        final_message = stringpred+extra
        print(final_message)
        #plt.show()
        clientsocket.send(final_message.encode("utf-8"))
        clientsocket.send(status.encode("utf-8"))
        clientsocket.shutdown(socket.SHUT_RDWR)
        clientsocket.close()
        running = False
        
