# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 23:13:41 2020

@author: karth
"""

#testing on a sample image
import os
import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt

#load model
model = model_from_json(open("models/fer.json", "r").read())
#load weights
model.load_weights('models/model.h5')

#croping the face from the given picture
def facecrop(image):  
    facedata = "haarcascade/haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(facedata)

    img = cv2.imread(image)

    try:
    
        minisize = (img.shape[1],img.shape[0])
        miniframe = cv2.resize(img, minisize)

        faces = cascade.detectMultiScale(miniframe)

        for f in faces:
            x, y, w, h = [ v for v in f ]
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

            sub_face = img[y:y+h, x:x+w]

            
            cv2.imwrite('capture.jpg', sub_face)

    except Exception as e:
        print (e)

#calling the function to extract face
facecrop('karthik.jpeg')

file='capture.jpg'
true_image = image.load_img(file)
img = image.load_img(file, color_mode = "grayscale", target_size=(48, 48))

x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)

x /= 255

objects = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
custom = model.predict(x)
custom=list(custom[0])
print(objects[custom.index(max(custom))])


#plotting the face
x = np.array(x, 'float32')
x = x.reshape([48, 48]);

plt.gray()
plt.imshow(true_image)
plt.show()
