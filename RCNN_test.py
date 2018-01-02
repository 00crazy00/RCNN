import numpy as np
import cv2
from keras import  backend as K
from keras.models import Sequential
from keras.layers.convolutional import  Conv2D,MaxPooling2D,Convolution2D
from keras.layers.core import Activation,Flatten,Dense
from keras.utils import np_utils
from keras.optimizers import SGD,RMSprop,Adam
from keras.models import load_model
import h5py
from keras.models import model_from_json

# 读取model
model = model_from_json(open('my_model_srcnn.json').read())
model.load_weights('my_model_srcnnw.h5')

#读取测试图片
test=cv2.imread("E:\\project\\RCNN\\1281.jpg")
low=np.asarray(test)
low = np.reshape(low, (1, 128, 128, 3))
low.astype('float32')
low=low/255

a=model.predict(low)
a=a*255
new=a[0,:,:,:]
cv2.imwrite("E:\\project\\RCNN\\new.jpg",new)