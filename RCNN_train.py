
import numpy as np
import cv2
from keras import  backend as K
from keras.models import Sequential
from keras.layers.convolutional import  Conv2D,MaxPooling2D,Convolution2D
from keras.layers.core import Activation,Flatten,Dense
from keras.utils import np_utils
from keras.optimizers import SGD,RMSprop,Adam


#读取高分辨率图像

ipaths=['E:\\project\\RCNN\\img128high'+str(i)+'.jpg' for i in range(1,4127)] #输入你电脑上图片的路径
img128=[]
for i in range(4126):
    img=cv2.imread(ipaths[i])
    img=np.asarray(img)
    img128.append(img)
high=np.array(img128)
high = np.reshape(high, (4126, 128, 128, 3))
high.astype('float32')
high=high/255
high=high[:4000,:,:,:]
#读取低分辨率图像
ipaths=['E:\\project\\RCNN\\img128low'+str(i)+'.jpg' for i in range(1,4127)] #输入你电脑上图片的路径
img128a=[]
for i in range(4126):
    img=cv2.imread(ipaths[i])
    img=np.asarray(img)
    img128a.append(img)
low=np.array(img128a)
low = np.reshape(low, (4126, 128, 128, 3))
low.astype('float32')
low=low/255
low=low[:4000,:,:,:]

class SRCNNRGB:
    @staticmethod
    def build(input_shape):
        model=Sequential()
        #CON => RELU
        model.add(Convolution2D(6,kernel_size=3,padding="same",input_shape=input_shape))
        model.add(Activation("relu"))
        # CON => RELU
        model.add(Convolution2D(6,kernel_size=1,padding="same"))
        model.add(Activation("relu"))
        model.add(Convolution2D(3,kernel_size=3,padding="same"))
        return model
input_size=(128,128,3)
K.set_image_dim_ordering("tf")
model=SRCNNRGB.build(input_shape=input_size)
model.compile(loss="mean_squared_error",optimizer=Adam())
history=model.fit(low,high,batch_size=1,epochs=10,verbose=1) #参数根据你显卡的性能调，我的显卡gtx850m

#储存训练好的模型
import h5py
json_string = model.to_json()#等价于 json_string = model.get_config()
open('my_model_srcnn.json','w').write(json_string)
model.save_weights('my_model_srcnnw.h5')