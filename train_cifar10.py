import model_cifar10
import numpy as np
from keras.datasets import cifar10
import keras
import numpy as np
from keras.optimizers import Adam
from keras.layers import Input,Conv2D,MaxPooling2D,Flatten,Dense,Reshape,Deconv2D,UpSampling2D
from keras.layers.merge import Concatenate,add
from keras.layers.merge import Multiply
from keras.models import Model
import random
import PIL.Image as pI
from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator




def get_Gauss(mode,batch_size=32):
    if mode not in ['vector','pic']:
        raise Exception('mode error ,please choose a mode in vector and pic')
    elif mode=='vector':
        batch_vec=np.random.randn(batch_size, 1, 1, 64)
        return batch_vec
    else:
        batch_pic=np.random.randn(batch_size,32,32,1)
        return batch_pic

batch_size=100





(x_train,y_train),(x_test,y_test)=  cifar10.load_data()


#x_train=np.concatenate([x_train,x_test])
x_train=x_train/255
np.random.shuffle(x_train)

print(y_train.shape)
print(y_train[3,])
#这竟然不是原地操作，必须返回值

y_train=np.asarray([1.0, ]*50000)
for i in range(5000):
    y_train[i, ]=i+1
print(y_train[0:10, ])


#print(y_train[3,])
image_generator=ImageDataGenerator()
image_label=image_generator.flow(x_train,y_train,batch_size=3,shuffle=True)






combine_e_dis,combine_de_dis,modelencodor,modeldecoder,modeldiscriminator=model_cifar10.build_Model()
combine_e_dis.compile(optimizer=Adam(lr=0.0001,	beta_1=0.5,	beta_2=0.999),loss='binary_crossentropy',metrics=['accuracy'])
combine_de_dis.compile(optimizer=Adam(lr=0.0001,beta_1=0.5,	beta_2=0.999),loss='binary_crossentropy',metrics=['accuracy'])

print(combine_e_dis.summary(),combine_de_dis.summary())




#3237500
for i in range(3):
    print(i)
    ima,lab=image_label.next()
    modelencodor.trainable=False
    modeldecoder.trainable=False
    combine_e_dis.fit(x=[ima,get_Gauss('vector',batch_size=batch_size)],y=lab,batch_size=batch_size,epochs=1,shuffle=False)
    ima, lab = image_label.next()
    combine_e_dis.fit(x=[ima, get_Gauss('vector', batch_size=batch_size)], y=lab, batch_size=batch_size, epochs=1,
                      shuffle=False)
    ima, lab = image_label.next()
    combine_e_dis.fit(x=[ima, get_Gauss('vector', batch_size=batch_size)], y=lab, batch_size=batch_size, epochs=1,
                      shuffle=False)
    lab=lab-1
    combine_de_dis.fit(x=get_Gauss('vector',batch_size=batch_size),y=lab,batch_size=batch_size,epochs=1,shuffle=False)
    combine_de_dis.fit(x=get_Gauss('vector', batch_size=batch_size), y=lab, batch_size=batch_size, epochs=1,
                       shuffle=False)
    combine_de_dis.fit(x=get_Gauss('vector', batch_size=batch_size), y=lab, batch_size=batch_size, epochs=1,
                       shuffle=False)


    ima, lab = image_label.next()
    modelencodor.trainable=True
    modeldiscriminator.trainable=False
    combine_e_dis.fit(x=[ima, get_Gauss('vector', batch_size=batch_size)], y=lab-1, batch_size=batch_size, epochs=1,shuffle=False)
    ima, lab = image_label.next()
    combine_e_dis.fit(x=[ima, get_Gauss('vector', batch_size=batch_size)], y=lab - 1, batch_size=batch_size, epochs=1,
                      shuffle=False)
    ima, lab = image_label.next()
    combine_e_dis.fit(x=[ima, get_Gauss('vector', batch_size=batch_size)], y=lab - 1, batch_size=batch_size, epochs=1,
                      shuffle=False)


    modeldecoder.trainable=True
    combine_de_dis.fit(x=get_Gauss('vector', batch_size=batch_size), y=lab, batch_size=batch_size, epochs=1,shuffle=False)
    combine_de_dis.fit(x=get_Gauss('vector', batch_size=batch_size), y=lab, batch_size=batch_size, epochs=1,
                       shuffle=False)
    combine_de_dis.fit(x=get_Gauss('vector', batch_size=batch_size), y=lab, batch_size=batch_size, epochs=1,shuffle=False)
    combine_de_dis.fit(x=get_Gauss('vector', batch_size=batch_size), y=lab, batch_size=batch_size, epochs=1,
                       shuffle=False)
    modeldiscriminator.trainable=True

    if i%2==0:
        x=modeldecoder.predict(get_Gauss('vector',1))
        x=np.array(x)*255
        x = np.reshape(x, (32,32,3))

        x=np.array(x,dtype=np.int32)

        ima=pI.fromarray(x,'RGB')
        ima.save('./cifar.png',"png")
        ima.close()





