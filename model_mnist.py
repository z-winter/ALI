import keras
import numpy as np
from keras.layers import Input,Conv2D,MaxPooling2D,Flatten,Dense,Reshape,Deconv2D,UpSampling2D,Activation,BatchNormalization,advanced_activations
from keras.layers.merge import Concatenate,Add
from keras.layers.merge import Multiply
from keras.models import Model

import random

def build_Model():
    #编码器模型
    input=Input((28,28,1),name='image_input',dtype=np.float32)
    inpute = Input((8,))
    model1convlayer1=Conv2D(64,(3,3),strides=(1,1),padding="valid")(input)
    model1convlayer1=BatchNormalization(axis=-1)(model1convlayer1)
    model1convlayer1=advanced_activations.LeakyReLU()(model1convlayer1)
    model1convlayer1=MaxPooling2D()(model1convlayer1)
    model1convlayer2=Conv2D(64,(3,3),strides=(1,1),padding="valid")(model1convlayer1)
    model1convlayer2=BatchNormalization(axis=-1)(model1convlayer2)
    model1convlayer2 = advanced_activations.LeakyReLU()(model1convlayer2)
    model1convlayer2 = MaxPooling2D()(model1convlayer2)
    model1convlayer3 = Conv2D(16, (3,3), strides=(1, 1), padding="valid",)(model1convlayer2)
    model1convlayer3=BatchNormalization(axis=-1)(model1convlayer3)
    model1convlayer3 = advanced_activations.LeakyReLU()(model1convlayer3)
    model1convlayer3 = MaxPooling2D()(model1convlayer3)
    u=Flatten()(model1convlayer3)
    Du=Dense(16,activation='relu')(u)
    Du = Dense(8, activation='relu')(Du)
    model2convlayer1 = Conv2D(64, (3, 3), strides=(1, 1), padding="valid")(input)
    model2convlayer1 = BatchNormalization(axis=-1)(model2convlayer1)
    model2convlayer1 = advanced_activations.LeakyReLU()(model2convlayer1)
    model2convlayer1 = MaxPooling2D()(model2convlayer1)
    model2convlayer2 = Conv2D(64, (3, 3), strides=(1, 1), padding="valid")(model2convlayer1)
    model2convlayer2 = BatchNormalization(axis=-1)(model2convlayer2)
    model2convlayer2 = advanced_activations.LeakyReLU()(model2convlayer2)
    model2convlayer2 = MaxPooling2D()(model2convlayer2)
    model2convlayer3 = Conv2D(16, (3, 3), strides=(1, 1), padding="valid")(model2convlayer2)
    model2convlayer3 = BatchNormalization(axis=-1)(model2convlayer3)
    model2convlayer3 = advanced_activations.LeakyReLU()(model2convlayer3)
    model2convlayer3 = MaxPooling2D()(model2convlayer3)
    delta = Flatten()(model2convlayer3)
    Ddel = Dense(16,activation='relu')(delta)
    Ddel = Dense(8, activation='relu')(Ddel)
    #e.append(random.gauss(0,1))
    DdelM=Multiply()([Ddel,inpute])
    f=Add()([Du,DdelM])


    #解码器模型
    inputdecodor=Input((8,),name='inputdecoder')
    inputdecodor1=Dense(16,activation='relu')(inputdecodor)
    inputdee = Input((28,28,1))
    f4dec=Reshape((4,4,1))(inputdecodor1)
    deconvlayer1=Deconv2D(16,(3,3),padding='valid')(f4dec)
    deconvlayer1=BatchNormalization(axis=-1)(deconvlayer1)
    deconvlayer1 = advanced_activations.LeakyReLU()(deconvlayer1)
    deconvlayer2 = Deconv2D(64, (5, 5), padding='valid')(deconvlayer1)
    deconvlayer2=BatchNormalization(axis=-1)(deconvlayer2)
    deconvlayer2= advanced_activations.LeakyReLU()(deconvlayer2)
    deconvlayer3 = Deconv2D(64, (5, 5), padding='valid')(deconvlayer2)
    deconvlayer3=BatchNormalization(axis=-1)(deconvlayer3)
    deconvlayer3 = advanced_activations.LeakyReLU()(deconvlayer3)
    pic=UpSampling2D(size=(2,2))(deconvlayer3)
    pic1layeru=Conv2D(1,(2,2),padding='same',activation='sigmoid')(pic)
    inputdecodor2=Dense(16,activation='relu')(inputdecodor)
    f4dec2 = Reshape((4, 4, 1))(inputdecodor2)
    deconv2layer1 = Deconv2D(16, (3, 3), padding='valid')(f4dec2)
    deconv2layer1=BatchNormalization(axis=-1)(deconv2layer1)
    deconv2layer1 = advanced_activations.LeakyReLU()(deconv2layer1)
    deconv2layer2 = Deconv2D(64, (5, 5), padding='valid')(deconv2layer1)
    deconv2layer2=BatchNormalization(axis=-1)(deconv2layer2)
    deconv2layer2 = advanced_activations.LeakyReLU()(deconv2layer2)
    deconv2layer3 = Deconv2D(64, (5, 5), padding='valid')(deconv2layer2)
    deconv2layer3=BatchNormalization(axis=-1)(deconv2layer3)
    deconv2layer3 = advanced_activations.LeakyReLU()(deconv2layer3)
    pic2 = UpSampling2D(size=(2, 2))(deconv2layer3)
    pic1layerdelta = Conv2D(1, (2, 2), padding='same',activation='sigmoid')(pic2)

    pm=Multiply()([pic1layerdelta,inputdee])
    pic1layer=Add()([pm,pic1layeru])



    #判别器模型
    inputdisimag = Input((28, 28, 1), name='image_input_dis')
    inputdiscode=Input((8,),name='code')
    modeldconvlayer1 = Conv2D(64, (3, 3), strides=(1, 1), padding="valid")(inputdisimag)
    modeldconvlayer1=BatchNormalization(axis=-1)(modeldconvlayer1)
    modeldconvlayer1 = advanced_activations.LeakyReLU()(modeldconvlayer1)
    modeldconvlayer1 = MaxPooling2D()(modeldconvlayer1)
    modeldconvlayer2 = Conv2D(64, (3, 3), strides=(1, 1), padding="valid")(modeldconvlayer1)
    modeldconvlayer2=BatchNormalization(axis=-1)(modeldconvlayer2)
    modeldconvlayer2 = advanced_activations.LeakyReLU()(modeldconvlayer2)
    modeldconvlayer2 = MaxPooling2D()(modeldconvlayer2)
    modeldconvlayer3 = Conv2D(16, (3, 3), strides=(1, 1), padding="valid")(modeldconvlayer2)
    modeldconvlayer3=BatchNormalization(axis=-1)(modeldconvlayer3)
    modeldconvlayer3 = advanced_activations.LeakyReLU()(modeldconvlayer3)
    modeldconvlayer3 = MaxPooling2D()(modeldconvlayer3)
    imagc = Flatten()(modeldconvlayer3)
    Dimagc = Dense(16,activation='relu')(imagc)
    Dimagc = Dense(8, activation='relu')(Dimagc)
    Dimagc = advanced_activations.LeakyReLU()(Dimagc)
    con_imagc_code=Concatenate(axis=1)([Dimagc,inputdiscode])
    df1=Dense(16,activation='relu')(con_imagc_code)
    df1 = Dense(8, activation='relu')(df1)
    dfout = Dense(1, activation='sigmoid')(df1)



    #三个独立模型，以用来在训练的时候做区别
    modelencodor=Model([input,inpute],f)
    modeldecoder=Model([inputdecodor,inputdee],pic1layer)
    modeldiscriminator=Model([inputdisimag,inputdiscode],dfout)


    '''做三个模型之间的连接，连接的时候，首先需要输出单模型的输出，然后用这个输出输入到下一个模型，
        而不能直接用前模型的输出层作为输入项，因为后模型的输入是以Ｉｎｐｕｔ形式定义的'''
    fcode=modelencodor([input,inpute])
    pic_from_de=modeldecoder([inputdecodor,inputdee])
    fake=modeldiscriminator([input,fcode])
    true=modeldiscriminator([pic_from_de,inputdecodor])

    combine_e_dis=Model([input,inpute],fake)
    combine_de_dis=Model([inputdecodor,inputdee],true)

    return combine_e_dis,combine_de_dis,modelencodor,modeldecoder,modeldiscriminator



























