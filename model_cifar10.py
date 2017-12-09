import keras
import numpy as np
from keras.layers import Input,Conv2D,MaxPooling2D,Flatten,Dense,Reshape,Deconv2D,BatchNormalization,advanced_activations,MaxoutDense,Dropout
from keras.layers.merge import Concatenate,Add
from keras.layers.merge import Multiply
from keras.models import Model
from keras.initializers import RandomNormal,TruncatedNormal


import random

def build_Model():
    #编码器模型
    encoder_input=Input((32,32,3),name='image_input',dtype=np.float32)
    gauss_encoder = Input((1,1,64,),name='gauss_input')
    model1convlayer1=Conv2D(32,(5,5),strides=(1,1),padding="valid",kernel_initializer=TruncatedNormal(mean=0,stddev=1))(encoder_input)
    model1convlayer1=BatchNormalization(axis=-1)(model1convlayer1)
    model1convlayer1=advanced_activations.LeakyReLU(0.1)(model1convlayer1)
    model1convlayer2=Conv2D(64,(4,4), strides=(2,2),padding="valid",kernel_initializer=TruncatedNormal(mean=0,stddev=1))(model1convlayer1)
    model1convlayer2=BatchNormalization(axis=-1)(model1convlayer2)
    model1convlayer2 = advanced_activations.LeakyReLU(0.1)(model1convlayer2)
    model1convlayer3 = Conv2D(128, (4,4), strides=(1,1), padding="valid",kernel_initializer=TruncatedNormal(mean=0,stddev=1))(model1convlayer2)
    model1convlayer3=BatchNormalization(axis=-1)(model1convlayer3)
    model1convlayer3 = advanced_activations.LeakyReLU(0.1)(model1convlayer3)
    model1convlayer4 = Conv2D(256, (4,4), strides=(2,2), padding="valid",kernel_initializer=TruncatedNormal(mean=0,stddev=1))(model1convlayer3)
    model1convlayer4 = BatchNormalization(axis=-1)(model1convlayer4)
    model1convlayer4 = advanced_activations.LeakyReLU(0.1)(model1convlayer4)
    model1convlayer5 = Conv2D(512, (4, 4), strides=(1, 1), padding="valid",kernel_initializer=TruncatedNormal(mean=0,stddev=1) )(model1convlayer4)
    model1convlayer5 = BatchNormalization(axis=-1)(model1convlayer5)
    model1convlayer5 = advanced_activations.LeakyReLU(0.1)(model1convlayer5)
    model1convlayer6 = Conv2D(512, (1, 1), strides=(1, 1), padding="valid",kernel_initializer=TruncatedNormal(mean=0,stddev=1) )(model1convlayer5)
    model1convlayer6 = BatchNormalization(axis=-1)(model1convlayer6)
    model1convlayer6 = advanced_activations.LeakyReLU(0.1)(model1convlayer6)
    z_u = Dense(64,)(model1convlayer6)
    z_delta = Dense(64,)(model1convlayer6)
    DdelM = Multiply()([z_delta, gauss_encoder])
    encoder_out = Add()([z_u, DdelM])

    #解码器模型
    decoder_input=Input((1,1,64),name='z_input')
    deconvlayer1=Deconv2D(256,(4,4),padding='valid',kernel_initializer=TruncatedNormal(mean=0,stddev=1))(decoder_input)
    deconvlayer1 = BatchNormalization(axis=-1)(deconvlayer1)
    deconvlayer1 = advanced_activations.LeakyReLU(0.1)(deconvlayer1)
    deconvlayer2 = Deconv2D(128, (4, 4),strides=(2,2), padding='valid',kernel_initializer=TruncatedNormal(mean=0,stddev=1))(deconvlayer1)
    deconvlayer2 = BatchNormalization(axis=-1)(deconvlayer2)
    deconvlayer2 = advanced_activations.LeakyReLU(0.1)(deconvlayer2)
    deconvlayer3 = Deconv2D(64, (4, 4), padding='valid',kernel_initializer=TruncatedNormal(mean=0,stddev=1))(deconvlayer2)
    deconvlayer3 = BatchNormalization(axis=-1)(deconvlayer3)
    deconvlayer3 = advanced_activations.LeakyReLU(0.1)(deconvlayer3)
    deconvlayer4 = Deconv2D(32, (4, 4), strides=(2, 2), padding='valid',kernel_initializer=TruncatedNormal(mean=0,stddev=1))(deconvlayer3)
    deconvlayer4 = BatchNormalization(axis=-1)(deconvlayer4)
    deconvlayer4 = advanced_activations.LeakyReLU(0.1)(deconvlayer4)
    deconvlayer5 = Deconv2D(32, (5, 5), padding='valid',kernel_initializer=TruncatedNormal(mean=0,stddev=1))(deconvlayer4)
    deconvlayer5 = BatchNormalization(axis=-1)(deconvlayer5)
    deconvlayer5 = advanced_activations.LeakyReLU(0.1)(deconvlayer5)
    deconvlayer6 = Conv2D(32, (1, 1), strides=(1, 1), padding="valid",kernel_initializer=TruncatedNormal(mean=0,stddev=1))(deconvlayer5)
    deconvlayer6 = BatchNormalization(axis=-1)(deconvlayer6)
    deconvlayer6 = advanced_activations.LeakyReLU(0.1)(deconvlayer6)
    decoder_out = Conv2D(3, (1, 1), strides=(1, 1), padding="valid", activation='sigmoid',
                         kernel_initializer=RandomNormal(mean=0,stddev=0.1))(deconvlayer6)

    #判别器模型
    image_dis = Input((32, 32, 3), name='image_discriminator', dtype=np.float32)
    z_dis = Input((1, 1, 64), name='gauss_discriminator')
    disconvlayer1 = Conv2D(32, (5, 5), strides=(1, 1), padding="valid",kernel_initializer=TruncatedNormal(mean=0,stddev=1))(image_dis)
    disconvlayer1 = advanced_activations.LeakyReLU(0.1)(disconvlayer1)
    disconvlayer1 = Dropout(0.2)(disconvlayer1)
    disconvlayer2 = Conv2D(64, (4, 4), strides=(2, 2), padding="valid",kernel_initializer=TruncatedNormal(mean=0,stddev=1))(disconvlayer1)
    disconvlayer2 = advanced_activations.LeakyReLU(0.1)(disconvlayer2)
    disconvlayer2 = Dropout(0.5)(disconvlayer2)
    disconvlayer3 = Conv2D(128, (4, 4), strides=(1, 1), padding="valid",kernel_initializer=TruncatedNormal(mean=0,stddev=1) )(disconvlayer2)
    disconvlayer3 = advanced_activations.LeakyReLU(0.1)(disconvlayer3)
    disconvlayer3 = Dropout(0.5)(disconvlayer3)
    disconvlayer4 = Conv2D(256, (4, 4), strides=(2, 2), padding="valid",kernel_initializer=TruncatedNormal(mean=0,stddev=1) )(disconvlayer3)
    disconvlayer4 = advanced_activations.LeakyReLU(0.1)(disconvlayer4)
    disconvlayer4 = Dropout(0.5)(disconvlayer4)
    disconvlayer5 = Conv2D(512, (4, 4), strides=(1, 1), padding="valid",kernel_initializer=TruncatedNormal(mean=0,stddev=1) )(disconvlayer4)
    disconvlayer5 = advanced_activations.LeakyReLU(0.1)(disconvlayer5)
    disconvlayer5 = Dropout(0.5)(disconvlayer5)

    z_convlayer1 = Conv2D(512, (1, 1), strides=(1, 1), padding="valid",kernel_initializer=TruncatedNormal(mean=0,stddev=1) )(z_dis)
    z_convlayer1 = advanced_activations.LeakyReLU(0.1)(z_convlayer1)
    z_convlayer1 = Dropout(0.2)(z_convlayer1)
    z_convlayer2 = Conv2D(512, (1, 1), strides=(1, 1), padding="valid",kernel_initializer=TruncatedNormal(mean=0,stddev=1) )(z_convlayer1)
    z_convlayer2 = advanced_activations.LeakyReLU(0.1)(z_convlayer2)
    z_convlayer2 = Dropout(0.5)(z_convlayer2)

    con_imagc_code = Concatenate(axis=1)([disconvlayer5, z_convlayer2])
    xz_convlayer1 = Conv2D(1024, (1, 1), strides=(1, 1), padding="valid",kernel_initializer=TruncatedNormal(mean=0,stddev=1))(con_imagc_code)
    xz_convlayer1 = advanced_activations.LeakyReLU(0.1)(xz_convlayer1)
    xz_convlayer1 = Dropout(0.5)(xz_convlayer1)
    xz_convlayer2 = Conv2D(1024, (1, 1), strides=(1, 1), padding="valid",kernel_initializer=TruncatedNormal(mean=0,stddev=1) )(xz_convlayer1)
    xz_convlayer2 = advanced_activations.LeakyReLU(0.1)(xz_convlayer2)
    xz_convlayer2 = Dropout(0.5)(xz_convlayer2)
    xz_convlayer2 = Flatten()(xz_convlayer2)
    xz_convlayer3 = Dense(1, activation='sigmoid')(xz_convlayer2)
    dis_out = Dropout(0.5)(xz_convlayer3)



    #三个独立模型，以用来在训练的时候做区别
    modelencodor=Model([encoder_input,gauss_encoder],encoder_out)
    modeldecoder=Model(decoder_input,decoder_out)
    modeldiscriminator=Model([image_dis,z_dis],dis_out)


    '''做三个模型之间的连接，连接的时候，首先需要输出单模型的输出，然后用这个输出输入到下一个模型，
        而不能直接用前模型的输出层作为输入项，因为后模型的输入是以Ｉｎｐｕｔ形式定义的'''
    fcode=modelencodor([encoder_input,gauss_encoder])
    pic_from_de=modeldecoder(decoder_input)
    fake=modeldiscriminator([encoder_input,fcode])
    true=modeldiscriminator([pic_from_de,decoder_input])

    combine_e_dis=Model([encoder_input,gauss_encoder],fake)
    combine_de_dis=Model(decoder_input,true)

    return combine_e_dis,combine_de_dis,modelencodor,modeldecoder,modeldiscriminator



























