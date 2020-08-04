import tensorflow as tf
from keras.layers import Conv1D, Conv2D, Conv3D, SeparableConv1D, Dense
from keras.layers import MaxPool2D, GlobalAveragePooling1D, GlobalAveragePooling2D
from keras.layers import Lambda, Reshape, Multiply, Permute, Add
from keras.layers import Activation, LeakyReLU, Input, BatchNormalization
from keras.models import Model, Sequential

initializer = 'he_normal'
reduction = 16

def se_layer(x, reduction):
    C = x._keras_shape[-1]
    residual = x
    x = GlobalAveragePooling2D('channels_last')(x)
    x = Reshape((-1,C))(x)
    x = Dense(int(C // reduction))(x)
    x = LeakyReLU()(x)
    x = Dense(C, activation='sigmoid')(x)
    x = Reshape((1,1,C))(x)
    x = Multiply()([x, residual])
    return x

def temporal_module(x, in_channel, T):
    B,H,W,C = x._keras_shape
    x = Lambda(lambda x: tf.reshape(x, [-1, T, H, W, C]))(x)
    x = Conv3D(in_channel, kernel_size=[3,1,1], strides=[1,1,1], padding="same", kernel_initializer=initializer, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Lambda(lambda x: tf.reshape(x, [-1, H, W, C]))(x)
    return x

def sebottleneck(x, planes, strides):
    expansion = 4
    C = x._keras_shape[-1]
    if strides != 1 or C != planes*expansion:
        downsample = Sequential(
            [Conv2D(planes*expansion, kernel_size=3 if strides==1 else 2, strides=strides, padding='same', kernel_initializer=initializer, use_bias=False),
            BatchNormalization()]
        )
    else:
        downsample = None
    residual = x

    x = Conv2D(planes, kernel_size=1, strides=1, padding='valid', kernel_initializer=initializer, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(planes, kernel_size=3, strides=strides, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(planes*expansion, kernel_size=1, strides=1, padding='valid', kernel_initializer=initializer, use_bias=False)(x)
    x = BatchNormalization()(x)

    if downsample is not None:
        residual = downsample(residual)
    x = Add()([se_layer(x, reduction), residual])
    x = Activation('relu')(x)
    return x

def mask_layer(x, block, planes, blocks, strides):
    x = block(x, planes, strides=strides)
    for i in range(1, blocks):
        x = block(x, planes, strides=1)
    return x

def TemporalXception(x):
    C = x._keras_shape[-1]
    x = BatchNormalization()(x)
    x1 = Conv1D(C, kernel_size=1, strides=1, padding='valid', kernel_initializer=initializer, use_bias=False)(x)
    x = SeparableConv1D(C, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    x = Conv1D(C, kernel_size=1, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv1D(C, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    x = Conv1D(C, kernel_size=1, strides=1, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    
    x = Add()([x, x1])
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling1D('channels_last')(x)
    return x

def stnet_keras(input_size, num_classes, layers=[3,4,6,3]):
    inputs = Input(shape=input_size)
    base_ch = 64
    B,D,H,W,C = inputs._keras_shape
    T = 5
    N = 4
    assert T * N == D
    # reshape
    x = Lambda(lambda x:tf.reshape(x, [-1, N, H, W, C]))(inputs)
    x = Permute([2,3,4,1])(x)
    x = Lambda(lambda x:tf.reshape(x, [-1, H, W, C*N]))(x)
    # conv1
    x = Conv2D(64, kernel_size=7, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D((3,3),(2,2),padding='same')(x)
    # Res2
    x = mask_layer(x, sebottleneck, base_ch//2, layers[0], 1)
    # Res3
    x = mask_layer(x, sebottleneck, base_ch//1, layers[1], 2)
    # Temp1
    x = temporal_module(x, base_ch*4, T)
    # Res4
    x = mask_layer(x, sebottleneck, base_ch*2,  layers[2], 2)
    # Temp2
    x = temporal_module(x, base_ch*8, T)
    # Res5
    x = mask_layer(x, sebottleneck, base_ch*4,  layers[3], 2)
    # AvgPool
    x = GlobalAveragePooling2D('channels_last')(x)
    # reshape
    C = x._keras_shape[-1]
    x = Lambda(lambda x:tf.reshape(x, [-1, T, C]))(x)
    # TempXception
    x = TemporalXception(x)
    # fc
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=x)
    return model

if __name__ == "__main__":  
    with tf.device("/cpu:0"):
        model = stnet_keras([20, 128, 128, 1], 5)
        print(model.outputs[0].shape)