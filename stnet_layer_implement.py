import tensorflow as tf
import keras.backend as K
from keras.layers import Conv1D, Conv2D, Conv3D, SeparableConv1D, Dense
from keras.layers import MaxPool2D, GlobalAveragePooling1D, GlobalAveragePooling2D
from keras.layers import Lambda, Reshape, Multiply, Permute, Add, Concatenate
from keras.layers import Activation, LeakyReLU, Input, BatchNormalization, Layer
from keras.models import Model, Sequential

initializer = 'he_normal'
reduction = 16
expansion = 4

class SeLayer(Layer):
    def __init__(self, reduction, in_channel, **kwargs):
        super(SeLayer, self).__init__(**kwargs)
        self.reduction = reduction
        self.avgpool = GlobalAveragePooling2D('channels_last')
        self.fc1 = Dense(int(in_channel // reduction))
        self.fc2 = Dense(in_channel, activation='sigmoid')
        self.in_channel = in_channel
        
    def call(self, x):
        C = x._keras_shape[-1]
        residual = x
        x = self.avgpool(x)
        x = Reshape((-1, self.in_channel))(x)
        x = self.fc1(x)
        x = LeakyReLU()(x)
        x = self.fc2(x)
        x = Reshape((1, 1, self.in_channel))(x)
        x = Multiply()([x, residual])
        return x
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = dict(reduction=self.reduction, in_channel=self.in_channel)
        base_cfg = super(SeLayer, self).get_config()
        return dict(list(config.items())+list(base_cfg.items()))
    
class TemporalModule(Layer):
    def __init__(self, in_channel, T, **kwargs):
        super(TemporalModule, self).__init__(**kwargs)
        self.conv = Conv3D(in_channel, kernel_size=[3,1,1], strides=[1,1,1], padding="same", kernel_initializer=initializer, use_bias=False)
        self.bn = BatchNormalization()
        self.relu = Activation('relu')
        self.T = T
        self.in_channel = in_channel
        
    def call(self, x):
        B,H,W,C = x._keras_shape
        x = Lambda(lambda x: tf.reshape(x, [-1, self.T, H, W, C]))(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = Lambda(lambda x: tf.reshape(x, [-1, H, W, C]))(x)
        return x
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = dict(in_channel=self.in_channel, T=self.T)
        base_cfg = super(TemporalModule, self).get_config()
        return dict(list(config.items())+list(base_cfg.items()))
    
class SeBottleneck(Layer):
    def __init__(self, in_planes, planes, strides, **kwargs):
        super(SeBottleneck, self).__init__(**kwargs)
        self.planes = planes
        self.strides = strides
        if strides != 1 or in_planes != planes*expansion:
            self.downsample = Sequential(
                [Conv2D(planes*expansion, kernel_size=3 if strides==1 else 2, strides=strides, padding='same', kernel_initializer=initializer, use_bias=False),
                BatchNormalization()]
            )
        else:
            self.downsample = None
        self.conv1 = Conv2D(planes, kernel_size=1, strides=1, padding='valid', kernel_initializer=initializer, use_bias=False) 
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(planes, kernel_size=3, strides=strides, padding='same', kernel_initializer=initializer, use_bias=False)
        self.bn2 = BatchNormalization()
        self.conv3 = Conv2D(planes*expansion, kernel_size=1, strides=1, padding='valid', kernel_initializer=initializer, use_bias=False)
        self.bn3 = BatchNormalization()
        self.relu = Activation('relu')
        self.se = SeLayer(reduction, planes*expansion)
        
    def call(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            residual = self.downsample(residual)
        x = Add()([self.se(x), residual])
        x = self.relu(x)
        return x
    
    def compute_output_shape(self, input_shape):
        if self.strides == 2:
            return (input_shape[0], input_shape[1]//2, input_shape[2]//2, self.planes*expansion)
        else:
            return (input_shape[0], input_shape[1], input_shape[2], self.planes*expansion)
        
    def get_config(self):
        config = dict(in_planes=self.in_planes, planes=self.planes, strides=self.strides)
        base_cfg = super(SeBottleneck, self).get_config()
        return dict(list(config.items())+list(base_cfg.items()))
    
class ResLayer(Layer):
    def __init__(self, block, in_planes, planes, blocks, strides, **kwargs):
        super(ResLayer, self).__init__(**kwargs)
        self.strides = strides
        self.planes = planes
        self.layers = []
        self.layers.append(block(in_planes, planes, strides=strides))
        for i in range(1, blocks):
            self.layers.append(block(planes*4, planes, strides=1))
            
    def call(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def compute_output_shape(self, input_shape):
        if self.strides == 2:
            return (input_shape[0], input_shape[1]//2, input_shape[2]//2, self.planes*expansion)
        else:
            return (input_shape[0], input_shape[1], input_shape[2], self.planes*expansion)
        
    def get_config(self):
        config = dict(block=self.block, in_planes=self.in_planes, planes=self.planes, blocks=self.blocks, strides=self.strides)
        base_cfg = super(ResLayer, self).get_config()
        return dict(list(config.items())+list(base_cfg.items())) 
    
class TemporalXception(Layer):
    def __init__(self, planes, **kwargs):
        super(TemporalXception, self).__init__(**kwargs)
        self.bn0 = BatchNormalization()
        self.conv = Conv1D(planes, kernel_size=1, strides=1, padding='valid', kernel_initializer=initializer, use_bias=False)
        
        self.conv3x3_1 = SeparableConv1D(planes, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, use_bias=False)
        self.conv1x1_1 = Conv1D(planes, kernel_size=1, strides=1, padding='same', use_bias=False)
        self.bn1 = BatchNormalization()
        
        self.conv3x3_2 = SeparableConv1D(planes, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, use_bias=False)
        self.conv1x1_2 = Conv1D(planes, kernel_size=1, strides=1, padding='same', use_bias=False)
        self.bn2 = BatchNormalization()
        
        self.relu = Activation('relu')
        self.avgpool = GlobalAveragePooling1D('channels_last')
        self.planes = planes
        
    def call(self, x):
        # C = x._keras_shape[-1]
        x = self.bn0(x)
        x1 = self.conv(x)
        x = self.conv3x3_1(x)
        x = self.conv1x1_1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv3x3_2(x)
        x = self.conv1x1_2(x)

        x = Add()([x, x1])
        x = self.bn2(x)
        x = self.relu(x)
        x = self.avgpool(x)
        return x
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.planes)
    
    def get_config(self):
        config = dict(planes=self.planes)
        base_cfg = super(TemporalXception, self).get_config()
        return dict(list(config.items())+list(base_cfg.items())) 
    
class StNet(Layer):
    def __init__(self, num_classes, base_ch=64, layers=[3,4,6,3], T=5, N=4, return_feature=False, **kwargs):
        super(StNet, self).__init__(**kwargs)
        self.conv1 = Conv2D(base_ch, kernel_size=7, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)
        self.bn1 = BatchNormalization()
        self.relu = Activation('relu')
        self.pool = MaxPool2D((3,3),(2,2),padding='same')
        self.res2 = ResLayer(SeBottleneck, base_ch,   base_ch*2//expansion, layers[0], 1)
        self.res3 = ResLayer(SeBottleneck, base_ch*2, base_ch*4//expansion, layers[1], 2)
        self.temp1 = TemporalModule(base_ch*4, T)
        self.res4 = ResLayer(SeBottleneck, base_ch*4, base_ch*8//expansion, layers[2], 2)
        self.temp2 = TemporalModule(base_ch*8, T)
        self.res5 = ResLayer(SeBottleneck, base_ch*8, base_ch*16//expansion, layers[3], 2)
        self.avgpool = GlobalAveragePooling2D('channels_last')
        self.T = T
        self.N = N
        self.txep = TemporalXception(base_ch*16)
        self.fc = Dense(num_classes, activation='softmax')
        self.num_classes = num_classes
        self.layers = layers
        self.base_ch = base_ch
        self.return_feature = return_feature
    def call(self, x):
        # reshape
        _,D,H,W,C = x._keras_shape
        assert self.T*self.N == D
        x = Lambda(lambda x:tf.reshape(x, [-1, self.N, H, W, C]))(x)
        x = Permute([2,3,4,1])(x)
        x = Lambda(lambda x:tf.reshape(x, [-1, H, W, C*self.N]))(x)
        # conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        # res2
        x = self.res2(x)
        # res3
        x = self.res3(x)
        # temp1
        x = self.temp1(x)
        # res4
        x = self.res4(x)
        # temp2
        x = self.temp2(x)
        # res5
        x = self.res5(x)
        # AvgPool
        x = self.avgpool(x)
        # reshape
        C = x._keras_shape[-1]
        x = Lambda(lambda x:tf.reshape(x, [-1, self.T, C]))(x)
        # TempXception
        x = self.txep(x)
        if self.return_feature:
            return x
        # fc
        x = self.fc(x)
        return x
    def compute_output_shape(self, input_shape):
        if self.return_feature:
            return (input_shape, self.base_ch*16)
        else:
            return (input_shape, self.num_classes)
        
    def get_config(self):
        config = dict(num_classes=self.num_classes, base_ch=self.base_ch, layers=self.layers, T=self.T, N=self.N, return_feature=self.return_feature)
        base_cfg = super(StNet, self).get_config()
        return dict(list(config.items())+list(base_cfg.items())) 
    
def stnet(input_size, num_classes):
    inputs = Input(shape=input_size)
    outputs = StNet(num_classes)(inputs)
    model = Model(inputs=inputs, outputs=outputs)
    return model        
    
if __name__ == "__main__":

    with tf.device("/cpu:0"):
        fake = keras.layers.Input(shape=(20, 128, 128, 1))
        model = stnet([20, 128, 128, 1], 5)
        outputs = model(fake)
        print(outputs.shape)
