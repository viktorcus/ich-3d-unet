import keras
from keras import Model
from keras.layers import Input, Conv3D, BatchNormalization, MaxPooling3D, LeakyReLU, Conv3DTranspose, Concatenate

def build_UNet(input_size = (251, 251, 32, 1)):
    inputs = Input(input_size)
    conv1 = Conv3D(16, kernel_size=(3,3,3), padding='same')(inputs)
    conv1 = BatchNormalization(axis=4)(conv1)
    conv1 = LeakyReLU()(conv1)
    conv1 = Conv3D(16, kernel_size=(3,3,3), padding='same')(conv1)
    conv1 = BatchNormalization(axis=4)(conv1)
    conv1 = LeakyReLU()(conv1)
    pool1 = MaxPooling3D((2,2,2),padding='valid')(conv1)

    conv2 = Conv3D(32, kernel_size=(3,3,3), padding='same')(pool1)
    conv2 = BatchNormalization(axis=4)(conv2)
    conv2 = LeakyReLU()(conv2)
    conv2 = Conv3D(32, kernel_size=(3,3,3), padding='same')(conv2)
    conv2 = BatchNormalization(axis=4)(conv2)
    conv2 = LeakyReLU()(conv2)
    pool2 = MaxPooling3D((2,2,2), padding='valid')(conv2)

    conv3 = Conv3D(64, kernel_size=(3,3,3), padding='same')(pool2)
    conv3 = BatchNormalization(axis=4)(conv3)
    conv3 = LeakyReLU()(conv3)
    conv3 = Conv3D(64, kernel_size=(3,3,3), padding='same')(conv3)
    conv3 = BatchNormalization(axis=4)(conv3)
    conv3 = LeakyReLU()(conv3)
    pool3 = MaxPooling3D((2,2,2), padding='valid')(conv3)

    conv4 = Conv3D(128, kernel_size=(3,3,3), padding='same')(pool3)
    conv4 = BatchNormalization(axis=4)(conv4)
    conv4 = LeakyReLU()(conv4)
    conv4 = Conv3D(128, kernel_size=(3,3,3), padding='same')(conv4)
    conv4 = BatchNormalization(axis=4)(conv4)
    conv4 = LeakyReLU()(conv4)
    pool4 = MaxPooling3D((2,2,2), padding='valid')(conv4)
    
    dconv3 = Conv3DTranspose(64, kernel_size=(2,2,2), strides=(2,2,2), padding='valid')(conv4)#, output_padding=(0,0,1))(conv4)
    dconv3 = BatchNormalization(axis=4)(dconv3)
    dconv3 = LeakyReLU()(dconv3)
    dconv3 = Concatenate(axis=4)([dconv3, conv3])
    dconv3 = Conv3D(64, kernel_size=(3,3,3), padding='same')(dconv3)
    dconv3 = BatchNormalization(axis=4)(dconv3)
    dconv3 = LeakyReLU()(dconv3)


    dconv2 = Conv3DTranspose(32, kernel_size=(2,2,2), strides=(2,2,2), padding='valid', output_padding=(1,1,0))(dconv3)
    dconv2 = BatchNormalization(axis=4)(dconv2)
    dconv2 = LeakyReLU()(dconv2)
    dconv2 = Concatenate(axis=4)([conv2, dconv2])
    dconv2 = Conv3D(32, kernel_size=(3,3,3), padding='same')(dconv2)
    dconv2 = BatchNormalization(axis=4)(dconv2)
    dconv2 = LeakyReLU()(dconv2)

    dconv1 = Conv3DTranspose(16, kernel_size=(2,2,2), strides=(2,2,2), padding='valid', output_padding=(1,1,0))(dconv2)
    dconv1 = BatchNormalization(axis=4)(dconv1)
    dconv1 = LeakyReLU()(dconv1)
    dconv1 = Concatenate(axis=4)([dconv1, conv1])
    dconv1 = Conv3D(16, kernel_size=(3,3,3), padding='same')(dconv1)
    dconv1 = BatchNormalization(axis=4)(dconv1)
    dconv1 = LeakyReLU()(dconv1)
    
    outputs = Conv3D(1, kernel_size=(1,1,1), padding='same', activation='sigmoid')(dconv1)
    
    return Model(inputs, outputs)