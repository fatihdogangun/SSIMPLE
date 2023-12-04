#@title Unet2D
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Multiply, Flatten, LeakyReLU, Dropout, Input, Conv2D, MaxPooling2D, UpSampling2D, Activation, BatchNormalization, Add, Conv2DTranspose, LeakyReLU, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError


def conv2D_bn_nonlinear(x, num_out_chan, kernel_size, activation_type='relu', USE_BN=True, layer_name=''):
    with tf.name_scope(layer_name):
        x = Conv2D(num_out_chan, (kernel_size, kernel_size), activation=None, padding='same', kernel_initializer='truncated_normal')(x)
        if USE_BN:
            x = BatchNormalization()(x)
        if activation_type == 'LeakyReLU':
            return LeakyReLU()(x)
        else:
            return Activation(activation_type)(x)

def conv2Dt_bn_nonlinear(x, num_out_chan, kernel_size, activation_type='relu', USE_BN=True, layer_name=''):
    with tf.name_scope(layer_name):
        x = Conv2DTranspose(num_out_chan, (kernel_size, kernel_size), strides=(2, 2), padding='same', kernel_initializer='truncated_normal')(x)
        if USE_BN:
            x = BatchNormalization()(x)
        if activation_type == 'LeakyReLU':
            return LeakyReLU()(x)
        else:
            return Activation(activation_type)(x)

def createOneLevel_UNet2D(x, num_out_chan, kernel_size, depth, num_chan_increase_rate, activation_type, dropout_rate, USE_BN):
    if depth > 0:
        x = conv2D_bn_nonlinear(x, num_out_chan, kernel_size, activation_type=activation_type, USE_BN=USE_BN)
        x = Dropout(dropout_rate)(x)

        x = conv2D_bn_nonlinear(x, num_out_chan, kernel_size, activation_type=activation_type, USE_BN=USE_BN)
        x = Dropout(dropout_rate)(x)

        x_to_lower_level = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format='channels_last')(x)

        x_from_lower_level = createOneLevel_UNet2D(x_to_lower_level, int(num_chan_increase_rate*num_out_chan), kernel_size, depth-1, num_chan_increase_rate, activation_type, dropout_rate, USE_BN)
        x_conv2Dt = conv2Dt_bn_nonlinear(x_from_lower_level, num_out_chan, kernel_size, activation_type=activation_type, USE_BN=USE_BN)

        x = concatenate([x, x_conv2Dt], axis=3)

        x = conv2D_bn_nonlinear(x, num_out_chan, kernel_size, activation_type=activation_type, USE_BN=USE_BN)
        x = Dropout(dropout_rate)(x)

        x = conv2D_bn_nonlinear(x, num_out_chan, kernel_size, activation_type=activation_type, USE_BN=USE_BN)
        x = Dropout(dropout_rate)(x)

        return x
    else:
        x = conv2D_bn_nonlinear(x, num_out_chan, kernel_size, activation_type=activation_type, USE_BN=USE_BN)

        x = Dropout(dropout_rate)(x)

        x = conv2D_bn_nonlinear(x, num_out_chan, kernel_size, activation_type=activation_type, USE_BN=USE_BN)

        x = Dropout(dropout_rate)(x)

    return x


def UNet2D(row, col, kernel_size, num_out_chan_highest_level, depth, num_chan_increase_rate, activation_type, dropout_rate, USE_BN, SKIP_CONNECTION_AT_THE_END, num_input_chans, num_output_chans):

    input_img = Input(shape=(row, col, num_input_chans))

    x = conv2D_bn_nonlinear(input_img, num_out_chan_highest_level, kernel_size, activation_type=activation_type, USE_BN=USE_BN)
    temp = createOneLevel_UNet2D(x, num_out_chan_highest_level, kernel_size, depth-1, num_chan_increase_rate, activation_type, dropout_rate, USE_BN)
    output_img = conv2D_bn_nonlinear(temp, num_output_chans, kernel_size, activation_type="sigmoid", USE_BN=False)

    if SKIP_CONNECTION_AT_THE_END:
        if num_input_chans == num_output_chans:
            output_img = Add()([input_img, output_img])
            output_img = Activation('sigmoid')(output_img)
        else:
            # If the # of input channels != # of output channels, the input and output images are concatenated along the channel axis,
            # and a 1x1 convolution is applied to adjust the number of output channels.
            merged_img = concatenate([input_img, output_img], axis=-1)
            output_img = Conv2D(num_output_chans, kernel_size=(1, 1), activation= None, padding='same', kernel_initializer='truncated_normal')(merged_img)
            output_img = Activation('sigmoid')(output_img)

    return Model(inputs=input_img, outputs=output_img)

