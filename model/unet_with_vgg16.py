from keras.applications import VGG16
from keras.models import Model, Input
from keras import layers
from keras.optimizers import RMSprop
from model.losses import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss, dice_coeff
from keras import backend as K
#
def get_unet_with_vgg16(input_shape=(128, 128, 3),
                 num_classes=1):
    conv_base = VGG16(include_top=False, input_shape=input_shape)
    #conv_base.summary()
    layer_outputs = [layer.output for layer in conv_base.layers[:19]]

    x = layers.Conv2D(512, 3, activation='relu', padding='same', )(conv_base.output)
    x = layers.Conv2DTranspose(256, 3, strides=(2,2),padding='same', activation='relu')(x)
    x = layers.concatenate([x, layer_outputs[17]], axis=-1 )
    x = layers.Conv2D(512, 3, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(256, 3, strides=(2,2), padding='same', activation='relu')(x)
    #
    ## crop and concatenate
    #y = layers.Cropping2D(cropping=((1,0),(1,0)))(layer_outputs[13])
    x = layers.concatenate([x, layer_outputs[13]],axis=-1)
    ##--------------------
    x = layers.Conv2D(512, 3, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(128, 3, strides=(2,2), padding='same', activation='relu')(x)
    #
    ## crop and concatenate
    #y = layers.Cropping2D(cropping=((3,0),(3,0)))(layer_outputs[9])
    x = layers.concatenate([x, layer_outputs[9]], axis=-1)
    ##--------------------
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(64, 3, strides=(2,2), padding='same', activation='relu')(x)
    #
    ## crop and concatenate
    #y = layers.Cropping2D(cropping=((3,3),(3,3)))(layer_outputs[5])
    x = layers.concatenate([x, layer_outputs[5]], axis=-1)
    ##--------------------
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(32, 3, strides=(2,2), padding='same', activation='relu')(x)
    #
    ## crop and concatenate
    #y = layers.Cropping2D(cropping=((6,6),(6,6)))(layer_outputs[2])
    x = layers.concatenate([x, layer_outputs[2]], axis=-1)
    ##--------------------
    x = layers.Conv2D(1, 1, activation='sigmoid', padding='same')(x)
    model = Model(inputs=conv_base.inputs,outputs=x)
    model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=['acc',dice_coeff])
#    model.summary()
    return model
