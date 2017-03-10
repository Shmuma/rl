# Atari-specific options for environments
from keras.layers import Input, Flatten, Conv2D, MaxPooling2D, Permute, Reshape, Lambda, Dense
import tensorflow as tf

# how many steps to keep in input shape
HISTORY_STEPS = 4

# shape of prescaled image. If None, no prescaling will be performed
IMAGE_RESCALE = (84, 84)


def net_input(env_state_shape):
    """
    Create input part of the network with optional prescaling.
    :return: input_tensor, output_tensor
    """
    assert len(env_state_shape) == 3

    # Our input looks has this shape:
    # HISTORY_STEPS, 210, 160, 3

    # To be able to apply convolution to this, we need to convert it into this
    # 210, 160, 3*HISTORY_STEPS
    input_shape = (HISTORY_STEPS, ) + env_state_shape
    tgt_channels = env_state_shape[2] * HISTORY_STEPS
    tgt_shape = (env_state_shape[0], env_state_shape[1], tgt_channels)

    in_t = Input(shape=input_shape, name='input')
    out_t = Permute(dims=(2, 3, 4, 1), name="move_hist")(in_t)
    out_t = Reshape(target_shape=tgt_shape, name="squash_hist")(out_t)

    # optional rescale
    if IMAGE_RESCALE is not None:
        out_t = Lambda(lambda img: tf.image.resize_bicubic(img, IMAGE_RESCALE), name='rescale')(out_t)

    out_t = Lambda(lambda img: img / 255.0, name='normalize')(out_t)

    out_t = Conv2D(32, 5, 5, activation='relu', border_mode='same')(out_t)
    out_t = MaxPooling2D((2, 2))(out_t)
    out_t = Conv2D(32, 5, 5, activation='relu', border_mode='same')(out_t)
    out_t = MaxPooling2D((2, 2))(out_t)
    out_t = Conv2D(64, 4, 4, activation='relu', border_mode='same')(out_t)
    out_t = MaxPooling2D((2, 2))(out_t)
    out_t = Conv2D(64, 3, 3, activation='relu', border_mode='same')(out_t)
    out_t = Flatten(name='flat')(out_t)
    out_t = Dense(512, name='l1', activation='relu')(out_t)

    return in_t, out_t
