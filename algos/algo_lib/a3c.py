from keras.layers import Dense, Input, Lambda
from keras.models import Model
import keras.backend as K
import tensorflow as tf


def net_prediction(input_t, n_actions):
    """
    Make prediction part of A3C network
    :param input_t: flattened input from previous layer
    :return: policy_tensor and value_tensor
    """
    out_t = Dense(512, name='l1', activation='relu')(input_t)

    policy_t = Dense(n_actions, activation='softmax', name='policy')(out_t)
    value_t = Dense(1, name='value')(out_t)

    return policy_t, value_t


def net_loss(policy_t, value_t, n_actions):
    """
    Make traning part of the A3C network
    :param policy_t: policy tensor from prediction part
    :param value_t: value tensor from prediction part
    :param n_actions: count of actions in space
    :return: action_t, advantage_t, policy_loss_t
    """
    action_t = Input(batch_shape=(None, 1), name='action', dtype='int32')
    advantage_t = Input(batch_shape=(None, 1), name="advantage")

    tf.summary.scalar("value", K.mean(value_t))
    tf.summary.scalar("advantage_mean", K.mean(advantage_t))
    tf.summary.scalar("advantage_rms", K.sqrt(K.mean(K.square(advantage_t))))

    X_ENTROPY_BETA = 0.01

    def policy_loss_func(args):
        p_t, act_t, adv_t = args
        oh_t = K.one_hot(act_t, n_actions)
        oh_t = K.squeeze(oh_t, 1)
        p_oh_t = K.log(K.epsilon() + K.sum(oh_t * p_t, axis=-1, keepdims=True))
        res_t = adv_t * p_oh_t
        x_entropy_t = K.sum(p_t * K.log(K.epsilon() + p_t), axis=-1, keepdims=True)
        full_policy_loss_t = -res_t + X_ENTROPY_BETA * x_entropy_t
        tf.summary.scalar("loss_entropy", K.sum(x_entropy_t))
        tf.summary.scalar("loss_policy", K.sum(-res_t))
        tf.summary.scalar("loss_full", K.sum(full_policy_loss_t))
        return full_policy_loss_t

    loss_args = [policy_t, action_t, advantage_t]
    policy_loss_t = Lambda(policy_loss_func, output_shape=(1,), name='policy_loss')(loss_args)

    return action_t, advantage_t, policy_loss_t


def make_run_model(input_t, conv_output_t, n_actions):
    policy_t, value_t = net_prediction(conv_output_t, n_actions)
    return Model(input=input_t, output=[policy_t, value_t])


def make_train_model(input_t, conv_output_t, n_actions):
    policy_t, value_t = net_prediction(conv_output_t, n_actions)
    action_t, advantage_t, policy_loss_t = net_loss(policy_t, value_t, n_actions)
    return Model(input=[input_t, action_t, advantage_t], output=[value_t, policy_loss_t])
