from keras.layers import Dense, Input, Lambda, BatchNormalization
from keras.models import Model
import keras.backend as K
import tensorflow as tf


def net_prediction(input_t, n_actions):
    """
    Make prediction part of A3C network
    :param input_t: flattened input from previous layer
    :return: policy_tensor and value_tensor
    """
    value_t = Dense(1, name='value')(input_t)
    policy_t = Dense(n_actions, name='policy')(input_t)

    return policy_t, value_t


def net_loss(policy_t, value_t, n_actions, entropy_beta=0.01):
    """
    Make traning part of the A3C network
    :param policy_t: policy tensor from prediction part
    :param value_t: value tensor from prediction part
    :param n_actions: count of actions in space
    :param entropy_beta: entropy loss scaling factor
    :return: action_t, advantage_t, policy_loss_t
    """
    action_t = Input(batch_shape=(None, 1), name='action', dtype='int32')
    reward_t = Input(batch_shape=(None, 1), name="reward")

    def policy_loss_func(args):
        p_t, v_t, act_t, rew_t = args
        log_p_t = tf.nn.log_softmax(p_t)
        sigm_p_t = K.softmax(p_t)
        oh_t = K.one_hot(act_t, n_actions)
        oh_t = K.squeeze(oh_t, 1)
        p_oh_t = K.sum(log_p_t * oh_t, axis=-1, keepdims=True)
        adv_t = (rew_t - K.stop_gradient(v_t))
        tf.summary.scalar("advantage_mean", K.mean(adv_t))
        tf.summary.scalar("advantage_rms", K.sqrt(K.mean(K.square(adv_t))))

        res_t = adv_t * p_oh_t
        entropy_t = K.sum(sigm_p_t * log_p_t, axis=-1, keepdims=True)
        full_policy_loss_t = -res_t + entropy_beta * entropy_t
        tf.summary.scalar("loss_entropy", K.mean(entropy_t))
        tf.summary.scalar("loss_policy", K.mean(-res_t))
        return full_policy_loss_t

    loss_args = [policy_t, value_t, action_t, reward_t]
    policy_loss_t = Lambda(policy_loss_func, output_shape=(1,), name='policy_loss')(loss_args)

    tf.summary.scalar("value_mean", K.mean(value_t))
    tf.summary.scalar("reward_mean", K.mean(reward_t))

    return action_t, reward_t, policy_loss_t


def make_run_model(input_t, conv_output_t, n_actions):
    policy_t, value_t = net_prediction(conv_output_t, n_actions)
    return Model(input=input_t, output=[policy_t, value_t])


def make_train_model(input_t, conv_output_t, n_actions):
    policy_t, value_t = net_prediction(conv_output_t, n_actions)
    action_t, reward_t, policy_loss_t = net_loss(policy_t, value_t, n_actions)
    return Model(input=[input_t, action_t, reward_t], output=[policy_loss_t, value_t])


def make_models(input_t, conv_output_t, n_actions, **loss_opts):
    policy_t, value_t = net_prediction(conv_output_t, n_actions)
    action_t, reward_t, policy_loss_t = net_loss(policy_t, value_t, n_actions, **loss_opts)
    run_model = Model(input=input_t, output=[policy_t, value_t])
    train_model = Model(input=[input_t, action_t, reward_t], output=[policy_loss_t, value_t])
    return run_model, train_model
