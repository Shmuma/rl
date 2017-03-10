from keras.layers import Dense


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
