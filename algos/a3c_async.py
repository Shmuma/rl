#!/usr/bin/env python
import random
import numpy as np
import multiprocessing as mp

import time

import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adagrad


def make_model(input_size):
    in_t = Input(shape=(input_size,))
    out_t = Dense(10, activation='relu')(in_t)
    out_t = Dense(1)(out_t)

    return Model(input=in_t, output=out_t)




def player(model_queue, state_queue):
    with tf.device("/cpu:0"):
        run_model = make_model(2)
        run_model.summary()

        while True:
            time.sleep(1)
            r = run_model.predict_on_batch(np.array([[100, 100]]))
            print(r)
            if not model_queue.empty():
                weights = model_queue.get()
                if weights is None:
                    print("Stop requested, exit")
                    break
                run_model.set_weights(weights)
                print("New model received")
    pass



if __name__ == "__main__":
    train_model = make_model(2)

    train_model.summary()
    train_model.compile(optimizer=Adagrad(), loss='mse')

    model_queue = mp.Queue()
    state_queue = mp.Queue()

    p = mp.Process(target=player, args=(model_queue, state_queue))
    p.start()

    model_idx = 0
    min_loss = None
    for iter_idx in range(1000):
        time.sleep(0.1)
        a = random.randrange(100)
        b = random.randrange(100)
        l = train_model.train_on_batch(np.array([[a, b]]), np.array([a+b]))
        if min_loss is None or l < min_loss:
            min_loss = l
            print("%d: min loss %f" % (iter_idx, min_loss))
            model_queue.put(train_model.get_weights())

    p.join()

    pass
