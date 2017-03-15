import pickle
from keras.layers import Input, Dense, Flatten
from keras.optimizers import Adam, Adagrad, RMSprop
from keras.objectives import mean_squared_error
import keras.backend as K

from algo_lib.common import make_env, summarize_gradients, summary_value, HistoryWrapper
from algo_lib.a3c import make_models
from algo_lib.player import Player, generate_batches

HISTORY_STEPS = 4
SIMPLE_L1_SIZE = 50
SIMPLE_L2_SIZE = 50

env_wrappers = (HistoryWrapper(HISTORY_STEPS),)
env = make_env("CartPole-v0", None, wrappers=env_wrappers)
state_shape = env.observation_space.shape
n_actions = env.action_space.n

in_t = Input(shape=(HISTORY_STEPS,) + state_shape, name='input')
fl_t = Flatten(name='flat')(in_t)
l1_t = Dense(SIMPLE_L1_SIZE, activation='relu', name='in_l1')(fl_t)
out_t = Dense(SIMPLE_L2_SIZE, activation='relu', name='in_l2')(l1_t)

run_model, value_policy_model = make_models(in_t, out_t, n_actions, entropy_beta=0.1)
value_policy_model.summary()

PREFIX = "49667eca-5861-4397-89c2-0fd59ff17c4c"
MODEL_NAME = PREFIX + "-nan-model.h5"
value_policy_model.load_weights(MODEL_NAME)
post_w = value_policy_model.get_weights()

MODEL_NAME = PREFIX + "-pre-model.h5"
value_policy_model.load_weights(MODEL_NAME)
pre_w = value_policy_model.get_weights()

with open(PREFIX + "-out-x.dat", "rb") as fd:
    x = pickle.load(fd)

with open(PREFIX + "-out-y.dat", "rb") as fd:
    y = pickle.load(fd)

loss_dict = {
    'value': lambda y_true, y_pred: K.sqrt(mean_squared_error(y_true, y_pred)),
    'policy_loss': lambda y_true, y_pred: y_pred
}
