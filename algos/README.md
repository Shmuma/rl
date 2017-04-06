# RL algorithms
 
This dir contains implementation of various RL methods

## Asyncronous Advantage Actor-Critic (A3C)
 
 * a3c.py: minimalistic implementation, applicable to simple gym environments, like CartPole or MountainCar
 * a3c_atari.py: synchronous version with convolution nets
 * a3c_async.py: latest version with convolutions and async subprocesses.
 
 
## Other methods

* dqn.py: Q-iteration
* elite.py: variant of PG with examples filtering
* pg.py: policy gradient
