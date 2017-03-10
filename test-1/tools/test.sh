#!/bin/sh
ITER=138
./a3c_atari_play.py -r ../logs/breakout/model-${ITER}000.h5 -e Breakout-v0 -m break-${ITER}k -v
./a3c_atari_play.py -r ../logs/pong/model-${ITER}000.h5 -e Pong-v0 -m pong-${ITER}k -v
./a3c_atari_play.py -r ../logs/river/model-${ITER}000.h5 -e Riverraid-v0 -m river-${ITER}k -v
