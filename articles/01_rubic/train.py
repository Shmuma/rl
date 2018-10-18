#!/usr/bin/env python3
import argparse
import logging

from libcube import cubes
from libcube import model


log = logging.getLogger("train")

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cube", required=True, help="Type of cube to train, supported types=%s" % cubes.names())
    args = parser.parse_args()

    cube = cubes.get(args.cube)
    assert isinstance(cube, cubes.CubeEnv)

    log.info("Selected cube: %s", cube)

    net = model.Net(cube.encoded_shape, len(cube.action_enum))
    print(net)
