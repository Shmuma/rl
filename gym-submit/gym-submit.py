#!/usr/bin/env python
import os
import sys
import configparser
import argparse
import gym


ENV_VAR_NAME = 'OPENAI_GYM_KEY'
CONF_FILE_NAME = '~/.config/gym-submit.conf'
CONF_SECTION_NAME = 'gym-submit'
CONF_VALUE_NAME = 'Key'


def look_for_key():
    env_key = os.environ.get(ENV_VAR_NAME)
    if env_key is not None:
        return env_key

    conf_path = os.path.expanduser(CONF_FILE_NAME)
    if os.path.exists(conf_path):
        conf = configparser.ConfigParser()
        conf.read(conf_path)
        if CONF_SECTION_NAME in conf.sections():
            key = conf[CONF_SECTION_NAME].get(CONF_VALUE_NAME)
            if key:
                return key

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs='+', help="Directories to submit")
    parser.add_argument("-k", "--key", help="Submission key. If not provided, we'll check {env_name} "
                                            "and config {conf_name}".format(env_name=ENV_VAR_NAME,
                                                                            conf_name=CONF_FILE_NAME))
    args = parser.parse_args()

    if args.key is not None:
        key = args.key
    else:
        key = look_for_key()

    # if nothing have found, complain about it
    if key is None:
        print("""No OpenAI Gym key was provided. You can specify it:
        1. as -k argument,
        2. with {env_name} environment variable,
        3. put in file {conf_name} under section '{section_name}' and '{value_name}' value, like in example:

        [{section_name}]
        {value_name}=YOUR_KEY
        """.format(env_name=ENV_VAR_NAME, conf_name=CONF_FILE_NAME, section_name=CONF_SECTION_NAME,
                   value_name=CONF_VALUE_NAME))
        sys.exit(-1)

    for dir in args.dirs:
        gym.upload(dir, api_key=key)
