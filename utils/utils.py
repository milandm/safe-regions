import sys
import yaml


def is_debug_session():
    gettrace = getattr(sys, 'gettrace', None)
    debug_session = not ((gettrace is None) or (not gettrace()))
    return debug_session


def load_config_yml(config_file):
    try:
        with open(config_file) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            config = config['config']
            return config
    except:
        print('Config file {} is missing'.format(config_file))
        exit(1)
