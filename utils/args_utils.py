import argparse
import os
from loguru import logger

from .json_utils import read_json
from .yml_utils import read_yml
from .logger import create_logger

def set_env_vars(env_vars):
    for vars in env_vars:
        name, value = vars["name"], vars["value"]
        os.environ[name] = value
        logger.info(f"Set env var {name}={value}")

def get_args(config_path: str) -> argparse.Namespace:
    """Load config from json/yaml file.

    Args:
        config_path (str): Path to config file.
    Returns:
        args (argparse.Namespace): Config as namespace.
    """
    assert os.path.exists(config_path), "Config file not found at %s" % config_path
    config_ext = os.path.splitext(config_path)[-1]
    if config_ext == ".json":
        args = read_json(config_path)
    elif config_ext == ".yml" or config_ext == ".yaml":
        args = read_yml(config_path)
    else:
        raise Exception("Invalid config file format")
    args = argparse.Namespace(**args)
    create_logger(args)
    #输出load进来的配置
    logger.info("Loaded config from %s,%s" % (config_path,args))
    if "env_vars" in args:
        set_env_vars(args.env_vars)
    return args
