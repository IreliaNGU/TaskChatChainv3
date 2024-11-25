import os
from .server import *
from .session import *
from .user import *
from utils import get_args

args = get_args("config/config.yml")
args.nlu_config = "config/nlu.yml"
os.chdir(args.base_dir)
server = Server(args)

