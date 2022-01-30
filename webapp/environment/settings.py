import os
import pathlib
from os.path import dirname, join

import pandas as pd
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), os.getenv("ENVIRONMENT_FILE"))
load_dotenv(dotenv_path=dotenv_path, override=True)

APP_HOST = os.environ.get("HOST")
APP_PORT = int(os.environ.get("PORT"))
APP_DEBUG = bool(os.environ.get("DEBUG"))
DEV_TOOLS_PROPS_CHECK = bool(os.environ.get("DEV_TOOLS_PROPS_CHECK"))

UPLOAD_FOLDER_ROOT = join(os.getcwd(), "cached_files")
LOADING_TYPE = "default"
pathlib.Path(UPLOAD_FOLDER_ROOT).mkdir(parents=True, exist_ok=True)
