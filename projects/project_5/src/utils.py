import os
import json
from twarc import Twarc


def get_project_root():
    """Return the root path for the project."""
    curdir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(curdir, os.pardir))

def load_config(path):
    """Load the configuration from config."""
    with open(path, 'r') as f:
        return json.load(f)

def configure_twarc():
    """Passes api credentials into Twarc."""
    t = Twarc(
        os.getenv('CONSUMER_KEY'),
        os.getenv('CONSUMER_SECRET'),
        os.getenv('ACCESS_TOKEN'),
        os.getenv('ACCESS_TOKEN_SECRET')
    )
    return t
