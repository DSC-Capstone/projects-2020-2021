import pandas as pd
import numpy as np
import os
import json
# import brickschema
import re
import sys
sys.path.append('./src')


def run(action_arg):
    """Runs the tool"""

    if action_arg == 'clean':
        from src.utils import clean_extra_contents
        clean_extra_contents()

    elif action_arg == 'env-setup':
        os.system('chmod +x dependencies.sh')
        os.system('sh dependencies.sh')
        print('Requirements installed!')

    elif action_arg == 'all' or action_arg == 'test':
        if not os.path.isdir('../brick-builder') \
        and not os.path.isdir('../reconciliation-api'):
            run('env-setup')
        run('brickify')

    elif action_arg == 'brickify':
        from src.execute import automatic_OR
        filename = automatic_OR()

        os.system(
            'python ../brick-builder/make.py brick_builder_template.txt:' + filename)
    else:
        print('Please specify a valid argument!')


if __name__ == "__main__":
    action = 'brickify' if len(sys.argv) == 1 else sys.argv[1]
    run(action)
