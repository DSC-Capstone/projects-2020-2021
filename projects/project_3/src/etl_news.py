import os
import numpy as np
import json

from utils import get_project_root, configure_twarc


root = get_project_root()
# Default values, will be changed in setup_paths
__news_path__ = os.path.join(root, 'config', 'news_stations.txt')
__raw_data_path__ = os.path.join(root, 'data', 'raw', 'news')
__proc_data_path__ = os.path.join(root, 'data', 'processed', 'news')


def test():
    """Custom function for purely debugging purposes, to be deleted later"""
    t = configure_twarc()

    get_users(screen_name='BBCWorld')

def setup_paths(news_path, raw_data_path, proc_data_path, **kwargs):
    """Allows for defining paths to check for files"""
    global __news_path__, __raw_data_path__, __proc_data_path__
    __news_path__ = os.path.join(root, news_path)
    __raw_data_path__ = os.path.join(root, raw_data_path)
    __proc_data_path__ = os.path.join(root, proc_data_path)

def get_news_data(n=500, **kwargs):
    """Downloads everything.

    Attempts downloading retweets for the networks, timelines of retweeters,
    and compiles retweeter timelines to a single file if compiled file
    doesn't already exist.

    Args:
        n (int): number of retweeters to observe

    """
    setup_paths(**kwargs)
    with open(__news_path__, 'r') as fh:
        for line in fh:
            screen_name = str.strip(line)
            path_compiled = os.path.join(__proc_data_path__, f'{screen_name}_{n}_users.jsonl')

            if not os.path.isfile(path_compiled):
                timeline_to_retweets(screen_name=screen_name)
                get_users(screen_name=screen_name, n=n)

def timeline_to_retweets(screen_name):
    """Downloads retweets given the user screen name.

    Compiles a text file of tweet IDs from the user's timeline. Using that,
    attempts to download the retweets.

    """
    jsonl_path = get_user_timeline(screen_name=screen_name)
    txt_path = os.path.splitext(jsonl_path)[0] + '.txt'
    jsonl_path_rts = os.path.join(__raw_data_path__, f'{screen_name}_rts.jsonl')

    if not os.path.isfile(txt_path):
        twts = []
        with open(jsonl_path, 'r') as fh:
            for line in fh:
                twt = json.loads(line)
                twts.append(twt['id_str'])
        pre_rts = np.array(twts, dtype=np.int64)
        np.savetxt(txt_path, pre_rts, fmt='%i')

    download_retweets(txt_path, jsonl_path_rts)

def get_users(screen_name, n):
    """Downloads timelines of users retweeting the user.

    Compiles a text file of user IDs that retweeted the user's tweets. Sample
    n user ids from that compilation.

    """
    path_rts = os.path.join(__raw_data_path__, f'{screen_name}_rts.jsonl')
    user_data_path = os.path.join(__raw_data_path__, f'{screen_name}_users')
    path_users = os.path.join(user_data_path, f'{screen_name}_users.txt')
    path_sample = os.path.join(user_data_path, f'{screen_name}_{n}_users.txt')
    os.makedirs(user_data_path, exist_ok=True)

    if not os.path.isfile(path_sample):
        users = set()
        if not os.path.isfile(path_users):
            with open(path_rts, 'r') as fh:
                for line in fh:
                    rt = json.loads(line)
                    users.add(rt['user']['id_str'])
            ids = np.sort(np.array(list(users), dtype=np.int64))
            np.savetxt(path_users, ids, fmt='%i')
        ids = np.genfromtxt(path_users, dtype=np.int64)
        sample = np.random.choice(ids, n, replace=False)
        np.savetxt(path_sample, sample, fmt='%i')

    compile_users(screen_name, n)

def compile_users(screen_name, n):
    """Compiles n users' timelines to a single file.

    Download all n users' timelines before compiling into a single file.

    """
    user_data_path = os.path.join(__raw_data_path__, f'{screen_name}_users')
    path_sample = os.path.join(user_data_path, f'{screen_name}_{n}_users.txt')
    path_compiled = os.path.join(__proc_data_path__, f'{screen_name}_{n}_users.jsonl')

    if not os.path.isfile(path_compiled):
        with open(path_sample) as fh:
            for line in fh:
                user_id = str.strip(line)
                jsonl_path = os.path.join(user_data_path, f'{user_id}_tweets.jsonl')
                if not os.path.isfile(jsonl_path):
                    get_user_timeline(user_id=user_id, fp=user_data_path)

        with open(path_sample, 'r') as fh, open(path_compiled, 'w') as outfile:
            for line in fh:
                user_id = str.strip(line)
                jsonl_path = os.path.join(user_data_path, f'{user_id}_tweets.jsonl')
                with open(jsonl_path) as infile:
                    for line in infile:
                        outfile.write(line)

def get_user_timeline(user_id=None, screen_name=None, fp=__raw_data_path__):
    """Retrieves user timeline data given retweet using twarc.

    Requires either user_id or screen_name, not both.

    Args:
        user_id: A user's unique id
        screen_name: A user's Twitter handle
        fp: file path for the timeline file

    """
    t = configure_twarc()
    fn = screen_name if screen_name else user_id
    jsonl_path = os.path.join(fp, f'{fn}_tweets.jsonl')

    if not os.path.isfile(jsonl_path):
        with open(jsonl_path, 'w') as outfile:
            for tweet in t.timeline(user_id=user_id, screen_name=screen_name):
                outfile.write(json.dumps(tweet) + '\n')

    return jsonl_path

def download_retweets(txt_path, jsonl_path):
    """Retrieves retweets given ID using twarc."""
    t = configure_twarc()

    if not os.path.isfile(jsonl_path):
        with open(jsonl_path, 'w') as outfile, open(txt_path, 'r') as infile:
            for tweet in infile.read().split('\n'):
                for retweet in t.retweets(list(tweet)):
                    outfile.write(json.dumps(retweet) + '\n')
