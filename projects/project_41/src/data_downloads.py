import os

def download_8k(to_dir):
    print('  => Downloading 8K Courpus...')
    raw_8k_url = 'https://nlp.stanford.edu/projects/lrec2014-stock/8K.tar.gz'
    os.system('curl ' + raw_8k_url + ' --output ' + to_dir + '8K.tar.gz')
    os.system('tar -xf ' + to_dir + '8K.tar.gz -C ' + to_dir)
    for raw_8k_file in os.listdir(to_dir + '8K-gz/'):
        if 'gz' in raw_8k_file:
            os.system('gzip -d ' + to_dir + '8K-gz/' + raw_8k_file)
    print()
    print()

def download_price_history(to_dir):
    print('  => Downloading price history...')
    raw_pricehist_url = 'https://nlp.stanford.edu/projects/lrec2014-stock/price_history.tar.gz'
    os.system('curl ' + raw_pricehist_url + ' --output ' + to_dir + 'price_history.tar.gz')
    os.system('tar -xf ' + to_dir + 'price_history.tar.gz -C ' + to_dir)
    print()
    print()

def download_eps(to_dir):
    print('  => Downloading EPS...')
    raw_eps_url = 'https://nlp.stanford.edu/projects/lrec2014-stock/EPS.tar.gz'
    os.system('curl ' + raw_eps_url + ' --output ' + to_dir + 'EPS.tar.gz')
    os.system('tar -xf ' + to_dir + 'EPS.tar.gz -C ' + to_dir)
    print()
    print()
