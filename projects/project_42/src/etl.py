import numpy as np
import os
import sys
import pandas as pd
from pandas_profiling import ProfileReport
import re
import shutil
import requests


def download_dataset():
    # make directory
    if not os.path.exists('data'):
        os.mkdir('data')
    if os.path.exists('data/raw'):
        shutil.rmtree('data/raw')
    
    # download and unzip the dataset
    print("Downloading the dataset...")
    with open("data/MovieSummaries.tar.gz", "wb") as archive:
        url = "http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz"
        res = requests.get(url, timeout=60)
        if not res.ok:
            sys.exit(
                "Failed to connect to the data source, "
                "please download the data manually."
            )
        archive.write(res.content)
    shutil.unpack_archive("data/MovieSummaries.tar.gz", "data")

    # clean up
    os.rename("data/MovieSummaries", "data/raw")
    os.remove("data/MovieSummaries.tar.gz")


def get_data(autophrase_params, data_in, false_positive_phrases, false_positive_substrings):
    # Make data directories
    os.makedirs('data/temp', exist_ok=True)
    os.makedirs('data/out', exist_ok=True)

    # Read in raw data
    def normalize_languages(x):
        def is_utf8(value):
            try:
                value.encode()
            except UnicodeEncodeError:
                return False
            return True

        def sub(value):
            return re.sub(r' [Ll]anguages?', '', value)

        return list(np.unique([sub(value) for value in eval(x).values() if is_utf8(value)]))

    def normalize_countries(x):
        return sorted(eval(x).values())

    def normalize_genres(x):
        def sub(value):
            # Replace with a more common genre name
            if value == 'Animal Picture':
                return 'Animals'
            if value in ['Biographical film', 'Biopic [feature]']:
                return 'Biography'
            if value == 'Buddy Picture':
                return 'Buddy'
            if value == 'Comdedy':
                return 'Comedy'
            if value == 'Coming of age':
                return 'Coming-of-age'
            if value == 'Detective fiction':
                return 'Detective'
            if value == 'Education':
                return 'Educational'
            if value in ['Gay Interest', 'Gay Themed']:
                return 'Gay'
            if value == 'Gross out':
                return 'Gross-out'
            if value == 'Pornography':
                return 'Pornographic'
            if value == 'Social issues':
                return 'Social problem'
            return re.sub(' [Ff]ilms?| [Mm]ovies?', '', value)

        return list(np.unique([sub(value) for value in eval(x).values()]))

    def clean_summary(summary):
        return (
            summary
            .str.replace(r'{{.*?}}', '')  # Remove Wikipedia tags
            .str.replace(r'http\S+', '')  # Remove URLs
            .str.replace(r'\s+', ' ')  # Combine whitespace
            .str.strip()  # Strip whitespace
            .replace('', pd.NA)  # Replace empty strings with NA
        )

    movies = pd.read_csv(
        f'{data_in}/movie.metadata.tsv',
        converters={'languages': normalize_languages, 'countries': normalize_countries, 'genres': normalize_genres},
        delimiter='\t',
        header=None,
        index_col='id',
        names='id name date revenue runtime languages countries genres'.split(),
        usecols=[0, 2, 3, 4, 5, 6, 7, 8]
    ).assign(date=lambda x: pd.to_datetime(x.date, errors='coerce'))

    summaries = pd.read_csv(
        f'{data_in}/plot_summaries.txt',
        delimiter='\t',
        header=None,
        index_col='id',
        names='id summary'.split()
    ).assign(summary=lambda x: clean_summary(x.summary)).dropna()

    # Combine movie metadata and plot summaries into df
    df = movies.merge(summaries, on='id').sort_values('date').reset_index(drop=True)

    # Run AutoPhrase on plot summaries
    with open('data/temp/summaries.txt', 'w') as f:
        f.write('\n'.join(df.summary))

    autophrase_params = ' '.join([f'{param}={value}' for param, value in autophrase_params.items()])
    os.system(f'cd AutoPhrase && {autophrase_params} ./auto_phrase.sh && {autophrase_params} ./phrasal_segmentation.sh')

    # Add phrases to df
    def extract_highlighted_phrases(segmentation):
        def is_false_positive(s):
            s = s.lower()
            if len(s) == 1:  # Only 1 character
                return True
            if s in false_positive_phrases:
                return True
            for false_positive_substring in false_positive_substrings:
                if false_positive_substring in s:
                    return True
            return False

        return (
            segmentation
            .str.findall(r'<phrase>(.+?)</phrase>')
            .apply(lambda x: [s.lower() for s in x if not is_false_positive(s)])
            .apply(np.unique)
            .apply(list)
            .values
        )

    df['phrases'] = extract_highlighted_phrases(pd.read_csv(
        'model/autophrase/segmentation.txt',
        delimiter=r'\n',
        engine='python',
        header=None,
        squeeze=True
    ))

    # Export df
    df.to_pickle('data/out/data.pkl')
    ProfileReport(df).to_file('data/out/report.html')
