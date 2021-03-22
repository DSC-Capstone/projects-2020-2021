from bar_chart_race import bar_chart_race
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MultipleLocator
from math import ceil
import numpy as np
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer


def example_dataset_row(df, data_out, example_movie):
    """
    Save a JSON file with the row of the dataset corresponding to the given movie name
    """
    row = df[df.name == example_movie]
    if len(row) == 0:
        raise ValueError(f'{example_movie} is not in the dataset')
    if len(row) > 1:
        raise ValueError(f'More than one movie in the dataset has the name {example_movie}')
    row.squeeze().to_json(f'{data_out}/example_dataset_row.json', date_format='iso')


def number_movies_per_year_bar_chart(df, data_out, dpi, **kwargs):
    """
    Save a Figure with a bar chart of the number of movies per year.
    """
    fig, ax = plt.subplots(figsize=(7, 1.1))
    number_movies_per_year = df.year.value_counts().sort_index()
    number_movies_per_year.plot.bar(ax=ax)
    ax.set(xlabel='Year', ylabel='# Movies')
    ax.tick_params(rotation=0)
    ax.xaxis.set_major_locator(FixedLocator([i for i, year in enumerate(number_movies_per_year.index) if year % 10 == 0]))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    fig.savefig(f'{data_out}/number_movies_per_year_bar_chart.png', dpi=dpi, bbox_inches='tight')


def highest_grossing_movie_containing_phrase(df, phrase):
    """
    Return the highest-grossing movie containing the given phrase.
    Break ties by picking the movie with the earliest release date.
    """
    return df[df.phrases.apply(lambda x: phrase in x)].sort_values(['revenue', 'date'], ascending=[False, True]).iloc[0]


def phrase_tfidfs_by_year(df, year_start, year_end, phrase_count_threshold, **kwargs):
    """
    Return a DataFrame with the tf-idf of each phrase for each year (phrases are terms and years are documents).
    """
    # Create a Series with a list of phrases (allowing duplicates) for each year
    phrases_by_year = df.query(f'{year_start} <= date.dt.year <= {year_end}').groupby('year').phrases.agg(lambda x: sum(x, []))

    # Create a DataFrame with the count of each phrase for each year
    phrase_counts_by_year = pd.DataFrame(phrases_by_year.apply(Counter).tolist(), index=phrases_by_year.index).fillna(0).sort_index(1)
    # Only include phrases that appear at least `phrase_count_threshold` times total
    phrase_counts_by_year = phrase_counts_by_year.loc[:, phrase_counts_by_year.sum().ge(phrase_count_threshold)]

    # Create a DataFrame with the tf-idf of each phrase for each year
    tfidfs = pd.DataFrame(
        TfidfTransformer(sublinear_tf=True).fit_transform(phrase_counts_by_year).toarray(),
        index=phrase_counts_by_year.index,
        columns=phrase_counts_by_year.columns
    )
    return tfidfs


def phrase_tfidfs_by_decade(df, decade_start, decade_end, phrase_count_threshold, **kwargs):
    """
    Return a DataFrame with the tf-idf of each phrase for each decade (phrases are terms and decades are documents).
    """
    # Create a Series with a list of phrases (allowing duplicates) for each decade
    phrases_by_decade = df.query(f'{decade_start} <= date.dt.year <= {decade_end}').assign(decade=lambda x: x.year // 10 * 10).groupby('decade').phrases.agg(lambda x: sum(x, []))

    # Create a DataFrame with the count of each phrase for each decade
    phrase_counts_by_decade = pd.DataFrame(phrases_by_decade.apply(Counter).tolist(), index=phrases_by_decade.index).fillna(0).sort_index(1)
    # Only include phrases that appear at least `phrase_count_threshold` times total
    phrase_counts_by_decade = phrase_counts_by_decade.loc[:, phrase_counts_by_decade.sum().ge(phrase_count_threshold)]

    # Create a DataFrame with the tf-idf of each phrase for each decade
    tfidfs = pd.DataFrame(
        TfidfTransformer(sublinear_tf=True).fit_transform(phrase_counts_by_decade).toarray(),
        index=phrase_counts_by_decade.index,
        columns=phrase_counts_by_decade.columns
    )
    return tfidfs


def top_phrases_by_year_bar_chart(df, data_out, stop_words, dpi, **kwargs):
    """
    Save a Figure with a bar chart of the top phrases (ranked by tf-idf) for each year.
    """
    tfidfs = phrase_tfidfs_by_year(df, **kwargs).drop(columns=stop_words, errors='ignore')

    ncols = 5
    nrows = ceil(len(tfidfs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, sharex=True, figsize=(ncols * 2.5, nrows * 1.9), constrained_layout=True)
    axes = axes.flatten()
    for year, ax in zip(tfidfs.index, axes):
        tfidfs.loc[year].nlargest(10)[::-1].plot.barh(ax=ax)
        ax.set(title=year, xlabel='tf-idf')
        ax.tick_params(bottom=False, labelbottom=False)
    for ax in axes[len(tfidfs):]:
        ax.axis('off')
    fig.savefig(f'{data_out}/top_phrases_by_year_bar_chart.png', dpi=dpi, bbox_inches='tight')


def top_phrases_by_year_bar_chart_race(df, data_out, stop_words, n_bars, dpi, fps, seconds_per_period, **kwargs):
    """
    Save an MP4 with a bar chart race of the top phrases (ranked by tf-idf) for each year.
    """
    tfidfs = phrase_tfidfs_by_year(df, **kwargs).drop(columns=stop_words, errors='ignore')
    # Only keep columns for the top phrases to reduce unnecessary computation
    tfidfs = tfidfs[np.unique(tfidfs.apply(lambda x: x.nlargest(n_bars).index.tolist(), 1).sum())]

    # Prepare figure
    fig, ax = plt.subplots(dpi=dpi, figsize=(6, 3.5))
    fig.suptitle(f'Top Phrases From Wikipedia Movie Plot Summaries\n{tfidfs.index[0]}–{tfidfs.index[-1]}', y=.94)
    ax.set_facecolor('.9')
    ax.set_xlabel('tf-idf', style='italic')
    ax.tick_params(labelbottom=False, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=.4, right=.95, top=.82)

    # Create bar chart race
    bar_chart_race(
        df=tfidfs,
        fig=fig,
        filename=f'{data_out}/top_phrases_by_year_bar_chart_race.mp4',
        label_bars=False,
        n_bars=n_bars,
        period_fmt='{x:.0f}',
        period_label=dict(x=.96, y=.04, ha='right', va='bottom', size=30),
        period_length=seconds_per_period * 1000,
        steps_per_period=int(seconds_per_period * fps),
    )


def top_phrases_by_decade_bar_chart(df, data_out, stop_words, movie_name_overflow, dpi, compact, **kwargs):
    """
    Save a Figure with a bar chart of the top phrases (ranked by tf-idf) for each decade.
    If not compact, annotate each phrase with its corresponding highest-grossing movie within the decade.
    """
    if compact:
        tfidfs = phrase_tfidfs_by_decade(df, **kwargs).drop(columns=stop_words, errors='ignore')

        ncols = 4
        nrows = ceil(len(tfidfs) / ncols)
        fig, axes = plt.subplots(nrows, ncols, sharex=True, figsize=(ncols * 2.1, nrows * 1.9), constrained_layout=True)
        axes = axes.flatten()
        for decade, ax in zip(tfidfs.index, axes.flatten()):
            tfidfs.loc[decade].nlargest(10)[::-1].plot.barh(ax=ax)
            ax.set_title(f'{decade}s', size=15, x=0)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(bottom=False, labelbottom=False)
        for ax in axes[len(tfidfs):]:
            ax.axis('off')
        fig.savefig(f'{data_out}/top_phrases_by_decade_bar_chart_compact.png', dpi=dpi, bbox_inches='tight')

    else:
        def format_movie_name(movie, movie_name_overflow):
            name = movie['name'] if len(movie['name']) <= movie_name_overflow else movie['name'][:movie_name_overflow - 3].strip() + '...'
            return f'{name} ({movie.year})'

        tfidfs = phrase_tfidfs_by_decade(df, **kwargs).drop(columns=stop_words, errors='ignore')

        ncols = 2
        nrows = ceil(len(tfidfs) / ncols)
        fig, axes = plt.subplots(nrows, ncols, sharex=True, figsize=(ncols * 3.2, nrows * 2.5))
        axes = axes.flatten()
        for decade, ax in zip(tfidfs.index, axes.flatten()):
            # Plot top phrases
            phrases = tfidfs.loc[decade].nlargest(10)[::-1]
            phrases.plot.barh(ax=ax)

            # Annotate with corresponding movie names
            df_decade = df.query(f'{decade} <= date.dt.year < {decade + 10}')
            movie_names = [
                format_movie_name(highest_grossing_movie_containing_phrase(df_decade, phrase), movie_name_overflow)
                for phrase in phrases.index]
            for rect, movie_name in zip(ax.patches, movie_names):
                ax.annotate(
                    movie_name,
                    xy=(rect.get_width(), rect.get_y() + rect.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    size=7.5,
                    ha='left',
                    va='center',
                    style='italic'
                )

            ax.set(title=f'{decade}s', xlabel='tf-idf')
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(bottom=False, labelbottom=False)
        for ax in axes[len(tfidfs):]:
            ax.axis('off')
        fig.subplots_adjust(wspace=6)
        fig.savefig(f'{data_out}/top_phrases_by_decade_bar_chart.png', dpi=dpi, bbox_inches='tight')


def generate_figures(data_in, data_out, example_movie, **kwargs):
    """
    Generate figures.
    """
    # Read in data, add new columns as needed
    df = pd.read_pickle(data_in)
    df['year'] = df.date.dt.year.astype('Int64')

    # Generate figures
    os.makedirs(data_out, exist_ok=True)
    example_dataset_row(df, data_out, example_movie)
    number_movies_per_year_bar_chart(df, data_out, **kwargs)
    top_phrases_by_year_bar_chart(df, data_out, **kwargs)
    top_phrases_by_year_bar_chart_race(df, data_out, **kwargs)
    top_phrases_by_decade_bar_chart(df, data_out, **kwargs)
