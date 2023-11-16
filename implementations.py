'''
 File name: implementations.py
 Author: TheWestBobers
 Date created: 14/11/2023
 Date last modified: 14/11/2023
 Python Version: 3.11.4
 '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast

def ccdf(x):
    '''This function calculates the Complementary Cumulative Distribution Function (CCDF)
    of the data 'x' and prepares it for plotting.

    Parameters:
        x (array-like): The dataset for which the CCDF is to be calculated.
    
    Returns:
        ccdf_y: decreasing index, by a constant step, of the same size as x
        ccdf_x: x sorted (increasing)

    Explanation:
    when many x elements have close values, the curve will have a drop
    (because ccdf_y constantly decrease, while ccdf_x stagnate at close values)
    and when x elements have very different values, the curve will stay flat
    (because for one step, ccdf_y has a small change, and ccdf_x has a wide change)'''
    # Calculate the CCDF values.
    # 'ccdf_y' represents a decreasing index, and 'ccdf_x' contains 'x' values sorted in increasing order.
    ccdf_y = 1. - (1 + np.arange(len(x))) / len(x)
    ccdf_x = x.sort_values()

    # Return the sorted 'x' values and CCDF values.
    return ccdf_x, ccdf_y

def ax_settings(ax, xlabel='', ylabel='', title='', logx=False):
    '''Edit ax parameters for plotting'''
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logx:
        ax.set_xscale('log')
    ax.grid(linestyle='--', linewidth=0.5)

def dict_to_list(x):
    '''Convert data type (str) to dict, to list'''
    x = x.apply(lambda x: ast.literal_eval(x))
    x = x.apply(lambda x: list(x.values()))
    return x

def date_to_yyyy(x):
    '''Convert date yyyy-mm-dd (str) to yyyy (int)'''
    x = x.str.replace(r'-\d{2}-\d{2}$', '', regex=True)
    x = x.str.replace(r'-\d{2}$', '', regex=True)
    x = x.astype(int)
    return x

def top_count(x, top=15):
    rank = x.explode().value_counts()[:top]
    return rank

def data_viz(df, israw=False):
    '''Movies dataset features distributions'''

    # Create subplots
    fig, axs = plt.subplots(3, 2, figsize=(10,10))
    axs = axs.ravel()

    # Movies language distribution: BAR
    if israw:
        movies_lang = df.dropna(subset=['lang'])['lang']
        movies_lang = dict_to_list(movies_lang)
    else:
        movies_lang = df.lang
    langs = top_count(movies_lang)
    langs.index = langs.index.str.replace(' Language', '', regex=False) # Drop redundances in strings
    axs[0].barh(langs.index, langs.values)
    ax_settings(axs[0], xlabel='Nb of movies', title='Languages (top 15)', logx=True)

    # Movies release date distribution: HIST
    if israw:
        movies_date = df.dropna(subset=['date'])['date']
        movies_date = date_to_yyyy(movies_date)
        movies_date = movies_date.drop(movies_date.index[movies_date<1800]) # Drop outliers
    else:
        movies_date = df.date
    movies_date.hist(bins=movies_date.nunique(), ax=axs[1])
    ax_settings(axs[1], xlabel='Year', ylabel='Nb of movies', title='Release date')

    # Movies countries distribution: BAR
    if israw:
        movies_countries = df.dropna(subset=['countries'])['countries']
        movies_countries = dict_to_list(movies_countries)
    else:
        movies_countries = df.countries
    countries = top_count(movies_countries)
    axs[2].barh(countries.index, countries.values)
    ax_settings(axs[2], xlabel='Nb of movies', title='Countries (top 15)', logx=True)

    # Movies box-office distribution: PLOT
    if israw:
        movies_bo = df.dropna(subset=['box_office'])['box_office']
    else:
        movies_bo = df.box_office
    ccdf_bo_x, ccdf_bo_y = ccdf(movies_bo)
    axs[3].loglog(ccdf_bo_x, ccdf_bo_y)
    ax_settings(axs[3], xlabel='Box-office [$]', ylabel='CCDF', title='Box-office')

    # Movies genres distribution: BAR
    if israw:
        movies_genres = df.dropna(subset=['genres'])['genres']
        movies_genres = dict_to_list(movies_genres)
    else:
        movies_genres = df.genres
    genres = top_count(movies_genres)
    axs[4].barh(genres.index, genres.values)
    ax_settings(axs[4], xlabel='Nb of movies', title='Genres (top 15)', logx=True)

    # Movies runtime distribution: PLOT
    if israw:
        movies_run = df.dropna(subset=['runtime'])['runtime']
    else:
        movies_run = df.runtime
    ccdf_t_x, ccdf_t_y = ccdf(movies_run)
    axs[5].loglog(ccdf_t_x, ccdf_t_y)
    ax_settings(axs[4], xlabel='Runtime [min]', ylabel='CCDF', title='Runtime')

    plt.suptitle("Distributions of features in Movies dataset", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Movies name length
    fig, axs = plt.subplots(1, 1, figsize=(12,3))
    if israw:
        movies_namelen = df.dropna(subset=['name'])['name'].apply(lambda x: len(x))
    else:
        movies_namelen = df.name.apply(lambda x: len(x))
    movies_namelen.hist(bins=100)
    plt.xlabel('Name lengths')
    plt.ylabel('Nb of movies')
    plt.title('Name length')
    plt.grid(linestyle='--', linewidth=0.5)
    plt.show()

def data_missing(df):
    '''Handle missing data'''
    # Drop nan values for date, box-office, genres
    df = df.dropna(subset=['date'])
    df = df.dropna(subset=['box_office'])
    df = df.dropna(subset=['genres'])
    return df

def data_format(df):
    '''Format data types'''
    # Transform dict to list of str for lang, countries, genres
    df['lang'] = df['lang'].apply(lambda x: ast.literal_eval(x)).apply(lambda x: list(x.values()))
    df['countries'] = df['countries'].apply(lambda x: ast.literal_eval(x)).apply(lambda x: list(x.values()))
    df['genres'] = df['genres'].apply(lambda x: ast.literal_eval(x)).apply(lambda x: list(x.values()))
    # Use USA instead of United States of America
    df['countries'] = df['countries'].apply(lambda x: ['USA' if country == 'United States of America' else country for country in x])
    # Transform date to yyyy (int)
    df['date'] = df['date'].str.replace(r'-\d{2}-\d{2}$', '', regex=True)
    df['date'] = df['date'].str.replace(r'-\d{2}$', '', regex=True)
    df['date'] = df['date'].astype(int)
    return df

def data_clean(df):
    '''Clean data, outliers and features'''
    # Outliers, date before 1800
    df = df.drop(df.index[df['date']<1800])
    return df

def data_filter(df):
    '''Filter data'''
    # Keep only USA
    df = df[df.countries.apply(lambda x: 'USA' in x)]
    # Keep only english movies
    df = df[df.lang.apply(lambda x: 'English Language' in x)]
    return df

def create_subset(df, key):
    '''Creating a subset by selecting a specific genre (key)'''
    subset = df[df['genres'].apply(lambda x: key in x)]
    return subset