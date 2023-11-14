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

def process_name(df):
    '''Process feature for plotting
    Extract name lengths'''
    movies_name = df.dropna(subset=['name'])['name']
    name_len = movies_name.apply(lambda x: len(x))
    return name_len

def process_year(df):
    '''Process feature for plotting
    Extract release year in format yyyy (int)'''
    movies_year = df.dropna(subset=['date'])['date']
    # Process dates like: yyyy-mm and yyyy-mm-dd
    movies_year = movies_year.str.replace(r'-\d{2}-\d{2}$', '', regex=True)
    movies_year = movies_year.str.replace(r'-\d{2}$', '', regex=True)
    # Convert data type to int
    movies_year = movies_year.astype(int)
    # Drop outliers
    movies_year = movies_year.drop(movies_year.index[movies_year<1800])
    return movies_year

def process_bo(df):
    '''Process feature for plotting
    Extract box-office in format (int)'''
    movies_bo = df.dropna(subset=['box_office'])['box_office']
    movies_bo = movies_bo.astype(int)
    return movies_bo

def process_runtime(df):
    '''Process feature for plotting
    Extract runtime in format (int)'''
    movies_t = df.dropna(subset=['runtime'])['runtime']
    movies_t = movies_t.astype(int)
    return movies_t

def process_lang(df):
    '''Process feature for plotting
    Extract top 15 language in format (str)'''
    movies_lang = df.dropna(subset=['lang'])['lang']
    # Convert data type to dict, to array
    movies_lang = movies_lang.apply(lambda x: ast.literal_eval(x))
    movies_lang = movies_lang.apply(lambda x: list(x.values()))
    # Get unique languages
    langs = movies_lang.explode().value_counts()
    # Top 15 languages
    langs = langs[:15]
    # Drop redundances in strings
    langs.index = langs.index.str.replace(' Language', '', regex=False)
    return langs

def process_countries(df):
    '''Process feature for plotting
    Extract top 15 countries in format (str)'''
    movies_countries = df.dropna(subset=['countries'])['countries']
    # Convert data type to dict, to array
    movies_countries = movies_countries.apply(lambda x: ast.literal_eval(x))
    movies_countries = movies_countries.apply(lambda x: list(x.values()))
    # Get unique countries
    countries = movies_countries.explode().value_counts()
    # Top 15 countries
    countries = countries[:15]
    countries = countries.rename({'United States of America': 'USA'})
    return countries

def process_genres(df):
    '''Process feature for plotting
    Extract top 15 genres in format (str)'''
    movies_genre = df.dropna(subset=['genres'])['genres']
    # Convert data type to dict, to array
    movies_genre = movies_genre.apply(lambda x: ast.literal_eval(x))
    movies_genre = movies_genre.apply(lambda x: list(x.values()))
    # Get unique genres
    genres = movies_genre.explode().value_counts()
    # Top 15 genres
    genres = genres[:15]
    return genres

def data_viz(df):
    '''Movies dataset features distributions
    For clean visualization, we simply drop nan values'''

    # Create subplots
    fig, axs = plt.subplots(3, 2, figsize=(10,10))
    axs = axs.ravel()

    # Movies language distribution: BAR
    movies_lang = process_lang(df)
    axs[0].barh(movies_lang.index, movies_lang.values)
    axs[0].set_xlabel('Nb of movies')
    axs[0].set_title('Languages (top 15)')
    axs[0].set_xscale('log')
    axs[0].grid(linestyle='--', linewidth=0.5)

    # Movies release date distribution: HIST
    movies_year = process_year(df)
    movies_year.hist(bins=movies_year.nunique(), ax=axs[1])
    axs[1].set_xlabel('Year')
    axs[1].set_ylabel('Nb of movies')
    axs[1].set_title('Release date')
    axs[1].grid(linestyle='--', linewidth=0.5)

    # Movies countries distribution: BAR
    movies_countries = process_countries(df)
    axs[2].barh(movies_countries.index, movies_countries.values)
    axs[2].set_xlabel('Nb of movies')
    axs[2].set_title('Countries (top 15)')
    axs[2].set_xscale('log')
    axs[2].grid(linestyle='--', linewidth=0.5)

    # Movies box-office distribution: PLOT
    movies_bo = process_bo(df)
    ccdf_bo_x, ccdf_bo_y = ccdf(movies_bo)
    axs[3].loglog(ccdf_bo_x, ccdf_bo_y)
    axs[3].set_xlabel('Box-office [$]')
    axs[3].set_ylabel('CCDF')
    axs[3].set_title('Box-office')
    axs[3].grid(linestyle='--', linewidth=0.5)

    # Movies genres distribution: BAR
    movies_genres = process_genres(df)
    axs[4].barh(movies_genres.index, movies_genres.values)
    axs[4].set_xlabel('Nb of movies')
    axs[4].set_title('Genres (top 15)')
    axs[4].set_xscale('log')
    axs[4].grid(linestyle='--', linewidth=0.5)

    # Movies runtime distribution: PLOT
    movies_run = process_runtime(df)
    ccdf_t_x, ccdf_t_y = ccdf(movies_run)
    axs[5].loglog(ccdf_t_x, ccdf_t_y)
    axs[5].set_xlabel('Runtime [min]')
    axs[5].set_ylabel('CCDF')
    axs[5].set_title('Runtime')
    axs[5].grid(linestyle='--', linewidth=0.5)

    plt.suptitle("Distributions of features in Movies dataset", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Movies name length
    fig, axs = plt.subplots(1, 1, figsize=(12,3))
    movies_namelen = process_name(df)
    movies_namelen.hist(bins=100)
    plt.xlabel('Name lengths')
    plt.ylabel('Nb of movies')
    plt.title('Name length')
    plt.grid(linestyle='--', linewidth=0.5)
    plt.show()

def data_clean(df):
    '''Transform data and clean to have nice input'''
    # Make list of str for genres
    df['genres'] = df.dropna(subset=['genres'])['genres'].apply(lambda x: ast.literal_eval(x)).apply(lambda x: list(x.values()))
    # Make list of str for countries
    df['countries'] = df.dropna(subset=['countries'])['countries'].apply(lambda x: ast.literal_eval(x)).apply(lambda x: list(x.values()))
    return df