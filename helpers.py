'''
 File name: helpers.py
 Author: TheWestBobers
 Date created: 04/12/2023
 Date last modified: 22/12/2023
 Python Version: 3.11.4
 '''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import nltk
import networkx as nx
import math
import re
import statsmodels.api as sm
import string
import json

import pyspark.pandas as ps
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize

import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import itertools


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

def generate_missing_info(df):
    """
    Generate a DataFrame containing information about missing data in each column of the given DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with columns 'Column' and 'Missing Data (%)'.
    """
    missing_percentage = (df.isna().mean() * 100).round(2)

    missing_info = pd.DataFrame({
        'Column': missing_percentage.index,
        'Missing Data (%)': missing_percentage.values
    }).set_index("Column")

    return missing_info

def check_doublons(df, col_check, year, runtime):
    for c in col_check:
        duplicates = df[df.duplicated([c, year, runtime], keep=False)]  
        if not duplicates.empty:
            print(f'Rows with real duplicates: ')
            print(duplicates[[c, year, runtime]])
            print('-' * 80)
        else:
            print(f'No duplicates')
            print('-' * 80)
    return None


def fuse_duplicates(df, col_check, year, runtime, col_len, col_null):
    df_clean = df.copy(deep=True)
    df_clean[runtime] = df_clean[runtime].fillna(-1)
    for c in col_check:
        duplicates = df_clean[df_clean.duplicated([c, year, runtime], keep=False)]  
        if not duplicates.empty:
            print(f'Fusing duplicates: ')

            for index, group in duplicates.groupby([c, year, runtime]):
                if len(group) > 1:
                    higher_index = group.index.max()
                    lower_index = group.index.min()
                    # Fuse 'languages', 'countries', 'genres'
                    for col in col_len:
                        if len(group.loc[higher_index, col]) > len(group.loc[lower_index, col]):
                            df_clean.at[lower_index, col] = group.loc[higher_index, col]
                    # Fuse 'release_month', 'box_office_revenue', 'runtime'
                    for col in col_null:
                        if pd.isnull(group.loc[lower_index, col]) and not pd.isnull(group.loc[higher_index, col]):
                            df_clean.at[lower_index, col] = group.loc[higher_index, col]
                        elif not pd.isnull(group.loc[lower_index, col]) and not pd.isnull(group.loc[higher_index, col]):
                            if group.loc[lower_index, col] != group.loc[higher_index, col]:
                                # Calculate mean if values are different
                                mean_value = group.loc[:, col].mean()
                                df_clean.at[lower_index, col] = mean_value

                    df_clean = df_clean.drop(higher_index)

            print('Duplicates fused successfully.')
            print('-' * 80)
        else:
            print(f'No duplicates')
            print('-' * 80)
    
    df_clean[runtime] = df_clean[runtime].replace(-1, pd.NA)
    return df_clean.reset_index(drop=True)

# def separate_values_biased(df, col, target):
#     new_cols = df[col].str.split(', ', expand=True).rename(columns=lambda x: f"{col}_{x+1}")
#     usa_column = new_cols.apply(lambda row: target in row.values, axis=1)
#     df[col] = np.where(usa_column, target, new_cols.iloc[:, 0]) 
#     return df

def separate_values_biased(df, col, target):
    def choose_country(country_list):
        if target in country_list:
            return target
        elif country_list:
            return country_list[0]
        else:
            return np.nan 
    df[col] = df[col].apply(lambda x: choose_country(eval(x) if isinstance(x, str) else x))
    return df

def calculate_missing_percentage(df, groupby_column, target_column):
    missing_percentage = df.groupby(groupby_column)[target_column].apply(lambda x:                                                        (x.isnull().sum() / len(x)) * 100).reset_index().set_index(groupby_column)
    return missing_percentage

def fuse_columns(x, y, column_name):
    if pd.notna(x) and pd.notna(y):
        # Both entries are present
        if x == y:
            # Entries are the same
            return x
        else:
            # Take the mean of the entries
            return (x + y) / 2
    elif pd.notna(x):
        # x is present, y is missing
        return x
    elif pd.notna(y):
        # y is present, x is missing
        return y
    else:
        # Both entries are missing
        return pd.NA

def fuse_scores(df, score_col1, score_col2, votes_col1, votes_col2):
    # Create a new column for fused scores
    numerator = (df[score_col1].fillna(0) * df[votes_col1].fillna(0) +
                 df[score_col2].fillna(0) * df[votes_col2].fillna(0))
    
    denominator = df[votes_col1].fillna(0) + df[votes_col2].fillna(0)

    # Avoid division by zero
    df['review'] = numerator / denominator.replace(0, float('nan'))

    # Create a new column for fused votes, including NaN when the sum is zero
    df['nbr_review'] = df[votes_col1].fillna(0) + df[votes_col2].fillna(0)
    df['nbr_review'] = df['nbr_review'].replace(0, float('nan'))

    # Drop the unnecessary columns
    df = df.drop([score_col1, score_col2, votes_col1, votes_col2], axis=1)
    return df
    
def ax_settings(ax, xlabel='', ylabel='', title='', logx=False, logy=False):
    '''Edit ax parameters for plotting'''
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
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
    fig, axs = plt.subplots(2, 2, figsize=(10,10))
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


def data_missing(df):
    '''Handle missing data'''
    # Drop nan values for date, box-office, genres
    df = df.dropna(subset=['date'])
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
    df = df.drop(df.index[df['year']<1800])
    # Redundant movies
    
    return df

def data_filter(df):
    '''Filter data'''
    # Keep only USA
    df = df[df.countries.apply(lambda x: 'USA' in x)]
    # Keep only english movies
    df = df[df.lang.apply(lambda x: 'English Language' in x)]
    # Keep only movies from 1910 to 2010
    df = df[(df.year>=1910) & (df.year<=2010)]
    return df

def create_subset(df, key):
    '''Creating a subset by selecting a specific genre (key)'''
    subset = df[df['genres'].apply(lambda x: key in x)]
    return subset

parse_json = lambda s: json.loads(s)

parse_json = lambda s: json.loads(s)
def parse_genre(movies):
    """
    Processes the 'genres' column in a movies DataFrame.
    """
    movies["genres"] = movies["genres"].apply(parse_json)
    movies["genres_dict"] = movies["genres"].copy()
    movies["genres"] = movies["genres"].apply(lambda d: list(d.values()) if isinstance(d, dict) else d) 

def date_conversion( movies):
    dates_copy = movies['date'].copy()
    movies['date'] = pd.to_datetime(movies['date'], errors='coerce')
    # Attempt to convert the 'date' column to datetime objects
    movies['date'] = pd.to_datetime(dates_copy.copy(), errors='coerce', format='%Y-%m-%d')
    
    # For entries where the conversion failed (which should be the ones with only the year), convert using just the year
    movies.loc[movies['date'].isna(), 'date'] = pd.to_datetime(dates_copy.copy()[movies['date'].isna()], errors='coerce', format='%Y')
    
    # Extract the year from the datetime objects
    movies['year'] = movies['date'].dt.year

def get_initial_data(PATH_HEADER):
    """
    Reads and processes various datasets related to movies, characters, and TV tropes.

    This function loads data from four different files located at a specified path:
    - Character metadata (TSV format)
    - Movie metadata (TSV format)
    - Plot summaries (TXT format)
    - TV Tropes clusters (TXT format)

    Each dataset is read and processed into a pandas DataFrame with specified column names.
    The plot summaries and TV Tropes data are read as raw text files, while the character and movie
    metadata are read as tab-separated values (TSV).

    Parameters:
    PATH_HEADER (str): The base path where the data files are located. It is concatenated
                       with file names to access each dataset.

    Returns:
    tuple: A tuple containing four pandas DataFrames in the order:
           - summaries (DataFrame): Plot summaries with Wikipedia and Freebase IDs.
           - characters (DataFrame): Detailed character information including name, actor details, and metadata.
           - tvtropes (DataFrame): TV Tropes data categorized with additional JSON parsed information.
           - movies (DataFrame): Movie metadata including ID, name, date, revenue, runtime, and other attributes.
    """
    CHARACTER = "character.metadata.tsv"
    MOVIE = "movie.metadata.tsv"
    PLOT_SUMMARIES = "plot_summaries.txt"
    TVTROPES = "tvtropes.clusters.txt" 
    column_names = [
        "WikipediaID",
        "FreebaseID",
    ]
    summaries = pd.read_csv(PATH_HEADER + PLOT_SUMMARIES, names=column_names, header=None, delimiter="\t")
    file_path = PATH_HEADER + TVTROPES 
    parsed_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file: #parse json
            category, json_str = line.split('\t', 1)
            json_data = json.loads(json_str)   
            json_data['category'] = category  
            parsed_data.append(json_data)   
    
    tvtropes = pd.DataFrame(parsed_data)
    column_names = [
        "WikipediaID",
        "FreebaseID",
        "date",
        "Character Name",
        "Actor DOB",
        "Actor Gender",
        "Actor Height",
        "Actor Ethnicity",
        "Actor Name",
        "Actor Age at Movie Release",
        "Freebase Character Map",
        "what",
        "wut"
        ]
    characters = pd.read_csv(PATH_HEADER + CHARACTER, delimiter='\t', header=None, names=column_names)
    column_names = [
    "wikipediaID",
    "freebaseID",
    "name",
    "date",
    "revenue",
    "runtime",
    "languages",
    "countries",
    "genres"
    ]
    movies = pd.read_csv(PATH_HEADER + MOVIE, sep='\t', names=column_names, header=None)
    parse_genre(movies)
    date_conversion(movies)
    movies = movies.dropna()
    return summaries, characters, tvtropes, movies

def trope_originators(tvtropes, movies):
    """
    Identifies the first movies associated with each TV Trope category based on release dates.

    This function merges two DataFrames: 'tvtropes', containing TV tropes and associated movies, and 
    'movies', containing movie names and their release dates. It aims to find the earliest movie 
    associated with each trope category.

    The process involves:
    1. Renaming the 'movie' column in 'tvtropes' to 'name' for consistency.
    2. Merging 'tvtropes' with 'movies' on the 'name' column to associate movies with their release dates.
    3. Dropping rows where the release date is missing or invalid.
    4. Sorting the resulting DataFrame by trope category and release date.
    5. Selecting the first occurrence of each trope category.

    The final result is displayed, showing the earliest movie for each TV trope category.

    Parameters:
    tvtropes (DataFrame): A pandas DataFrame with TV tropes data. It must have a column named 'movie'.
    movies (DataFrame): A pandas DataFrame with movie data. It must have columns named 'name' and 'date'.

    Returns:
    None: The function does not return a value. It displays the result directly using the `display` function.
    """
    tvtropes["name"] = tvtropes["movie"]
    tvtropes_with_dates = pd.merge(tvtropes, movies[["name","date"]], on='name', how='left')
    # Exclude rows where 'date' is NaT (not a time) or missing
    tvtropes_copy = tvtropes.copy() 
    tvtropes_with_dates_copy = tvtropes_with_dates.copy()
    tvtropes = tvtropes_with_dates
    tvtropes = tvtropes.dropna(subset=['date'])
    
    # Continue with the rest of the process as before
    tvtropes_sorted = tvtropes.sort_values(by=['category', 'date'])
    first_movies_per_category = tvtropes_sorted.drop_duplicates(subset='category', keep='first')
    pd.set_option('display.max_rows', None)
    display(first_movies_per_category[['category', 'name', 'date']])
    pd.reset_option('display.max_rows')


def generate_years_list(start_year, end_year):
    """
    Generates a list of years from the start year to the end year, inclusive.

    Parameters:
    start_year (int): The year to start the list.
    end_year (int): The year to end the list. Must be greater than or equal to start_year.

    Returns:
    list: A list of years from start_year to end_year, inclusive.
    """
    if start_year > end_year:
        raise ValueError("End year must be greater than or equal to start year")

    return [year for year in range(start_year, end_year + 1)]

def get_genre_counts(movies):
    """
    Counts the occurrences of each genre by year in a given movie dataset.
    This function processes a DataFrame containing movie data, specifically focusing on the 'genres' column.
    
    Parameters:
    movies (DataFrame): A pandas DataFrame with movie data. It must include 'year' and 'genres' columns, 
                        where 'genres' contains lists of genres for each movie.

    Returns:
    Series: A pandas Series with a multi-level index (year, genre). Each value in the Series represents 
            the count of a particular genre in a specific year.
    """
    df = movies.copy()
    movies_exploded = df.explode('genres').copy()
    # Explode the 'genres' list into separate rows
    df_exploded = df.explode('genres')
    # Group by year and genre, then count occurrences
    genre_counts = df_exploded.groupby(['year', 'genres']).size()
    genre_counts.reset_index
    return genre_counts

def get_genre_counts_dataframe(movies):
    """
    Generates a DataFrame that provides a breakdown of movie genres by year, including their counts and percentage of total movies for each year.
    Returns:
    DataFrame: A DataFrame with columns 'year', 'genres', 'count', and 'percentage'. The 'count' column indicates the number of movies in each genre for a 
    given year, and 'percentage' shows the proportion of movies in that genre compared to the total movies in that year.

    Example of returned DataFrame structure:
        year    genres       count    percentage
        2000    Action       50       25.0
        2000    Comedy       150      75.0
    """
    df = movies.copy()
    movies_exploded = df.explode('genres').copy()
    # Explode the 'genres' list into separate rows
    df_exploded = df.explode('genres')
    # Group by year and genre, then count occurrences
    genre_counts = df_exploded.groupby(['year', 'genres']).size()
    movies_copy = genre_counts.copy(deep = True)
    # Convert the Series into a DataFrame
    genre_counts_df = genre_counts.reset_index(name='count')
    
    # Calculate the total count for each year
    total_counts = genre_counts_df.groupby('year')['count'].transform('sum')
    
    # Calculate the percentage
    genre_counts_df['percentage'] = (genre_counts_df['count'] / total_counts) * 100
    return genre_counts_df

def plot_genres_percentages_per_year(movies, start_year, end_year, start_popularity,end_popularity,ylim):
    """
    Analyzes and visualizes the popularity of movie genres over a specified range of years.

    This function takes a DataFrame of movies, explodes the 'genres' column to count occurrences
    of each genre per year, and then visualizes the percentage distribution of the top genres 
    based on their popularity rankings within the specified year range.

    The visualization is a stacked bar chart showing the proportional representation of each 
    genre per year.

    Parameters:
    movies (DataFrame): A DataFrame containing movie data. Must include 'year' and 'genres' columns.
    start_year (int): The starting year for the analysis.
    end_year (int): The ending year for the analysis.
    start_popularity (int): The starting index for selecting top genres based on popularity.
    end_popularity (int): The ending index for selecting top genres based on popularity.
    ylim(int): 
    
    Returns:
    None: This function does not return anything. It plots the results directly using matplotlib.

    Note:
    - The function relies on an external function 'generate_years_list' to create a list of years.
    - The function relies on an external function 'get_genre_counts' to get the genre counts.
    - It assumes that the 'genres' column in the input DataFrame is a list of genres for each movie.
    """
    genre_counts = get_genre_counts(movies)
    # Convert the Series into a DataFrame
    genre_counts_df = genre_counts.reset_index(name='count')
    
    # Calculate the total count for each year
    total_counts = genre_counts_df.groupby('year')['count'].transform('sum')
    
    # Calculate the percentage
    genre_counts_df['percentage'] = (genre_counts_df['count'] / total_counts) * 100
    years_to_plot = generate_years_list(start_year, end_year)
    data_to_plot = genre_counts_df[genre_counts_df['year'].isin(years_to_plot)]
    
    top_genres = genre_counts_df['genres'].value_counts().iloc[ start_popularity:end_popularity].index
    
    
    data_filtered = data_to_plot[data_to_plot['genres'].isin(top_genres)]
    
    
    data_sorted = data_filtered.sort_values(['year', 'genres'])
    
    unique_genres = data_sorted['genres'].unique()
    unique_years = data_sorted['year'].unique()
    
    genre_colors = {genre: plt.cm.tab20(i / len(unique_genres)) for i, genre in enumerate(unique_genres)}
    
    fig, ax = plt.subplots(figsize=(18, 8))
    
    
    for year in unique_years: # Create a stacked bar chart
        bottom = np.zeros(len(unique_years))  # Initialize the bottom array for this year
        year_data = data_sorted[data_sorted['year'] == year]
    
        for genre in unique_genres:
            genre_data = year_data[year_data['genres'] == genre]
            if not genre_data.empty:
                percentage = genre_data['percentage'].values[0]
                ax.bar(str(year), percentage, bottom=bottom, color=genre_colors[genre], label=genre)
                bottom += percentage
    
    
    ax.set_ylim(0, ylim)
    
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Percentage')
    ax.set_title('Stacked Genre Percentages by Year for Top 15-30 Genres')
    
    ax.legend(genre_colors.keys(), title="Genres", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    plt.show()

def print_highest_revenue_in_genre_period(data, genre, year):
    """
    Prints the highest revenue movie of a given genre between 10 years and 4 years before a specified year.

    Parameters:
    data (DataFrame): The DataFrame containing movie data with 'genres', 'year', and 'revenue' columns.
    genre (str): The movie genre.
    year (int): The reference year.
    """
    # Filter the DataFrame for the given genre and within the specified year range
    start_year = year - 10
    end_year = year - 4
    
    df_filtered = data.copy()[(data["genres"] == genre)  
                       & (data["year"] > start_year) 
                       & (data["year"] <= end_year)
    ]
    # Check if there are movies in the filtered DataFrame
    if not df_filtered.empty:
        # Find the movie with the highest revenue
        highest_revenue_row = df_filtered.loc[df_filtered['revenue'].idxmax()]
        print(f"Highest revenue movie in '{genre}' genre between {start_year} and {end_year}:")
        print(highest_revenue_row["name"])
    else:
        print(f"No movies found in '{genre}' genre between {start_year} and {end_year}.")

def hype_generators(movies,start_year, end_year, start_popularity,end_popularity):
    """
    This function identifies genres of movies that have shown the highest percentage change in popularity
    within a specified time period and among a specified range of popularity rankings.

    Parameters:
    movies (DataFrame): A pandas DataFrame containing movie data.
    start_year (int): The starting year for the analysis.
    end_year (int): The ending year for the analysis.
    start_popularity (int): The starting rank for considering popularity of genres.
    end_popularity (int): The ending rank for considering popularity of genres.

    The function performs the following steps:
    1. Calculates the percentage change in popularity for each movie genre from year to year.
    2. Filters the data to include only the years between start_year and end_year.
    3. Determines the change in popularity for each genre and drops any NaN values that may arise.
    4. Focuses on genres that are ranked between start_popularity and end_popularity in terms of frequency.
    5. Identifies the genre with the highest percentage change per year.
    6. Prints a summary of the genres with the highest change in percentage between the specified years,
    and highlights these genres as potential 'hype generators'.

    Returns:
    None: This function prints the results to the console and does not return any value.
    """
    # Calculate percentage change for each genre from year to year
    genre_counts_df = get_genre_counts_dataframe(movies)
    # Drop NaN values that result from the first instance of each genre
    genre_counts_df = genre_counts_df[(genre_counts_df["year"] > start_year) & (genre_counts_df["year"] < end_year)]
    genre_counts_df['percentage_change'] = genre_counts_df.groupby('genres')['percentage'].pct_change()
    genre_counts_df = genre_counts_df.dropna()
    #print(genre_counts_df)
    # Select the genres ranked from 20th to 40th
    genre_value_counts= genre_counts_df['genres'].value_counts()
    top_genres = genre_value_counts.iloc[start_popularity:end_popularity].index
    genre_counts_df = genre_counts_df[genre_counts_df["genres"].isin(top_genres)]
    # Find the genre with the highest percentage change per year
    highest_change_per_year = genre_counts_df.loc[genre_counts_df.groupby('genres')['percentage_change'].idxmax()]
    highest_change_per_year = highest_change_per_year.reset_index()
    print("Highest change in percentage by genre between 1990 and  2013 (hype))")
    print(highest_change_per_year.to_string(index=False))
    print("Hype generator by genre: ")
    movies_exploded = movies.copy().explode('genres')
    for index, row in highest_change_per_year.iterrows():
        print_highest_revenue_in_genre_period(movies_exploded,row["genres"], row["year"])

#-------------------------------------------------------------------------------------------------------
# ARTHUR
def select_subsets(movies, min_len=0):
    '''Return subsets of the 'movies' dataset, by genres
    don't use genres with too few movies'''
    all_genres = list(set(itertools.chain.from_iterable(movies.genres.tolist())))
    all_genres.sort()
    subsets = [(g, create_subset(movies,g)) for g in all_genres]
    subsets = [element for element in subsets if len(element[1])>min_len]
    return subsets

def select_subsets_double(subsets, min_len=0):
    '''Return subsets of the 'movies' dataset, by genres
    don't use genres with too few movies'''
    subsets_double = []
    for i, s in enumerate(subsets):
        genres_double = list(set(itertools.chain.from_iterable(s[1].genres.tolist()))-set([s[0]]))
        genres_double.sort()
        s_double = [((s[0], g), create_subset(s[1],g)) for g in genres_double]
        subsets_double.append(s_double)
    
    subsets_double = [s for s_double in subsets_double for s in s_double]
    subsets_double = [s for s in subsets_double if len(s[1])>=min_len]
    subsets_double_unique = []
    unique_combinations = set()
    for (str1, str2), df in subsets_double:
        # Sort the tuple to ensure ('Comedy', 'Action') is treated the same as ('Action', 'Comedy')
        sorted_tuple = tuple(sorted((str1, str2)))
        if sorted_tuple not in unique_combinations:
            # Add the sorted tuple to the set of unique combinations
            unique_combinations.add(sorted_tuple)
            # Add the original tuple and df to the new list of unique subsets
            subsets_double_unique.append(((str1, str2), df))

    return subsets_double_unique

def viz_subset(i, subsets, movies):
    '''Visualize a subset i'''
    key = subsets[i][0]
    subset = subsets[i][1]
    print('Subset: {}'.format(key))
    print("\t{} | {} (size subset | movies)".format(len(subset),len(movies)))
    print("\t= {} %".format(round(len(subset)/len(movies)*100, 4)))

    # Percentages
    movies_by_year = movies.groupby('year').count()['id_wiki']
    distrib = (subset.groupby('year').count()['id_wiki'] / movies_by_year).fillna(0)

    # Plot release dates distribution
    fig, axs = plt.subplots(1, 3, figsize=(15,4))
    axs = axs.ravel()

    movies.year.hist(bins=movies.year.nunique(), ax=axs[0], histtype='step')
    ax_settings(axs[0], xlabel='Year', ylabel='Nb of movies', title='All movies release year distribution')
    axs[0].set_xlim((1910,2010))

    subset.year.hist(bins=subset.year.nunique(), ax=axs[1], histtype='step')
    ax_settings(axs[1], xlabel='Year', ylabel='Nb of movies', title='Subset : {}'.format(key))
    axs[1].set_xlim((1910,2010))

    distrib.plot(ax=axs[2])
    ax_settings(axs[2], xlabel='Year', ylabel='Fraction of the genre by year [%]', title='Subset : {}'.format(key))
    axs[2].set_xlim((1910,2010))
    axs[2].set_ylim(bottom=0)

    plt.tight_layout()
    plt.show()

def butter_lowpass_filter(data, cutoff, fs, order):
    '''Apply Butterworth lowpass filter on the data'''
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def find_local_maxima(x):
    local_maxima = []
    n = len(x)

    for i in range(n):
        # Skip first and last elements
        if 0 < i < n - 1 and x[i] > x[i - 1] and x[i] > x[i + 1]:
            local_maxima.append(i)

    return local_maxima

def find_inflection_points(x):
    inflection_points = []

    # First, calculate the first differences (delta_x)
    delta_x = [x[i+1] - x[i] for i in range(len(x) - 1)]

    # Then, calculate the differences of these deltas (second derivative approximation)
    delta2_x = [delta_x[i+1] - delta_x[i] for i in range(len(delta_x) - 1)]

    # Now, look for sign changes in delta2_x
    for i in range(1, len(delta2_x)):
        if delta2_x[i] * delta2_x[i-1] < 0:  # Sign change
            inflection_points.append(i)  # Using i here as it represents the original index in x

    return inflection_points

def peak_detection(distrib, frac):
    '''Detect peak of a signal:
    - Find local max
    - Peak should be above overall frac
    - Peak quality should be above threshold = frac*0.2'''

    # Signal analysis
    x = butter_lowpass_filter(distrib, cutoff=5, fs=len(distrib), order=3)
    peaks = find_local_maxima(x)
    inflexions = find_inflection_points(x)

    # Keep only peaks above overall frac
    peaks = [p for p in peaks if x[p] > frac]
    # Quality analysis
    inflexions = [max([i for i in inflexions if i<p], default=0) for p in peaks]
    quality = [(x[p]-x[i])/frac for p,i in zip(peaks, inflexions)]
    
    return peaks, inflexions, quality

def get_peaks(movies, subsets, i):
    '''Get peaks of a subset, and their quality'''
    # Preprocess the subset
    subset = subsets[i][1]
    movies_by_year = movies.groupby('year').count()['id_wiki']
    distrib = (subset.groupby('year').count()['id_wiki'] / movies_by_year * 100).fillna(0)
    frac = len(subset)/len(movies) * 100

    # Find peaks and quality
    peaks, inflexions, quality = peak_detection(distrib, frac)
    return list(distrib.index[peaks]), list(distrib.index[inflexions]), quality

def viz_peaks(movies, subsets, i, search=None, pivotals=None):
    '''Visualize the peaks or trends of a subset i'''
    key = subsets[i][0]
    subset = subsets[i][1]
    movies_by_year = movies.groupby('year').count()['id_wiki']
    distrib = (subset.groupby('year').count()['id_wiki'] / movies_by_year * 100).fillna(0)
    frac = len(subset)/len(movies) * 100

    # Low pass filter
    x = butter_lowpass_filter(distrib, cutoff=5, fs=len(distrib), order=3)

    # Find peaks and quality
    peaks, inflexions, quality = peak_detection(distrib, frac)

    # Plot the data
    fig, axs = plt.subplots(1,1,figsize=(12, 6))
    tab10_palette = plt.cm.get_cmap("tab10").colors
    plt.plot(distrib.index, distrib, '--', label='Original distribution', alpha=0.3, color=tab10_palette[2])
    plt.plot(distrib.index, x, label='Smoothed distribution', color=tab10_palette[1])
    plt.plot(distrib.index[peaks], x[peaks], "o", color='k', label='Peaks')
    plt.plot(distrib.index[inflexions], x[inflexions], "+", color='k', label='Inflexions')
    plt.plot(distrib.index, np.ones_like(distrib)*frac, "--", color=tab10_palette[3], label='Subset historic fraction', alpha=0.3)
    y_offset = 0.05 * (plt.ylim()[1] - plt.ylim()[0])
    plt.text(distrib.index[0], frac-0.7*y_offset, 'historic fraction: '+str(round(frac,3))+' %', ha='left', va='bottom', fontsize=8, color=tab10_palette[3], alpha=1)
    for p, q in zip(peaks, quality):
        year = distrib.index[p]
        value = x[p]
        tr = 0.2
        c = 'red' if q < tr else 'k'
        y_offset = 0.05 * (plt.ylim()[1] - plt.ylim()[0])
        plt.text(year+3, value, str(year), ha='center', va='bottom', fontsize=15, color=c)
        plt.text(year+3, value+y_offset, 'Q:'+str(round(q,3)), ha='center', va='bottom', fontsize=12, color=c)
    for i, q in zip(inflexions, quality):
        year = distrib.index[i]
        value = x[i]
        tr = 0.2
        c = 'red' if q < tr else 'k'
        y_offset = 0.05 * (plt.ylim()[1] - plt.ylim()[0])
        plt.text(year-3, value+0.2*y_offset, str(year), ha='center', va='bottom', fontsize=12, color=c)
    if search != None:
        plt.axvspan(max(1910,2*search[1]-search[0]), min(2010,search[1]+2), color='green', alpha=0.2)
    if pivotals != None:
        pivotal_y = [x[distrib.index==y] for y in pivotals[0]]
        plt.plot(pivotals[0], pivotal_y, "o", color=tab10_palette[3], label='Pivotals', markeredgecolor='black')
        for i, y in enumerate(pivotal_y):
            plt.text(pivotals[0][i], y-1.2*y_offset, str(pivotals[1][i]), ha='center', va='bottom', fontsize=12, fontweight='bold', color=tab10_palette[0])
            plt.text(pivotals[0][i], y-2*y_offset, '('+str(pivotals[0][i])+')', ha='center', va='bottom', fontsize=12, fontweight='bold', color=tab10_palette[0])
    plt.xlabel('Year')
    plt.ylabel('Fraction of the genre by year [%]')
    plt.title('Subset : {} (size {})'.format(key, len(subset)))
    plt.grid(alpha=0.3, axis='y')
    plt.legend()
    plt.show()
    return fig

def get_all_viz(movies, subsets):
    '''Save all figs in a folder'''
    folder_path = os.path.abspath(os.curdir)
    for i in range(len(subsets)):
        fig = viz_peaks(movies, subsets, i)
        file_name = 'img/viz/'+str(i)+'_'+subsets[i][0].replace('/',' ')+'.png'  # or use .jpg, .pdf, etc.
        save_path = os.path.join(folder_path, file_name)
        fig.savefig(save_path, dpi=300)

def find_subset(subsets, key):
    '''Find a peticular subset with its key'''
    result = None
    for i, s in enumerate(subsets):
        if s[0]==key:
            result = i
    return result

def find_subset_double(subsets_double, key1, key2):
    '''Find a peticular subset with its key'''
    result = None
    for i, s in enumerate(subsets_double):
        if ((s[0][0]==key1) & (s[0][1]==key2)) | ((s[0][0]==key2) & (s[0][1]==key1)):
            result = i
    return result

def get_trends(movies, subsets, threshold):
    '''Returns a list of tuples of this format : ('genre_name', [peak_years], [inflexion_years])
    for all combination of Genre and Peak'''
    trends = []
    for i, s in enumerate(subsets):
        peaks = []
        inflexions = []
        quality = []
        for p,inflex,q in zip(*get_peaks(movies, subsets, i)):
            if q>threshold:
                peaks.append(p)
                inflexions.append(inflex)
                quality.append(q)
        trends.append((s[0],peaks,inflexions,quality))
    return trends

def range_search(subsets, key, year_min, year_max):
    '''Return a dataframe of a movies subset within a range, before a date'''
    subset = subsets[find_subset(subsets, key)][1]
    range_results = subset[(subset.year<=year_max) & (subset.year>=year_min)]
    return range_results

def get_candidates(subsets, trends):
    '''Return all candidates movies to be pivotal, for each subset
    in a range of years before the inflexion year: the difference between the peak and the inflexion
    Output format: array of ('genre_name', peak_year, inflexion_year, DF)'''
    candidates = [(trend[0], peak, inflex, range_search(subsets, trend[0], 2*inflex-peak, inflex+2)) for trend in trends
                                                                                  for peak, inflex in zip(trend[1], trend[2])]
    return candidates

def range_search_double(subsets, key, year_min, year_max):
    '''Return a dataframe of a movies subset within a range, before a date'''
    subset = subsets[find_subset_double(subsets, key[0], key[1])][1]
    range_results = subset[(subset.year<=year_max) & (subset.year>=year_min)]
    return range_results

def get_candidates_double(subsets_double, trends):
    '''Return all candidates movies to be pivotal, for each subset
    in a range of years before the inflexion year: the difference between the peak and the inflexion
    Output format: array of ('genre_name', peak_year, inflexion_year, DF)'''
    candidates = [(trend[0], peak, inflex, range_search_double(subsets_double, trend[0], 2*inflex-peak, inflex+2)) for trend in trends
                                                                                  for peak, inflex in zip(trend[1], trend[2])]
    return candidates

def find_candidates(candidates, key, peak=None):
    '''Search candidates for the trend corresponding to parameters
    Input:
        candidates: list of candidates
        key: candidates corresponding to a genre name
        year: candidates of a genre corresponding to a peak year'''
    result = []
    for i, c in enumerate(candidates):
        if c[0]==key:
            if peak==None:
                result.append(i)
            elif peak=='first':
                return i
            elif c[1]==peak:
                return i
    return result

def show_candidates(movies, subsets, candidates, key, peak='first'):
    '''Display candidates for trend i'''
    i = find_candidates(candidates,key,peak=peak)
    fig = viz_peaks(movies, subsets, find_subset(subsets, key), search=candidates[i][1:3])
    print('Candidates of pivotal of genre {}, for trend peak in {} and trend inflexion in {}'
          .format(candidates[i][0],candidates[i][1],candidates[i][2]))
    print('Nb of candidates: {}'.format(len(candidates[i][3])))
    c = candidates[i][3].sort_values('year')
    return c

def show_candidates_double(movies, subsets, candidates, key1, key2, peak='first'):
    '''Display candidates for trend i'''
    i = find_candidates(candidates,(key1,key2),peak=peak)
    fig = viz_peaks(movies, subsets, find_subset_double(subsets, key1, key2), search=candidates[i][1:3])
    print('Candidates of pivotal of genre {}, for trend peak in {} and trend inflexion in {}'
          .format(candidates[i][0],candidates[i][1],candidates[i][2]))
    print('Nb of candidates: {}'.format(len(candidates[i][3])))
    c = candidates[i][3].sort_values('year')
    return c

def get_pivotals_simple(movies, subsets, pivotals, key):
    p = pivotals[pivotals['trend_genre']==key]
    pivotal_year = []
    pivotal_name = []
    for id in p['id_wiki']:
        pivotal_year.append(movies[movies.id_wiki==id]['year'].values[0])
        pivotal_name.append(movies[movies.id_wiki==id]['name'].values[0])
    fig = viz_peaks(movies, subsets, find_subset(subsets, key), pivotals=(pivotal_year, pivotal_name))
    return fig

def get_pivotals_double(movies, subsets, pivotals, key1, key2):
    p = pivotals[pivotals['trend_genre']==(key1, key2)]
    pivotal_year = []
    pivotal_name = []
    for id in p['id_wiki']:
        pivotal_year.append(movies[movies.id_wiki==id]['year'].values[0])
        pivotal_name.append(movies[movies.id_wiki==id]['name'].values[0])
    fig = viz_peaks(movies, subsets, find_subset_double(subsets, key1, key2), pivotals=(pivotal_year, pivotal_name))
    return fig

def show_pivotal(pivotals, movies, i):
    pivotal_genre = pivotals['trend_genre'].iloc[i]
    pivotal_peak = pivotals['trend_id'].iloc[i].split('of years ')[1]
    pivotal_name = pivotals['name'].iloc[i]
    pivotal_year = movies[movies.id_wiki==pivotals['id_wiki'].iloc[i]]['year'].values[0]
    print('==== PIVOTAL MOVIE ====')
    print('For genre {} of the trend peak {}'.format(pivotal_genre, pivotal_peak))
    print('\t🏆🏆 >> PIVOTAL IS {} ({})'.format(pivotal_name, pivotal_year))
    print('')

def get_all_viz_pivotal(movies, subsets, subsets_double, pivotals_simple, pivotals_double):
    '''Save all figs in a folder'''
    folder_path = os.path.abspath(os.curdir)
    for i, key in enumerate(list(pivotals_simple['trend_genre'].unique())):
        fig = get_pivotals_simple(movies, subsets, pivotals_simple, key)
        file_name = 'images/pivotals/'+str(i)+'_Pivotals_'+key.replace('/',' ')+'.png'  # or use .jpg, .pdf, etc.
        save_path = os.path.join(folder_path, file_name)
        fig.savefig(save_path, dpi=300)
    for i, key in enumerate(list(pivotals_double['trend_genre'].unique())):
        fig = get_pivotals_double(movies, subsets_double, pivotals_double, key[0], key[1])
        file_name = 'images/pivotals/'+str(i+len(list(pivotals_simple['trend_genre'].unique())))+'_Pivotals_'+str(key).replace('/',' ')+'.png'  # or use .jpg, .pdf, etc.
        save_path = os.path.join(folder_path, file_name)
        fig.savefig(save_path, dpi=300)

#-------------------------------------------------------------------------------------------------------
# PAUL
# PAUL
def remove_stopwords_and_punctuation(text):
    # Tokenize the text into words
    words = word_tokenize(text)

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    filtered_words = [word.lower() for word in words if word.lower() not in stop_words and word.lower() not in punctuation]

    return filtered_words


def apply_stemming(words):
    # Apply stemming using PorterStemmer
    stemmer = SnowballStemmer("english")
    stemmed_words = [stemmer.stem(word) for word in words]

    return stemmed_words

def tok_and_stem(text, stemmer=None):
    if stemmer is None:
        stemmer = SnowballStemmer("english")
    
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = [token for token in tokens if re.search('[a-zA-Z]', token)]
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def similarity_calculation(data_frame, text_column):
    # Merge DataFrame with movie data
    

    # Initialize SnowballStemmer
    stemmer = SnowballStemmer("english")

    # Tokenize and stem the text data
    

    # Create TF-IDF matrix
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000, min_df=0.1,
                                       stop_words='english', use_idf=True, tokenizer=tok_and_stem, ngram_range=(1,2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(data_frame[text_column])

    # Calculate cosine similarity
    similarity_distance = cosine_similarity(tfidf_matrix)


    return similarity_distance



def similarity_plot(similarity_distance, merged_df, film_names):
    indices = merged_df[merged_df['name'].isin(film_names)].index

    film_names = merged_df.loc[indices, 'name'].tolist() 

    # Subset the matrix based on specified indices or use the first 100 rows and columns by default
    if indices is not None:
        subset_matrix = similarity_distance[np.ix_(indices, indices)]
    else:
        subset_matrix = similarity_distance[:100, :100]

    # Create a heatmap using Seaborn
    sns.set(style="white")  # Optional: Set the background style
    plt.figure(figsize=(10, 8))  # Set the figure size
    
    # Use a logarithmic color map with a white center
    sns.heatmap(subset_matrix, cmap="Blues",norm=LogNorm(), annot=True, fmt=".2f", linewidths=.5)
   
    # Add film names to the x and y axes
    plt.xticks(np.arange(len(indices)) + 0.5, [film_names[i] for i in range(len(indices))], rotation=45, ha='right')
    plt.yticks(np.arange(len(indices)) + 0.5, [film_names[i] for i in range(len(indices))], rotation=0)

     # Show the plot
    plt.title('Subset of Similarity Distance Matrix')
    plt.show()

def calculate_mean_similarity_1(similarity_matrix, chosen_movie_index, movie_indices):
   
    # Extract similarity values for the chosen movie and the set of movies
    similarities = similarity_matrix[chosen_movie_index, movie_indices]

    # Calculate the mean similarity
    mean_similarity = np.mean(similarities)

    return mean_similarity
import numpy as np

def calculate_mean_similarity_2(movie_index, merged_df, similarity_matrix, genre):
    """
    Calculate the mean similarity of the plot of a given film compared to films of the same genre
    released 10 years before and 10 years after.

    Parameters:
    - movie_index (int): Index of the film in the DataFrame.
    - merged_df (pandas.DataFrame): Merged DataFrame containing movie information.
    - similarity_matrix (numpy.ndarray): Matrix containing pairwise similarities between movie plots.
    - genre (str): Genre of the films to compare.

    Returns:
    - tuple: Mean similarity of the given film's plot to films released 10 years before and after.
    """
    # Find the row corresponding to the given film
    film_row = merged_df.iloc[movie_index]

    # Extract relevant information
    release_year = film_row['year']

    # Calculate the release year range for 5 years before and 5 years after
    before_year = release_year - 7
    after_year = release_year + 7

    # Filter movies of the same genre released 5 years before and after
    similar_movies_before = merged_df[
        (merged_df['genres'].apply(lambda genres: genre in genres)) &
        (merged_df['year'].between(before_year, release_year - 1))
    ]

    similar_movies_after = merged_df[
        (merged_df['genres'].apply(lambda genres: genre in genres)) &
        (merged_df['year'].between(release_year + 1, after_year))
    ]

    # Get the indices of the movies
    similar_indices_before = similar_movies_before.index.tolist()
    similar_indices_after = similar_movies_after.index.tolist()

    # Calculate the mean similarity
    mean_similarity_before = np.mean(similarity_matrix[movie_index, similar_indices_before])
    mean_similarity_after = np.mean(similarity_matrix[movie_index, similar_indices_after])

    return mean_similarity_before, mean_similarity_after


def calculate_mean_similarity_4(df_candidates, merged_df, similarity_matrix, genre):
    """
    Calculate the mean similarity of the plot of a given film compared to films of the same genre
    released 10 years before and 10 years after.

    Parameters:
    - df_candidates (pandas.DataFrame): DataFrame containing movie candidates and their information.
    - merged_df (pandas.DataFrame): Merged DataFrame containing movie information.
    - similarity_matrix (numpy.ndarray): Matrix containing pairwise similarities between movie plots.
    - genre (str): Genre of the films to compare.

    Returns:
    - pandas.DataFrame: Updated DataFrame (df_candidates) with a new column for delta_similarity.
    """
    delta_similarity_list = []
    for id_wiki, release_year in zip(df_candidates['id_wiki'], df_candidates['year']):
        # Calculate the release year range for 5 years before and 5 years after
        before_year = release_year - 10
        after_year = release_year + 10

        # Filter movies of the same genre released 5 years before and after
        similar_movies_before = merged_df[
            (merged_df['genres'].apply(lambda genres: genre in genres)) &
            (merged_df['year'].between(before_year, release_year - 1))
        ]

        similar_movies_after = merged_df[
            (merged_df['genres'].apply(lambda genres: genre in genres)) &
            (merged_df['year'].between(release_year + 1, after_year))
        ]

        # Get the indices of the movies
        similar_indices_before = similar_movies_before.index.tolist()
        similar_indices_after = similar_movies_after.index.tolist()

        # Check if the DataFrame is not empty before accessing the index
        if not merged_df[merged_df['id_wiki'] == id_wiki].empty:
            index_sim_mat = merged_df[merged_df['id_wiki'] == id_wiki].index.values[0]

            # Calculate the mean similarity
            mean_similarity_before = np.mean(similarity_matrix[index_sim_mat, similar_indices_before])
            mean_similarity_after = np.mean(similarity_matrix[index_sim_mat, similar_indices_after])

            # Append the mean_similarity_after value to the list
            delta_similarity_list.append((mean_similarity_after - mean_similarity_before)*100)
        else:
            # Handle the case when no match is found for the id_wiki
            delta_similarity_list.append(np.nan)

    # Add a new column 'delta_similarity' to df_candidates
    df_candidates['delta_similarity'] = delta_similarity_list

    return df_candidates

def calculate_mean_similarity_5(df_candidates, merged_df, similarity_matrix, genre):
    """
    Calculate the mean similarity of the plot of a given film compared to films of the same genre
    released 10 years before and 10 years after.

    Parameters:
    - df_candidates (pandas.DataFrame): DataFrame containing movie candidates and their information.
    - merged_df (pandas.DataFrame): Merged DataFrame containing movie information.
    - similarity_matrix (numpy.ndarray): Matrix containing pairwise similarities between movie plots.
    - genre (str): Genre of the films to compare.

    Returns:
    - pandas.DataFrame: Updated DataFrame (df_candidates) with a new column for delta_similarity.
    """
    mean_similarity_before = []
    mean_similarity_after = []
    for id_wiki, release_year in zip(df_candidates['id_wiki'], df_candidates['year']):
        # Calculate the release year range for 5 years before and 5 years after
        before_year = release_year - 7
        after_year = release_year + 7

        # Filter movies of the same genre released 5 years before and after
        similar_movies_before = merged_df[
            merged_df['genres'].apply(lambda genres: genre in genres) &
            (merged_df['year'].between(before_year, release_year - 1))
        ]

        similar_movies_after = merged_df[
            merged_df['genres'].apply(lambda genres: genre in genres) &
            (merged_df['year'].between(release_year + 1, after_year))
        ]

        # Get the indices of the movies
        similar_indices_before = similar_movies_before.index.tolist()
        similar_indices_after = similar_movies_after.index.tolist()

        # Check if the DataFrame is not empty before accessing the index
        if not merged_df[merged_df['id_wiki'] == id_wiki].empty:
            index_sim_mat = merged_df[merged_df['id_wiki'] == id_wiki].index.values[0]

            # Calculate the mean similarity
            mean_similarity_before_1 = np.mean(similarity_matrix[index_sim_mat, similar_indices_before])
            mean_similarity_after_1 = np.mean(similarity_matrix[index_sim_mat, similar_indices_after])

            # Append the mean_similarity_after value to the list
            mean_similarity_before.append((mean_similarity_before_1))
            mean_similarity_after.append((mean_similarity_after_1))

        else:
            # Handle the case when no match is found for the id_wiki
            mean_similarity_before.append((np.nan))
            mean_similarity_after.append((np.nan))

    # Add a new column 'delta_similarity' to df_candidates
    df_candidates['mean_similarity_before'] = mean_similarity_before
    df_candidates['mean_similarity_after'] = mean_similarity_after

    return df_candidates


def calculate_mean_similarity(df_candidates, merged_df, similarity_matrix, genre):
    """
    Calculate the mean similarity of the plot of a given film compared to films of the same genre
    released 10 years before and 10 years after.

    Parameters:
    - df_candidates (pandas.DataFrame): DataFrame containing movie candidates and their information.
    - merged_df (pandas.DataFrame): Merged DataFrame containing movie information.
    - similarity_matrix (numpy.ndarray): Matrix containing pairwise similarities between movie plots.
    - genre (str): Genre of the films to compare.

    Returns:
    - pandas.DataFrame: Updated DataFrame (df_candidates) with a new column for delta_similarity.
    """
    mean_similarity_before = []
    mean_similarity_after = []
    for id_wiki, release_year in zip(df_candidates['id_wiki'], df_candidates['year']):
        # Calculate the release year range for 5 years before and 5 years after
        before_year = release_year - 7
        after_year = release_year + 7

        # Filter movies of the same genre released 5 years before and after
        similar_movies_before = merged_df[
            merged_df['genres'].apply(lambda genres: any(g in genres for g in genre)) &
            (merged_df['year'].between(before_year, release_year - 1))
        ]

        similar_movies_after = merged_df[
            merged_df['genres'].apply(lambda genres: any(g in genres for g in genre)) &
            (merged_df['year'].between(release_year + 1, after_year))
        ]

        # Get the indices of the movies
        similar_indices_before = similar_movies_before.index.tolist()
        similar_indices_after = similar_movies_after.index.tolist()

        # Check if the DataFrame is not empty before accessing the index
        if not merged_df[merged_df['id_wiki'] == id_wiki].empty:
            index_sim_mat = merged_df[merged_df['id_wiki'] == id_wiki].index.values[0]

            # Calculate the mean similarity
            mean_similarity_before_1 = np.mean(similarity_matrix[index_sim_mat, similar_indices_before])
            mean_similarity_after_1 = np.mean(similarity_matrix[index_sim_mat, similar_indices_after])

            # Append the mean_similarity_after value to the list
            mean_similarity_before.append((mean_similarity_before_1))
            mean_similarity_after.append((mean_similarity_after_1))

        else:
            # Handle the case when no match is found for the id_wiki
            mean_similarity_before.append((np.nan))
            mean_similarity_after.append((np.nan))

    # Add a new column 'delta_similarity' to df_candidates
    df_candidates['mean_similarity_before'] = mean_similarity_before
    df_candidates['mean_similarity_after'] = mean_similarity_after

    return df_candidates



def process_candidates(candidates, min_elements, movies_features, merged_df, similarity_matrix):
    result_df = pd.DataFrame()
    columns_to_drop = ['id_freebase', 'name', 'year', 'revenue', 'runtime', 'lang', 'countries', 'genres']
    columns_to_drop_2 = ['id_freebase', 'runtime', 'lang', 'countries', 'has_won', 'nominated', 'revenue_part']

    for trend_id in range(len(candidates)):
        if len(candidates[trend_id][3]) >= min_elements:
            trend_year = int(candidates[trend_id][2])
            genre = candidates[trend_id][0]

            if isinstance(genre, tuple):
                # If genre is a tuple, convert it to a string or handle it appropriately
                genre_str = ', '.join(genre)  # Convert the tuple to a string
            else:
                genre_str = genre

            df = candidates[trend_id][3].copy()
            df = df.drop(columns=columns_to_drop).copy()
            df = df.merge(movies_features, on='id_wiki')
            
            df['trend_year']=trend_year
            df['trend_number'] = 1000+int(trend_id)
            df['trend_genre'] = genre_str  # Use the converted genre string
            df['trend_id']= genre_str + ' of years ' + str(trend_year)
            
            df['year_from_trend'] = trend_year - df['year']
            df = calculate_mean_similarity_5(df, merged_df, similarity_matrix, genre)
            df.drop(columns=columns_to_drop_2, inplace=True)
            result_df = pd.concat([result_df, df], ignore_index=True)

    return result_df

def process_candidates2(candidates, min_elements, movies_features, merged_df, similarity_matrix):
    result_dfs = []  # Accumulate DataFrames in a list

    columns_to_drop = ['id_freebase', 'name', 'year', 'revenue', 'runtime', 'lang', 'countries', 'genres']
    columns_to_drop_2 = ['id_freebase', 'runtime', 'lang', 'countries', 'has_won', 'nominated', 'revenue_part']

    for trend_id, candidate in enumerate(candidates):
        if len(candidate[3]) >= min_elements:
            trend_year = int(candidate[2])
            genre = candidate[0]

            # Use apply to convert genre to string
            genre_str = ', '.join(genre) if isinstance(genre, tuple) else genre

            df = candidate[3].copy()
            df.drop(columns=columns_to_drop, inplace=True)
            
            # Merge movies_features after dropping unnecessary columns
            df = df.merge(movies_features, on='id_wiki')
            
            df['trend_year']=trend_year
            df['trend_number'] = trend_id
            df['trend_genre'] = genre_str
            df['year_from_trend'] = trend_year - df['year']
            df['trend_id']= genre_str + ' of years ' + str(trend_year)
            df = calculate_mean_similarity(df, merged_df, similarity_matrix, genre)
            df.drop(columns=columns_to_drop_2, inplace=True)

            result_dfs.append(df)

    result_df = pd.concat(result_dfs, ignore_index=True)  # Concatenate once after the loop
    return result_df

    

def filter_candidates(df, min_movies_per_trend=3):
    # Copy the input DataFrame to avoid modifying the original data
    result_df_copy = df.copy()
    
    # Drop rows with no delta_similarity                                                           
    result_df_copy.dropna(subset=['mean_similarity_before'], inplace=True)
    result_df_copy.dropna(subset=['mean_similarity_after'], inplace=True)
    # Count movies per trend
    movie_counts_per_trend = result_df_copy['trend_id'].value_counts()
    
    # Find trends with fewer than min_movies_per_trend movies
    trends_with_few_movies = movie_counts_per_trend[movie_counts_per_trend < min_movies_per_trend].index
    
    # Remove rows where 'trend_number' is in trends_with_few_movies
    result_df_filtered = result_df_copy[~result_df_copy['trend_id'].isin(trends_with_few_movies)]
    
    unique_names_count = len(result_df_filtered['name'].unique())  # Number of different movies
    print('There are {} different movies'.format(unique_names_count))
    
    return result_df_filtered


def standardize_features(df, features_to_standardize):
    # Copy the input DataFrame to avoid modifying the original data
    result_df_copy = df.copy()
    scaler = StandardScaler()
    # Group by 'trend_number' and standardize the features within each group
    def standardize_group(group):
    
        group[features_to_standardize] = scaler.fit_transform(group[features_to_standardize])
        return group



    result_df_standardized = result_df_copy.groupby('trend_id').apply(standardize_group)
    result_df_ungrouped = result_df_standardized.drop('trend_id', axis=1).reset_index().copy()



    unique_names_count = len(df['name'].unique())
    print('There are {} different movies'.format(unique_names_count))

    return result_df_ungrouped
def compute_plot_similarity(movie1_id, movie2_id,similarity_matrix):
        return similarity_matrix[movie1_id, movie2_id]
    
def training (result_df_standardized) :
    # PIVOTAL MOVIES
    STAR_WARS_IV = result_df_standardized[(result_df_standardized['id_wiki'] == 52549) & (result_df_standardized['trend_genre'] == 'Science Fiction')].head(1)
    THE_KARATE_KID = result_df_standardized[(result_df_standardized['id_wiki'] == 657809) & (result_df_standardized['trend_genre'] == 'Martial Arts Film')].head(1)
    TOY_STORY = result_df_standardized[(result_df_standardized['id_wiki'] == 53085) & (result_df_standardized['trend_genre'] == 'Computer Animation')].head(1)
    THE_SEARCHERS = result_df_standardized[(result_df_standardized['id_wiki'] == 76335) & (result_df_standardized['trend_genre'] == 'Epic Western')].head(1)
    BONNIE_AND_CLYDE= result_df_standardized[(result_df_standardized['id_wiki'] == 68245) & (result_df_standardized['trend_genre'] == 'Gangster Film')].head(1)
    PHILADELPHIA = result_df_standardized[(result_df_standardized['id_wiki'] == 468293) & (result_df_standardized['trend_genre'] == 'Gay')].head(1)
    PULP_FICTION = result_df_standardized[(result_df_standardized['id_wiki'] == 54173) & (result_df_standardized['trend_genre'] == 'Crime Comedy')].head(1)
    THE_LION_KING = result_df_standardized[(result_df_standardized['id_wiki'] == 88678) & (result_df_standardized['trend_genre'] == 'Animation')].head(1)
    THE_EXORCIST = result_df_standardized[(result_df_standardized['id_wiki'] == 725459) & (result_df_standardized['trend_genre'] == 'Horror')].head(1)
    THE_SHINING = result_df_standardized[(result_df_standardized['id_wiki'] == 1186616) & (result_df_standardized['trend_genre'] == 'Psychological thriller')].head(1)
    TITANIC= result_df_standardized[(result_df_standardized['id_wiki'] == 52371) & (result_df_standardized['trend_genre'] == 'Tragedy')].head(1)
    ALIEN = result_df_standardized[(result_df_standardized['id_wiki'] == 23487440) & (result_df_standardized['trend_genre'] == 'Creature Film')].head(1)
    PIVOTAL_LIST = [
        STAR_WARS_IV,
        THE_KARATE_KID,
        TOY_STORY,
        THE_SEARCHERS,
        BONNIE_AND_CLYDE,
        PHILADELPHIA,
        PULP_FICTION,
        THE_LION_KING,
        THE_EXORCIST,
        THE_SHINING,
        TITANIC,
        ALIEN
    ]

    PIVOTAL_DF = pd.concat(PIVOTAL_LIST, ignore_index=True)

    # NON-PIVOTAL MOVIES
    NON_PIVOTAL_0 =  result_df_standardized[(result_df_standardized['id_wiki'] != 52549) & (result_df_standardized['trend_genre'] == 'Science Fiction')]
    NON_PIVOTAL_1 = result_df_standardized[(result_df_standardized['id_wiki'] != 91133) & (result_df_standardized['trend_genre'] == 'Martial Arts Film')].head(1)
    NON_PIVOTAL_2 = result_df_standardized[(result_df_standardized['id_wiki'] != 53085) & (result_df_standardized['trend_genre'] == 'Computer Animation')].head(1)
    NON_PIVOTAL_3 = result_df_standardized[(result_df_standardized['id_wiki'] != 76335) & (result_df_standardized['trend_genre'] == 'Epic Western')].head(1)
    NON_PIVOTAL_4 = result_df_standardized[(result_df_standardized['id_wiki'] != 68245) & (result_df_standardized['trend_genre'] == 'Gangster Film')].head(1)
    NON_PIVOTAL_5 = result_df_standardized[(result_df_standardized['id_wiki'] != 468293) & (result_df_standardized['trend_genre'] == 'Gay')].head(1)
    NON_PIVOTAL_6 = result_df_standardized[(result_df_standardized['id_wiki'] != 54173) & (result_df_standardized['trend_genre'] == 'Crime Comedy')].head(1)
    NON_PIVOTAL_7 = result_df_standardized[(result_df_standardized['id_wiki'] != 88678) & (result_df_standardized['trend_genre'] == 'Animation')].head(1)
    NON_PIVOTAL_8 = result_df_standardized[(result_df_standardized['id_wiki'] != 725459) & (result_df_standardized['trend_genre'] == 'Horror')].head(1)
    NON_PIVOTAL_9 = result_df_standardized[(result_df_standardized['id_wiki'] != 1186616) & (result_df_standardized['trend_genre'] == 'Psychological thriller')].head(1)
    NON_PIVOTAL_10 = result_df_standardized[(result_df_standardized['id_wiki'] != 52371) & (result_df_standardized['trend_genre'] == 'Tragedy')].head(1)
    NON_PIVOTAL_11 = result_df_standardized[(result_df_standardized['id_wiki'] != 23487440) & (result_df_standardized['trend_genre'] == 'Creature Film')].head(1)

    NON_PIVOTAL_LIST = [
        NON_PIVOTAL_0,
        NON_PIVOTAL_1,
        NON_PIVOTAL_2,
        NON_PIVOTAL_3,
        NON_PIVOTAL_4,
        NON_PIVOTAL_5,
        NON_PIVOTAL_6,
        NON_PIVOTAL_7,
        NON_PIVOTAL_8,
        NON_PIVOTAL_9,
        NON_PIVOTAL_10,
        NON_PIVOTAL_11
    ]

    # Concatenate the DataFrames
    NON_PIVOTAL_DF = pd.concat(NON_PIVOTAL_LIST, ignore_index=True)

    # Concatenate all DataFrames
    all_movies=pd.concat([PIVOTAL_DF,NON_PIVOTAL_DF])

    # Extract relevant features for

    # Extract relevant features for training
    features_train = all_movies[['rating', 'nbr_won','votes', 'nbr_nomination', 'revenue_norm', 'mean_similarity_before','mean_similarity_after','year_from_trend']]

    # Create the target variable y_train (1 for pivotal movies, 0 for non-pivotal movies)
    y_train = [1] * len(PIVOTAL_DF) + [0] * len(NON_PIVOTAL_DF)
    
    return features_train,y_train

def perform_logistic_regression(training_points, labels, feature_names):
    # Create a logistic regression model
    logistic = LogisticRegression(solver='lbfgs')
    
    # Fit the model to the training data
    logistic.fit(training_points, labels)
    
    # Print coefficients for each feature
    coefficients = logistic.coef_[0]
    print("Coefficents: \n")
    for feature_name, coefficient in zip(feature_names, coefficients):
        rounded_coefficient = round(coefficient, 5)
        print(f"{feature_name}: {rounded_coefficient}")
    
    # Print intercept
    rounded_intercept = round(logistic.intercept_[0], 5)
    print("\n Intercept:", rounded_intercept)
    return logistic

def find_most_likely_pivotal_movie(result_df_standardized, logistic):
    most_likely_indices = []
    all_probabilities = []  # List to store probabilities for each movie
    
    # Group by trend_id
    grouped_by_trend = result_df_standardized.groupby('trend_id')

    # Dictionary to store most likely pivotal movie for each group
    most_likely_pivotal = {'name': [], 'id_wiki': [], 'trend_genre': [], 'trend_id': []}

    # Iterate over groups
    for trend_id, group_df in grouped_by_trend:
        # Select features and labels for the group
        features_group = group_df[['rating', 'nbr_won', 'votes', 'nbr_nomination', 'revenue_norm', 'mean_similarity_before', 'mean_similarity_after', 'year_from_trend']]

        # Predict probabilities for being pivotal
        probabilities = logistic.predict_proba(features_group)[:, 1]
        all_probabilities.extend(probabilities)  # Append probabilities to the list

        # Identify the index with the highest probability
        most_likely_index = probabilities.argmax()
        most_likely_indices.append(most_likely_index)
        
        # Get the details of the most likely pivotal movie
        most_likely_movie = group_df.iloc[most_likely_index]

        # Store the result in the dictionary
        most_likely_pivotal['name'].append(most_likely_movie['name'])
        most_likely_pivotal['id_wiki'].append(most_likely_movie['id_wiki'])
        most_likely_pivotal['trend_id'].append(trend_id)
        most_likely_pivotal['trend_genre'].append(most_likely_movie['trend_genre'])

    # Create the DataFrame
    pivotal_mov = pd.DataFrame(most_likely_pivotal)
    

    # Display the most likely pivotal movie for each group
    for trend_id in pivotal_mov['trend_id'].unique():
        movie_info = pivotal_mov[pivotal_mov['trend_id'] == trend_id].iloc[0]
        print(f"Trend {trend_id}  : Most Likely Pivotal Movie - {movie_info['name']}") 
        
    return pivotal_mov, most_likely_indices,all_probabilities

def viz_network(movie_name,trend_genre,merged_df,movies_features,pivotal_mov,similarity_matrix,df_plot):
    pivotal=movie_name
    target_id_wiki = pivotal_mov[pivotal_mov['name'] == movie_name]['id_wiki'].values[0]
    release_date = merged_df.loc[merged_df['id_wiki'] == target_id_wiki, 'year'].values[0]
    succes_movies = movies_features[movies_features['genres'].apply(lambda genres: all(g in genres for g in trend_genre))]
    succes_movies = succes_movies.merge(df_plot, on='id_wiki')
    post_succes_movies= succes_movies[(succes_movies['year']>=release_date -2) & (succes_movies['year']< release_date +7)]
    best_movies=post_succes_movies.sort_values(by='revenue_norm', ascending=False)
    first_8_id_wiki = best_movies['id_wiki'].head(7)
    id_wiki_list = first_8_id_wiki.tolist()
    if target_id_wiki not in id_wiki_list:
        id_wiki_list.append(target_id_wiki)
    
    G = nx.Graph()
    tab10_palette = plt.cm.get_cmap("tab10").colors
    # Dictionary to store cumulative similarity scores for each node
    cumulative_similarity_scores = {}

    # Iterate over pairs of id_wiki values
    for i in range(len(id_wiki_list)):
        for j in range(i + 1, len(id_wiki_list)):
            movie1_id = merged_df[merged_df['id_wiki'] == id_wiki_list[i]].index[0]
            movie2_id = merged_df[merged_df['id_wiki'] == id_wiki_list[j]].index[0]

            movie1_name = merged_df.loc[movie1_id, 'name']
            movie2_name = merged_df.loc[movie2_id, 'name']

            # Assuming compute_plot_similarity returns a single similarity score
            similarity_score = compute_plot_similarity(movie1_id, movie2_id,similarity_matrix)

            # Update cumulative similarity scores for each node
            cumulative_similarity_scores[movie1_id] = cumulative_similarity_scores.get(movie1_id, []) + [similarity_score]
            cumulative_similarity_scores[movie2_id] = cumulative_similarity_scores.get(movie2_id, []) + [similarity_score]

            # Add nodes
            G.add_node(movie1_id, label=movie1_name, size=sum(cumulative_similarity_scores[movie1_id]))
            G.add_node(movie2_id, label=movie2_name, size=sum(cumulative_similarity_scores[movie2_id]))

            # Add edge with weight
            G.add_edge(movie1_id, movie2_id, weight=similarity_score)

    # Compute median similarity scores for each node
    median_similarity_scores = {node: np.median(scores) for node, scores in cumulative_similarity_scores.items()}

    # Find the central node with the max median similarity score
    central_node = max(median_similarity_scores, key=median_similarity_scores.get)

    # Set a custom color for the pivotal node

    # Determine the number of nodes excluding the central node
    num_other_nodes = len(G.nodes) - 1

    # Set the radius of the circle
    radius = 2.0  # You can adjust this value based on your preference

    # Calculate the angle between each node on the circle
    angle_increment = 2 * math.pi / num_other_nodes

    # Calculate positions for each node on the circle
    pos = {central_node: [0, 0]}  # Set the position of the central node
    j = 0
    for i, node in enumerate(G.nodes):
        if node != central_node:
            j += 1
            angle = j * angle_increment
            x = radius * math.cos(angle) + pos[central_node][0]
            y = radius * math.sin(angle) + pos[central_node][1]
            pos[node] = [x, y]

    # Extract edge weights
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

    # Normalize edge weights to use for scaling edge thickness
    max_weight = max(edge_weights)
    min_weight = min(edge_weights)
    scaled_weights = [5 * (weight - min_weight) / (max_weight - min_weight) + 1 for weight in edge_weights]

    # Extract node sizes, scaling them with a square root function
    # Extract node sizes, scaling them with a square root function
    sqrt_scaling_factor = 6.5  # Adjust this factor for square root scaling
    factor= 10
    node_sizes = [(median_similarity_scores[n]*factor) ** sqrt_scaling_factor for n in G.nodes]
    

    # Extract node colors
    node_colors = [tab10_palette[4] if G.nodes[n]['label'] == pivotal else tab10_palette[1] for n in G.nodes]
    node_labels = {n: G.nodes[n]['label'] for n in G.nodes}
    # Set up a colormap based on similarity scores
    cmap = plt.cm.get_cmap('Blues')  # Using 'Blues' colormap for shades of blue  # Using 'viridis' for lower values
    norm = Normalize(vmin=min_weight, vmax=max_weight)
    
    # Set a larger plot size
    plt.figure(figsize=(14, 9))

    # Draw the network with colored edges and edge thickness proportional to similarity score
    # Assuming node_labels is a dictionary with node labels
    nx.draw(
        G,
        pos,
        with_labels=True,
        labels=node_labels,
        node_size=node_sizes,
        node_color=node_colors,
        width=scaled_weights,
        edge_color=edge_weights,
        edge_cmap=cmap,
        edge_vmin=min_weight,
        edge_vmax=max_weight,
        font_size=8,
        font_color='black'
    )

    # Add a colorbar to show the mapping between similarity scores and colors
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap))
    cbar.set_label('Similarity Score')
    legend_labels = {'Pivotal Movie': tab10_palette[4], 'Non-Pivotal Movie': tab10_palette[1]}
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10) for label, color in
        legend_labels.items()]
    plt.legend(handles=legend_handles, loc='upper right')
   
    # Save the figure with tight layout
    # Create the 'Image' folder if it doesn't exist
    folder_path = 'Image'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save the figure with tight layout in the 'Image' folder
    file_path = os.path.join(folder_path, 'similarity_network_' + movie_name + '.png')
    plt.savefig(file_path, bbox_inches='tight')
    plt.savefig('similarity_network_' + movie_name +'.png', bbox_inches='tight')
    
    plt.show()
#-------------------------------------------------------------------------------------------------------
# MEHDI
    
def clean_imdb_data(imdb):
    """
    Cleans and preprocesses the IMDb dataset.

    This function performs several data cleaning steps on an IMDb DataFrame:
    - Filters out rows where the 'year' field contains a range or is invalid.
    - Converts the 'year' column to numeric and retains movies released until 2017.
    - Cleans and converts 'gross_income' and 'votes' columns to numeric values, removing undesired characters.
    - Extracts and converts the 'duration' field to numeric, dropping rows with invalid or zero duration.
    - Drops rows with missing critical data such as 'year', 'duration', 'votes', and 'gross_income'.

    Parameters:
    imdb (DataFrame): The raw IMDb DataFrame to be cleaned.

    Returns:
    DataFrame: The cleaned IMDb DataFrame with the applied preprocessing steps.
    """
    # Cleaning the year column
    imdb = imdb[~imdb['year'].astype(str).str.contains('-|–')]
    imdb.loc[:, 'year'] = imdb['year'].str.extract(r'(\d+)', expand=False)
    imdb.loc[:, 'year'] = imdb['year'].replace('', np.nan)
    imdb.loc[:, 'year'] = pd.to_numeric(imdb['year'], errors='coerce')
    imdb.loc[:, 'year'] = imdb['year'].astype('Int64', errors='ignore')

    # Reducing data to movies pre-2018
    imdb = imdb[imdb['year'] <= 2017].reset_index(drop=True)

    # Cleaning gross income
    def clean_gross_income(value):
        if isinstance(value, str):
            value = value.replace(',', '').replace('$', '')
            if 'M' in value:
                value = float(value.replace('M', '')) * (10**6)
        return float(value)

    imdb.loc[:, 'gross_income'] = imdb['gross_income'].apply(lambda x: clean_gross_income(x))
    imdb['gross_income'] = pd.to_numeric(imdb['gross_income'], errors='coerce')
    imdb.loc[:, 'gross_income'] = imdb['gross_income'].replace(0, np.nan)

    # Cleaning votes
    imdb.loc[:, 'votes'] = imdb['votes'].str.replace(',', '').astype(float)
    imdb.loc[:, 'votes'] = imdb['votes'].replace(0, np.nan)
    imdb.loc[pd.isna(imdb['votes']), 'rating'] = np.nan
    imdb = imdb[imdb['rating'] <= 10]

    # Cleaning duration
    imdb.loc[:, 'duration'] = imdb['duration'].str.extract(r'(\d+)', expand=False)
    imdb.loc[:,'duration'] = imdb['duration'].astype(float)
    imdb.loc[:, 'duration'] = imdb['duration'].replace(0, np.nan)
    imdb['duration'] = pd.to_numeric(imdb['duration'], errors='coerce')  

    # Dropping rows with missing year and duration
    imdb = imdb.dropna(subset=['year', 'duration'])

    # Dropping rows where both votes and gross_income are unknown
    imdb = imdb.dropna(subset=['votes', 'gross_income'], how='all').reset_index(drop=True)

    return imdb    

def check_doublons(df, col_check, year, runtime):
    for c in col_check:
        duplicates = df[df.duplicated([c, year, runtime], keep=False)]  
        if not duplicates.empty:
            print(f'Rows with real duplicates: ')
            print(duplicates[[c, year, runtime]])
        else:
            print(f'No duplicates')
    return '-'*80


def fuse_duplicates_imdb(imdb_df):
    # Group by 'name', 'year', and 'duration'
    grouped = imdb_df.groupby(['name', 'year', 'duration'])

    # Define a custom aggregation function to handle NaN values
    def custom_aggregate(series):
        non_nan_values = series.dropna()
        if non_nan_values.empty:
            return None
        return non_nan_values.mean()

    # Apply the custom aggregation function to 'gross_income'
    aggregated_gross_income = grouped['gross_income'].agg(custom_aggregate)

    # Merge the aggregated values back to the original DataFrame
    merged_df = imdb_df.merge(aggregated_gross_income.reset_index(), on=['name', 'year', 'duration'], how='left', suffixes=('', '_mean'))

    # Fill NaN values in 'gross_income' with the mean values
    merged_df['gross_income'] = merged_df['gross_income'].combine_first(merged_df['gross_income_mean'])

    # Drop unnecessary columns
    merged_df = merged_df.drop(columns=['gross_income_mean'])

    return merged_df

def calculate_weighted_average(df, col_check, col_rating, col_weight):
    # Define a custom aggregation function for weighted average of 'rating'
    def custom_weighted_average(df):
        weights = df[col_weight]
        values = df[col_rating]
        weighted_average = (weights * values).sum() / weights.sum() if weights.sum() != 0 else None
        return pd.Series({'weighted_avg_rating': weighted_average, 'sum_votes': df['votes'].sum()})

    # Apply the custom aggregation function to 'rating' and 'votes'
    weighted_avg_ratings = df.groupby([col_check, 'year', 'duration']).apply(custom_weighted_average)

    # Merge the aggregated values back to the original DataFrame
    df = df.merge(weighted_avg_ratings.reset_index(), on=[col_check, 'year', 'duration'], how='left')

    # Fill NaN values in 'rating' with the weighted average values
    df[col_rating] = df['weighted_avg_rating'].combine_first(df[col_rating])

    # Fill NaN values in 'votes' with the sum of votes
    df['votes'] = df['sum_votes']

    # Drop unnecessary columns
    df = df.drop(columns=['weighted_avg_rating', 'sum_votes'])

    # Round the 'rating' column to one decimal place
    df[col_rating] = df[col_rating].round(1)

    # Drop one duplicate per pair
    df = df.drop_duplicates(subset=[col_check, 'year', 'duration'])

    return df

def clean_name_map(df):
    """
    Extracts and cleans 'duration' and 'year' columns in a DataFrame.

    This function performs the following operations:
    - Extracts numeric values from the 'duration' column, treating them as floats.
    - Replaces any non-numeric or empty values in 'duration' with np.nan.
    - Extracts the year from the 'year' column, treating them as floats.
    - Replaces any non-numeric or empty values in 'year' with np.nan, and converts the column to 'Int64' dtype.
    - Drops rows with missing data in either 'duration' or 'year' columns.

    Parameters:
    df (DataFrame): The DataFrame with 'duration' and 'year' columns to be processed.

    Returns:
    DataFrame: The processed DataFrame with cleaned 'duration' and 'year' columns.
    """
    # Extract and clean duration
    df['duration'] = df['duration'].str.extract('(\d+\.\d+|\d+)').astype(float)
    df['duration'] = df['duration'].replace('', np.nan)

    # Extract and clean year
    df['year'] = df['year'].str.extract('(\d+)').astype(float)
    df['year'] = df['year'].replace('', np.nan)
    df.loc[:, 'year'] = df['year'].astype('Int64', errors='ignore')

    # Drop rows with missing data in duration and year
    cleaned_df = df.dropna(subset=['duration', 'year']).reset_index(drop=True)

    return cleaned_df

def clean_column_str(df, column_name):
    """
    Cleans a specified column in a DataFrame.

    This function performs the following cleaning operations on the specified column:
    - Removes dashes ('-') and colons (':').
    - Replaces multiple whitespace characters with a single space.
    - Trims leading and trailing whitespace.

    Parameters:
    df (DataFrame): The DataFrame containing the column to be cleaned.
    column_name (str): The name of the column to clean.

    Returns:
    DataFrame: The DataFrame with the cleaned column.
    """
    if column_name in df.columns:
        df[column_name] = df[column_name].apply(lambda x: str(x).replace('-', '').replace(':', ''))
        df[column_name] = df[column_name].str.replace('\s+', ' ', regex=True).str.strip()
    else:
        print(f"Column '{column_name}' not found in the DataFrame.")
    return df

def drop_duplicates(df, col_check):
    """
    Drop duplicates in a DataFrame based on a list of columns.

    Parameters:
    - df: DataFrame
    - columns: List of columns to consider for duplicate checking

    Returns:
    - DataFrame with duplicates dropped
    """
    # Check for duplicates based on the specified columns
    duplicates_mask = df.duplicated(subset=col_check, keep='first')

    # Drop one element of each duplicate pair
    df_cleaned = df[~duplicates_mask]

    return df_cleaned

def plot_yearly_distribution_imdb(df):
    """
    Plots the yearly distribution of ratings and revenues from a given DataFrame.

    This function creates two subplots:
    1. A bar plot showing the number of ratings per year.
    2. A bar plot showing the total revenue per year.

    Parameters:
    df (DataFrame): The DataFrame containing 'year', 'rating', and 'revenue' columns.

    The function doesn't return anything but displays the plots.
    """
    # Grouping data
    revenue_per_year = df.groupby('year')['revenue'].count().reset_index()
    reviews_count_per_year = df.groupby('year')['rating'].count().reset_index()

    # Creating figure and subplots
    fig = plt.figure(figsize=(14, 7))

    # Plot for number of ratings per year
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, first subplot
    plt.bar(reviews_count_per_year['year'], reviews_count_per_year['rating'])
    plt.xlabel('Year')
    plt.ylabel('Number of Ratings')
    plt.title('Number of Ratings per Year')

    # Plot for revenue per year
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, second subplot
    plt.bar(revenue_per_year['year'], revenue_per_year['revenue'])
    plt.xlabel('Year')
    plt.ylabel('Total Revenue (in millions)')
    plt.title('Total Revenue per Year')

    plt.tight_layout()
    plt.show()

def fuse_and_clean_annex(df):
    """
    Fuses and cleans various columns in the provided DataFrame.

    This function performs the following operations:
    - Fuses revenue data from two columns into one.
    - Fuses ratings and votes data from multiple columns into single 'rating' and 'votes' columns.
    - Addresses NaN values in the 'alt_name' column, replacing them with values from the 'name' column.
    - Calls external functions 'fuse_scores_v2' and 'fuse_duplicates_v2' for fusing scores and removing duplicates.

    Parameters:
    df (DataFrame): The DataFrame to be processed.

    Returns:
    DataFrame: The processed DataFrame with fused and cleaned data.
    """
    # Fuse the revenue columns
    df['revenue'] = df.apply(lambda row: fuse_columns_v2(row['revenue_x'], row['revenue_y']), axis=1)
    df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
    df = df.drop(['revenue_x', 'revenue_y'], axis=1)

    # Fuse the rating and votes columns
    df = fuse_scores_v2(df, score_col1='rating_x', score_col2='rating_y', votes_col1='votes_x', votes_col2='votes_y', score_col='rating', votes_col='votes')

    # Addressing alt_name nan issue
    df['alt_name'] = df['alt_name'].replace('nan', None)
    df['alt_name'].fillna(df['name'], inplace=True)
    
    # Fuse duplicates
    df = fuse_duplicates_v2(df=df, col_check=['name', 'alt_name'], year='year', runtime='runtime', col_null=['revenue'], col_score='rating', col_weight='votes')

    return df

def compare_revenue_columns(df, col1, col2):
    """
    Compares two revenue columns in a DataFrame and prints the percentage of times 
    one is higher, lower, or equal to the other.

    Parameters:
    df (DataFrame): The DataFrame containing the revenue columns.
    col1 (str): The name of the first revenue column.
    col2 (str): The name of the second revenue column.

    The function prints the comparison results and doesn't return anything.
    """
    # Ensure the columns exist
    if col1 not in df.columns or col2 not in df.columns:
        print(f"One or both columns '{col1}' and '{col2}' do not exist in the DataFrame.")
        return

    # Drop rows where either column is NaN
    valid_rows = df.dropna(subset=[col1, col2])

    # Calculate percentages
    percentage_higher = (valid_rows[col1] > valid_rows[col2]).mean() * 100
    percentage_lower = (valid_rows[col1] < valid_rows[col2]).mean() * 100
    percentage_equal = (valid_rows[col1] == valid_rows[col2]).mean() * 100

    # Print results
    print(f"The percentage of times '{col1}' is higher than '{col2}' when both are not NaN is: {percentage_higher:.2f}%") 
    print(f"The percentage of times '{col1}' is lower than '{col2}' when both are not NaN is: {percentage_lower:.2f}%") 
    print(f"The percentage of times '{col1}' is equal to '{col2}' when both are not NaN is: {percentage_equal:.2f}%")

def scale_and_fuse_revenue(df, col_to_scale, col_reference, new_col_name, drop_cols, rename_cols):
    """
    Scales underestimated revenues in one column based on another column and fuses them into a single column.

    Parameters:
    df (DataFrame): The DataFrame containing the revenue columns.
    col_to_scale (str): The name of the revenue column to be scaled.
    col_reference (str): The name of the reference revenue column for scaling.
    new_col_name (str): The name for the new fused revenue column.
    drop_cols (list): A list of column names to be dropped from the DataFrame.
    rename_cols (dict): A dictionary for renaming columns {old_name: new_name}.

    Returns:
    DataFrame: The DataFrame with scaled and fused revenue data and updated columns.
    """
    # Calculate scaling factor
    scaling_factor = (df[df[col_to_scale] < df[col_reference]][col_reference] / 
                      df[df[col_to_scale] < df[col_reference]][col_to_scale]).median()
    print('The scaling factor is:', scaling_factor)

    # Apply scaling factor
    df.loc[df[col_to_scale] < df[col_reference], col_to_scale] = (
        df[df[col_to_scale] < df[col_reference]][col_to_scale] * scaling_factor
    )

    # Convert to numeric
    df[col_to_scale] = pd.to_numeric(df[col_to_scale], errors='coerce')

    # Fuse revenues
    df[new_col_name] = df.apply(lambda row: fuse_columns_v2(row[col_to_scale], row[col_reference]), axis=1)

    # Drop and rename columns
    df = df.drop(columns=drop_cols)
    df = df.rename(columns=rename_cols)

    return df

def process_movies3_stats(df):
    """
    Processes the movies3_stats DataFrame by fusing scores, fusing revenues, dropping and renaming columns,
    and filtering out rows with NaN values in 'revenue' or 'votes'.

    Parameters:
    df (DataFrame): The DataFrame to be processed.

    Returns:
    DataFrame: The processed DataFrame.
    """
    # Fuse scores
    df = fuse_scores_stats(df=df, score_col1='rating_x', score_col2='rating_y', votes_col1='votes_x', votes_col2='votes_y')

    # Fuse revenue columns
    df['revenue'] = df.apply(lambda row: fuse_columns(x=row['revenue_x'], y=row['revenue_y'], column_name='') 
                             if pd.isna(row['revenue_y']) else row['revenue_y'], axis=1)

    # Drop and rename columns
    df = df.drop(columns=['revenue_x', 'revenue_y', 'rating_x', 'votes_x', 'countries_x', 'runtime_x'])
    df = df.rename(columns={'runtime_y': 'runtime', 'rating_y': 'rating', 'votes_y': 'votes', 'countries_y': 'countries'})
    
    # Filter out rows with NaN in 'revenue' or 'votes'
    mask = (df['revenue'].isna()) | (df['votes'].isna())
    df = df[~mask]

    return df

def clean_and_condense_awards(df):
    """
    Cleans and condenses an awards DataFrame.

    The function performs the following operations:
    - Removes rows where the 'name' column is NaN.
    - Cleans the 'name' column by removing certain characters and normalizing whitespace.
    - Condenses the DataFrame to aggregate information for each unique movie.

    Parameters:
    df (DataFrame): The awards DataFrame to be processed.

    Returns:
    DataFrame: The cleaned and condensed awards DataFrame.
    """
    # Remove missing name rows
    df_clean = df.loc[~df['name'].isna()].reset_index(drop=True)

    # Clean the names
    df_clean = clean_column_str(df_clean, 'name')

    # Condense the DataFrame
    aggregation_functions = {'winner': list}
    df_condensed = df_clean.groupby(['name', 'year']).agg(aggregation_functions).reset_index()

    return df_condensed

def fuse_columns_v2(x, y):
    if pd.notna(x) and pd.notna(y):
        # Both entries are present
        if x == y:
            # Entries are the same
            return x
        else:
            # Take the mean of the entries
            return (x + y) / 2
    elif pd.notna(x):
        # x is present, y is missing
        return x
    elif pd.notna(y):
        # y is present, x is missing
        return y
    else:
        # Both entries are missing
        return pd.NA
    

def fuse_scores_v2(df, score_col1, score_col2, votes_col1, votes_col2, score_col, votes_col):
    # Create a new column for fused scores
    numerator = (df[score_col1].fillna(0) * df[votes_col1].fillna(0) +
                 df[score_col2].fillna(0) * df[votes_col2].fillna(0))
    
    denominator = df[votes_col1].fillna(0) + df[votes_col2].fillna(0)

    # Avoid division by zero
    df[score_col] = numerator / denominator.replace(0, float('nan'))
    df[score_col] = df[score_col].round(2)

    # Create a new column for fused votes, including NaN when the sum is zero
    df[votes_col] = df[votes_col1].fillna(0) + df[votes_col2].fillna(0)
    df[votes_col] = df[votes_col].replace(0, float('nan'))

    # Drop the unnecessary columns
    df = df.drop([score_col1, score_col2, votes_col1, votes_col2], axis=1)
    return df


def fuse_duplicates_v2(df, col_check, year, runtime, col_null, col_score, col_weight):
    df_clean = df.copy(deep=True)
    df_clean[runtime] = df_clean[runtime].fillna(-1)
    for c in col_check:
        duplicates = df_clean[df_clean.duplicated([c, year, runtime], keep=False)]  
        if not duplicates.empty:
            print(f'Fusing duplicates: ')

            for index, group in duplicates.groupby([c, year, runtime]):
                if len(group) > 1:
                    higher_index = group.index.max()
                    lower_index = group.index.min()
                    # Fuse 'release_month', 'box_office_revenue', 'runtime'
                    for col in col_null:
                        if pd.isnull(group.loc[lower_index, col]) and not pd.isnull(group.loc[higher_index, col]):
                            df_clean.at[lower_index, col] = group.loc[higher_index, col]
                        elif not pd.isnull(group.loc[lower_index, col]) and not pd.isnull(group.loc[higher_index, col]):
                            if group.loc[lower_index, col] != group.loc[higher_index, col]:
                                # Calculate mean if values are different
                                mean_value = group.loc[:, col].mean()
                                df_clean.at[lower_index, col] = mean_value
                    
                    # Calculate weighted average for col_score
                    weighted_average = (group.loc[lower_index, col_score] * group.loc[lower_index, col_weight] +
                                        group.loc[higher_index, col_score] * group.loc[higher_index, col_weight]) / \
                                        (group.loc[lower_index, col_weight] + group.loc[higher_index, col_weight])

                    # Update col_score with the weighted average
                    df_clean.at[lower_index, col_score] = round(weighted_average, 1)

                    # Update col_weight with the sum of weights
                    df_clean.at[lower_index, col_weight] = group.loc[lower_index, col_weight] + group.loc[higher_index, col_weight]

                    df_clean = df_clean.drop(higher_index)

            print('Duplicates fused successfully.')
            print('-' * 80)
        else:
            print(f'No duplicates')
            print('-' * 80)
    
    df_clean[runtime] = df_clean[runtime].replace(-1, pd.NA)
    return df_clean.reset_index(drop=True)


def fuse_scores_stats(df, score_col1, score_col2, votes_col1, votes_col2):
    # Filter rows where score_col2 is NaN
    nan_rows = df[pd.isna(df[score_col2])]

    if not nan_rows.empty:
        # Create a new column for fused scores
        numerator = (nan_rows[score_col1].fillna(0) * nan_rows[votes_col1].fillna(0) +
                     nan_rows[score_col2].fillna(0) * nan_rows[votes_col2].fillna(0))
        
        denominator = nan_rows[votes_col1].fillna(0) + nan_rows[votes_col2].fillna(0)

        # Put fused ratings in score_col2, avoid division by zero
        nan_rows.loc[:, score_col2] = numerator / denominator.replace(0, float('nan'))

        # Put fused votes in votes_col2, including NaN when the sum is zero
        nan_rows.loc[:, votes_col2] = nan_rows[votes_col1].fillna(0) + nan_rows[votes_col2].fillna(0)
        nan_rows.loc[:, votes_col2] = nan_rows[votes_col2].replace(0, float('nan'))

        # Update the original DataFrame with the modified rows
        df.loc[nan_rows.index] = nan_rows

    return df


def fuse_winner_columns(df, winner_x_col, winner_y_col):
    """
    Fuse the 'winner_x' and 'winner_y' columns into a single 'winner' column,
    prioritizing non-null values. Drop the original 'winner_x' and 'winner_y' columns.

    Parameters:
    - df: DataFrame, the input DataFrame
    - winner_x_col: str, the column name for 'winner_x'
    - winner_y_col: str, the column name for 'winner_y'

    Returns:
    - DataFrame, the modified DataFrame with a single 'winner' column
    """
    df['winner'] = np.where(
        df[winner_x_col].notnull() & df[winner_y_col].notnull(),
        df[winner_x_col],
        np.where(
            df[winner_x_col].notnull(),
            df[winner_x_col],
            np.where(
                df[winner_y_col].notnull(),
                df[winner_y_col],
                np.nan
            )
        )
    )
    
    # Drop 'winner_x' and 'winner_y' columns
    df = df.drop([winner_x_col, winner_y_col], axis=1)
    
    return df

#-------------------------------------------------------------------------------------------------------
# MANU

def check_years(df):
    """
    Check whether the column 'year' in the dataframe is containing any holes (years within the yearspan of the whole set, for which there exists no data). 

    Parameters:
    - df (pandas.DataFrame): dataframe with at least one column 'years'.

    Returns:
    - no_data_years (list): List of years, for which no data is existing in the dataframe.
    """
    # create list of all years in yearspan of dataframe
    start_year = df.year.min()
    end_year = df.year.max ()
    years = np.arange(start_year, end_year + 1)
    # check whether there exists data for each year
    data_years = df['year'].unique().tolist()
    # create list with the years that have no data
    no_data_years = years[~np.isin(years, data_years)].tolist()

    return no_data_years, start_year, end_year

def revenue_inflation_correction(df, df_inflation):
    """
    Corrects 'revenue' in df by inflation rate described in df_inflation. 
    
    Parameters:
    - df (pandas.DataFrame): dataframe with at least the columns 'year' and 'revenue'. 
    - df_inflation (pandas.DataFrame): dataframe with at least the columns 'year' and 'amount'. Relates the value of money to a reference year (1800). 

    Returns:
    - df_out (pandas.DataFrame): dataframe df with additional column 'revenue_infl', which is the revenue but corrected to account for inflation.
    """
    # preparing the dataset
    no_data_years, start_year, end_year = check_years(df)
    df_inflation_prep = df_inflation[(df_inflation['year'] >= start_year) & (df_inflation['year'] <= end_year) & (~df_inflation['year'].isin(no_data_years))][['year','amount']]
    # merge data on 'year'
    df_out = pd.merge(df, df_inflation_prep, on='year', how='left')
    # divide 'revenue' by 'amount' to get 'revenue_infl' in US$1800
    df_out['revenue_infl'] = df_out['revenue'] / df_out['amount']
    # drop 'amount' and 'revenue' columns
    df_out = df_out.drop(columns=['amount'])
    display(df_out.sample(5))

    ## visualistaion
    # calculate the yearly total of revenues
    revenue_year_infl = df_out.groupby(['year']).revenue_infl.sum()
    revenue_year_orig = df_out.groupby(['year']).revenue.sum()
    years_in_df = df_out['year'].unique().tolist()
    years_in_df.sort()
    # Plot the adjusted and original yearly total revenues
    plt.semilogy(years_in_df, revenue_year_infl, label='Inflation Adjusted Revenue')
    plt.semilogy(years_in_df, revenue_year_orig, label='Original Revenue')
    plt.title('Revenue Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Revenue [US$]')
    plt.legend()
    plt.show()

    return df_out

def revenue_normalisation(df):
    """
    Normalizes 'revenue_infl' in df via a regression analysis. 
    
    Parameters:
    - df (pandas.DataFrame): dataframe with at least the columns 'year', 'revenue' and 'revenue_infl'. 
    
    Returns:
    - df_out (pandas.DataFrame): dataframe df with additional column 'revenue_norm', which is the revenue but corrected to account for inflation.
    """
    # define predictor and dependent variables
    X = df['year'].unique().tolist()
    X.sort()
    revenue_year_infl = df.groupby(['year']).revenue_infl.sum()
    y = revenue_year_infl.astype(float)
    y = np.asarray(y)
    X = np.asarray(X)
    # Create a statsmodels regression model
    model = sm.OLS(y, X).fit()
    # Print the regression results
    print(model.summary())
    # Predict the revenue using the model
    y_pred = model.predict(X)
    # normalize the data of each year
    revenue_normalized = revenue_year_infl - (y_pred - y_pred[0]*np.ones(y_pred.size))

    # prepare for merging
    revenue_normalized = revenue_normalized.reset_index()
    revenue_normalized.rename(columns={"year": "year", "revenue_infl": "revenue_norm_tot"}, inplace=True)
    revenue_normalized
    # merge data on 'year'
    df_out = pd.merge(df, revenue_normalized, on='year', how='left')
    # calculate revenue for each movie
    df_out['revenue_norm'] = df_out['revenue_part'] * df_out['revenue_norm_tot']
    # drop 'revenue_norm_tot' and 'revenue_infl' columns
    df_out = df_out.drop(columns=['revenue_norm_tot']) # optionally drop 'revenue_infl' and 'revenue' too
    display(df_out.sample(5))

    ## visualisation
    # check normalization
    revenue_year_norm = df_out.groupby(['year']).revenue_norm.sum()
    revenue_year_orig = df.groupby(['year']).revenue.sum()
    # Plot the original, inflation corrected and normalized data points
    plt.semilogy(X, revenue_year_orig, label='Original Data')
    plt.semilogy(X, revenue_year_infl, label='Inflation corrected Data')
    plt.semilogy(X, revenue_year_norm, label='Normalized Data')
    plt.title('Revenue Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Revenue [US$]')
    plt.legend()
    plt.show()

    return df_out