'''
 File name: helpers.py
 Author: TheWestBobers
 Date created: 04/12/2023
 Date last modified: 04/12/2023
 Python Version: 3.11.4
 '''

from implementations import *

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import nltk
import re
import statsmodels.api as sm

import pyspark.pandas as ps
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

import itertools

#-------------------------------------------------------------------------------------------------------
# ARTHUR
def select_subsets(movies):
    '''Return subsets of the 'movies' dataset, by genres
    don't use genres with too few movies'''
    min_len = 10

    all_genres = list(set(itertools.chain.from_iterable(movies.genres.tolist())))
    all_genres.sort()
    subsets = [(s,create_subset(movies,s)) for s in all_genres]
    subsets = [element for element in subsets if len(element[1])>min_len]
    return subsets

def viz_subset(i, subsets, movies):
    '''Visualize a subset i'''
    key = subsets[i][0]
    subset = subsets[i][1]
    print('Subset: {}'.format(key))
    print("\t{} | {} (size subset | movies)".format(len(subset),len(movies)))
    print("\t= {} %".format(round(len(subset)/len(movies), 4)))

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
    ax_settings(axs[2], xlabel='Year', ylabel='% of the year\'s market', title='Subset : {}'.format(key))
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

    # # Keep only peaks of quality good enough
    # tr = 0.2
    # peaks = [e for i, e in enumerate(peaks) if quality[i] > tr]
    # inflexions = [e for i, e in enumerate(inflexions) if quality[i] > tr]
    # quality = [e for i, e in enumerate(quality) if quality[i] > tr]
    return peaks, quality

def get_peaks(movies, subsets, i):
    '''Get peaks of a subset, and their quality'''
    # Preprocess the subset
    subset = subsets[i][1]
    movies_by_year = movies.groupby('year').count()['id_wiki']
    distrib = (subset.groupby('year').count()['id_wiki'] / movies_by_year).fillna(0)
    frac = len(subset)/len(movies)

    # Find peaks and quality
    peaks, quality = peak_detection(distrib, frac)
    return list(distrib.index[peaks]), quality

def viz_peaks(movies, subsets, i):
    '''Visualize the peaks or trends of a subset i'''
    key = subsets[i][0]
    subset = subsets[i][1]
    movies_by_year = movies.groupby('year').count()['id_wiki']
    distrib = (subset.groupby('year').count()['id_wiki'] / movies_by_year).fillna(0)
    frac = len(subset)/len(movies)

    # Low pass filter
    x = butter_lowpass_filter(distrib, cutoff=5, fs=len(distrib), order=3)

    # Find peaks and quality
    peaks, quality = peak_detection(distrib, frac)

    # Plot the data
    fig, axs = plt.subplots(1,1,figsize=(12, 6))
    plt.plot(distrib.index, distrib, label='Original signal')
    plt.plot(distrib.index, x, label='Smoothed signal')
    plt.plot(distrib.index[peaks], x[peaks], "o", color='k', label='Peaks')
    plt.plot(distrib.index, np.zeros_like(distrib), "--", color="gray")
    plt.plot(distrib.index, np.ones_like(distrib)*frac, "--", color="red", label='Subset overall fraction')
    for p, q in zip(peaks, quality):
        year = distrib.index[p]
        value = x[p]
        tr = 0.2
        c = 'red' if q < tr else 'k'
        y_offset = 0.05 * (plt.ylim()[1] - plt.ylim()[0])
        plt.text(year+5, value, str(year), ha='center', va='bottom', fontsize=15, color=c)
        plt.text(year+5, value+y_offset, 'Q:'+str(round(q,3)), ha='center', va='bottom', fontsize=12, color=c)
        plt.text(distrib.index[0], frac+y_offset/2, 'overall fraction: '+str(round(frac,3)), ha='left', va='bottom', fontsize=8, color='red', alpha=0.5)
    plt.xlabel('Year')
    plt.ylabel('% of the year\'s market')
    plt.title('Subset : {}'.format(key))
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

def get_trends(movies, subsets, threshold):
    return [(s[0], [p for p,q in zip(*get_peaks(movies, subsets, i)) if q>threshold]) for i, s in enumerate(subsets)]

def range_search(subsets, key, year, range_search):
    '''Return a dataframe of a movies subset within a range, before a date'''
    subset = subsets[find_subset(subsets, key)][1]
    range_results = subset[(subset.year<year) & (subset.year>=year-range_search)]
    return range_results

def get_candidates(subsets, trends):
    '''Return all candidates movies to be pivotal, for each subset
    Output format: array of ('name', year, DF)'''
    candidates = [(trend[0], peak, range_search(subsets, trend[0], peak, 10)) for trend in trends
                                                                              for peak in trend[1]]
    return candidates

def find_candidates(candidates, key, year=None):
    result = []
    for i, c in enumerate(candidates):
        if c[0]==key:
            if year==None:
                result.append(i)
            elif year=='first':
                return i
            elif c[1]==year:
                return i
    return result

def show_candidates(candidates, i):
    '''Display candidates for trend i'''
    print('Candidates of pivotal of genre {}, for trend in {}'.format(candidates[i][0],candidates[i][1]))
    return candidates[i][2]

#-------------------------------------------------------------------------------------------------------
# PAUL
def tokenize_and_stem(text, stemmer=None):
    if stemmer is None:
        stemmer = SnowballStemmer("english")
    
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = [token for token in tokens if re.search('[a-zA-Z]', token)]
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def plot_similarity_heatmap(data_frame, text_column):
    # Merge DataFrame with movie data
    

    # Initialize SnowballStemmer
    stemmer = SnowballStemmer("english")

    # Tokenize and stem the text data
    

    # Create TF-IDF matrix
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000, min_df=0.1,
                                       stop_words='english', use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(data_frame[text_column])

    # Calculate cosine similarity
    similarity_distance = cosine_similarity(tfidf_matrix)

    # Create a subset of the matrix (e.g., first 100 rows and columns)
    subset_matrix = similarity_distance[:100, :100]

    # Create a heatmap using Seaborn
    sns.set(style="white")  # Optional: Set the background style
    plt.figure(figsize=(10, 8))  # Set the figure size
    sns.heatmap(subset_matrix, cmap="viridis", annot=False, fmt=".2f", linewidths=.5)

    # Show the plot
    plt.title('Subset of Similarity Distance Matrix')
    plt.show()

    return similarity_distance

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
    before_year = release_year - 5
    after_year = release_year + 5

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


#-------------------------------------------------------------------------------------------------------
# MEHDI

def clean_gross_income(value):
    if isinstance(value, str):
        value = value.replace(',', '').replace('$', '')
        if 'M' in value:
            value = float(value.replace('M', '')) * (10**6)
    return float(value)


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