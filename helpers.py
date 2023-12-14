'''
 File name: helpers.py
 Author: TheWestBobers
 Date created: 04/12/2023
 Date last modified: 04/12/2023
 Python Version: 3.11.4
 '''

from implementations import *

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
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks

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

def get_peaks(movies, subsets, i):
    '''Get peaks of a subset, and their quality'''
    # Preprocess the subset
    subset = subsets[i][1]
    movies_by_year = movies.groupby('year').count()['id_wiki']
    distrib = (subset.groupby('year').count()['id_wiki'] / movies_by_year).fillna(0)
    frac = len(subset)/len(movies)

    # Low pass filter
    x = butter_lowpass_filter(distrib, cutoff=5, fs=len(distrib), order=3)

    # Find peaks
    ## > width is the min size of the trend
    ### let's assume we look for trend that lasted at least 5 years
    ## > height, prominence are thresholds for peak detection
    ### let's set them to the subset overall fraction
    peaks, props = find_peaks(x, width=5, height=frac, prominence=frac)
    prominences = list(props['prominences'])
    quality = prominences/x.max()
    quality = [min(1,q) for q in quality]
    return list(distrib.index[peaks]), quality

def viz_peaks(movies, subsets, i):
    '''Visualize the peaks or trends of a subset i'''
    key = subsets[i][0]
    subset = subsets[i][1]
    movies_by_year = movies.groupby('year').count()['id_wiki']
    distrib = (subset.groupby('year').count()['id_wiki'] / movies_by_year).fillna(0)
    frac = len(subsets[i][1])/len(movies)

    # Low pass filter
    x = butter_lowpass_filter(distrib, cutoff=5, fs=len(distrib), order=3)

    # Find peaks
    ## try wlen for "Window length for calculating the prominence"
    peaks, _ = find_peaks(x, width=5, height=frac, prominence=frac)

    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(distrib.index, distrib, label='Original signal')
    plt.plot(distrib.index, x, label='Smoothed signal')
    plt.plot(distrib.index[peaks], x[peaks], "o", color='k', label='Peaks')
    plt.plot(distrib.index, np.zeros_like(distrib), "--", color="gray")
    plt.plot(distrib.index, np.ones_like(distrib)*frac, "--", color="red", label='Subset overall fraction')
    plt.xlabel('Year')
    plt.ylabel('% of the year\'s market')
    plt.title('Subset : {}'.format(subsets[i][0]))
    plt.grid(alpha=0.3, axis='y')
    plt.legend()
    plt.show()

def find_subset(subsets, key):
    '''Find a peticular subset with its key'''
    result = None
    for i, s in enumerate(subsets):
        if s[0]==key:
            result = i
    return result

def get_candidates(subsets, key, year):
    '''Return a dataframe of candidate movies to be pivotal
    for the peak 'year' of the subset 'key' within 'range_search'.'''
    range_search = 10
    subset = subsets[find_subset(subsets, key)][1]
    return subset[(subset.year<year) & (subset.year>=year-range_search)]

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

def calculate_mean_similarity(similarity_matrix, chosen_movie_index, movie_indices):
   
    # Extract similarity values for the chosen movie and the set of movies
    similarities = similarity_matrix[chosen_movie_index, movie_indices]

    # Calculate the mean similarity
    mean_similarity = np.mean(similarities)

    return mean_similarity
def calculate_mean_similarity(movie_index, merged_df, similarity_matrix, genre):
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

#-------------------------------------------------------------------------------------------------------
