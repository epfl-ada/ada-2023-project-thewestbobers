'''
 File name: helpers.py
 Author: TheWestBobers
 Date created: 04/12/2023
 Date last modified: 04/12/2023
 Python Version: 3.11.4
 '''

from implementations import *

import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks

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
    '''Get peaks of a subset'''
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
    peaks, _ = find_peaks(x, width=5, height=frac, prominence=frac)
    return list(distrib.index[peaks])

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