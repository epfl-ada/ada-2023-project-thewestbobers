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
import json

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

