# Pivotal Movies: Catch them if you can!

## Abstract
Pivotal movie can be characterized as a film that has significantly influenced the industry in the subsequent years of its release, by creating a trend. Such movies might be key representatives in showcasing the diversity and evolution of the cinematic landscape. Moreover, pivotal movies often become important cinematographic references. They might pioneer new techniques in visual effects, storytelling, or character development, which future filmmakers adopt or adapt in their work. They are more than just successful, they are inspirations for change and innovation in the world of cinema. They leave an indelible mark on the industry, influencing generations of filmmakers and shaping the trajectory of films.


# Datastory

[The West Bobers Website](https://davidlacour.github.io/TheWestBobersWebsite/)
https://davidlacour.github.io/TheWestBobersWebsite/


# The Data

The data for our analysis is gathered from different sources, resulting in one big dataframe, in which we collect all project-relevant information from the datasets. 


## CMU Movie Summary Corpus

The [CMU Movie Summary Corpus](https://www.cs.cmu.edu/~ark/personas/) dataset serves as the base dataframe, which we enriched with information from the other sets. It includes the Wikipedia summary of 42’306 movies, along with metadata (movie box office revenue, genre, release date, runtime, language) that has been extracted from Freebase. There is even more information included in this dataset, which is however not of importance for this project. 


## Movielens

The dataset [Movielens](https://grouplens.org/datasets/movielens/) contains 45’466 movies, including their budget, box office revenue and reviews. With the help of this dataset we can fill some gaps in the revenues, as well as add the movie’s budget and create a review score. 


## MovieStats

The dataset [MovieStats](https://github.com/danielgrijalva/movie-stats) contains 7’668 movies, including their budget, box office revenue and reviews, by scrapping IMDb between 1980 and 2010. We include this set to complete missing values and compute more precise metrics. 


## Award dataset

The [Awards](https://www.kaggle.com/datasets/unanimad/the-oscar-award) dataset includes the annually nominees and award winners of the well known “Oscars”, the awards for artistic and technical merit in the film industry given annually by the Academy of Motion Picture Arts and Sciences (AMPAS). This data helps to bring another measure of quality of the movies to the dataset. 


## US dollar Inflation

The revenues over all datasets are given in US dollars. Due to inflation, one dollar in 1914 (the earliest date of release for a movie in the dataset) compared to one dollar in 2012 (the most recent movie in the dataset) is not worth the same. The purchasing power of the money has changed and this effect has to be accounted for, if two movies from different moments in time should be compared to each other. The [effect of inflation](https://www.officialdata.org/us/inflation/1800?amount=1#buying-power) can be accounted for with this additional dataset, relating the worth of US dollar in each year to the worth of US dollar in 1800. This is achieved with the help of this dataset. 


## IMDB datasets:

These two datasets ([1](https://www.kaggle.com/datasets/ashishjangra27/imdb-movies-dataset) and [2](https://www.kaggle.com/datasets/ashirwadsangwan/imdb-dataset))contain data for 2.5 million movies and series, listed on the official website of IMDB. They help us to fill gaps in the other dataframes and bring more completeness to our data. 


## Final Dataframe

We performed processes of data exploration, cleaning and augmentation by merging the datasets. The exact procedure can be consulted in detail in the notebook [milestoneP3](milestoneP3.ipynb).

We ended up with a clean dataset, containing movies in English that have been produced in the USA between 1910 and 2010. The final dataframe includes the following characteristics for each movie: 

- id_wiki: The ID of the movie for Wikipedia. 
- name: The title of the movie. 
- year: The release year of the movie. 
- alt_name: The alternative title of the movie.
- revenue_norm: The normalized box office revenue, which has been corrected for inflation effects. 
- rating: The movies rating (put together from what?)
- votes: The number of ratings the movie received. 
- genres: To what genres the movie belongs to. 
- nbr_won: The number of “Oscars” won. 
- nbr_nomination: The number of categories, the movie has been nominated in for the “Oscars”. 


# Organization within the team 👥

<img src="images/organigram.PNG" style="width:40%; height:auto;">

| Team Member | Tasks |
|-------------|-------|
| David       | - website - plots interactifs |
| Manuel      | - augmentation - website |
| Arthur      | - shape analysis - trends - ML|
| Mehdi       | - augmentation - cleaning - website |
| Paul        | - cleaning - ML - network |

