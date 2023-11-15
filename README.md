# Pivotal Movies

Readme.md (up to 1000 words)

## Abstract (150 words)
Pivotal movie is a movie that has influenced the industry in the subsequent years of its release, by creating a trend. First, it might be a good representatives in the landscape of movies diversity. But they might also be an important cinematographic reference.
The [CMU Movies Summary Corpus](https://www.cs.cmu.edu/~ark/personas/) contains data of movies release date, genres and box-office that are crucial for our analysis.

## Research questions 🔎
Research questions we would like to address during the project:
- Which movies are pivotal in history of cinema ?
- How to detect them ?
- Can we bring context to explain why they are pivotal ?
- In which subsets can we detect pivotal movies, with a trend shape ?

## Additional datasets 📈
- Discuss data size and format

[Movielens](https://grouplens.org/datasets/movielens/) contains movie reviews (grade from 0 to 5 stars), it will allow us to compute more precise metrics.

[MovieStats](https://github.com/danielgrijalva/movie-stats) contains movies budget and box office, by scrapping IMDb, to complete missing values from our dataset.

## Methods ⚙️

### Step 1: Pre-processing
Pre-processing consist in formatting the data in a way that facilitate further analysis and computations. We will handle missing data and outliers, and normalize the data.

We will also filter the dataset, to be more precise on the data covered. We decided to focus on the movies produced by the US. We could introduce more filtering later if needed…

### Step 2: Subsets
There are several ways to create a subset. It has to be relevant enough to analyze a trend. The easier approach is to use genres of movies, but other methods could be interesting to investigate. For example, we could extract vocabulary from a summary (ex: spaceships). By the way, our dataset provides us very interesting substance : processed NLP which extracts tropes from summaries (type of character in a movie). Thus, we could analyze occurrences of tropes over time, and possibly draw trends.

We will start by creating simple subsets of genres. Then we will explore other ideas as extra.

### Step 3: Shape analysis
We noticed that the number of movies over time has exploded in early 2000's (Fig. 1), then a rough analysis of the distribution wouldn’t be robust. To get more interesting results, we’d like to compare and visualize the evolution of fraction of movies from a specific subset. By plotting this curve, we are seeking an unusual shape, such as a bump or high variation. We typically recognize an unusual shape if it differs from the baseline (constant, for fractions).

A nice visualization would be a stacked plot to combine both number of releases and fractions of subsets. Here, a problem we might encounter is the high number of genres, because that would require too many colors and overload the graph. An idea to solve this issue is to group genres into 5-10 main categories, and have a more readable plot. For example the genre “airplanes and airport” isn’t that representative yet for a first visualization, however it could be that this category reveals a peak of trend with further analysis…

### Step 4: Range selection of prior movies
Once the unusual shape(s) has been identified, we will select a range prior to the trend peak, assuming the pivotal movie lies inside of it. It is important to choose a proper range so we don't miss the pivotal movie (too short range), and we don't predict a movie without relation (too big range). Let’s say the production of a movie takes 1 year, the first approach is to select a range of 5 prior years, which seems reasonable. Otherwise, a more precise method that requires more work and hypothesis would be to identify a bump as a roughly (skewed) gaussian curve. Then we could select a range of 1-2 standard deviations prior to the mean/median/mode (Fig. 4).

### Step 5: Pivotal Score
Finally, we will elect the most probable pivotal movie of the selected range, which maximizes a score. From our definition of pivotal movie, the score would be based on money generated (which reflects how many people watched the movie) and public advise (how was the movie recieved). The metrics used here would be box-office and review score. Then if several movies reached the top score within a certain threshold, our intuition is to prefer the earliest movie released, because it would be the most likely to influence later releases.

We might investigate further metrics, such as differentiating public and press review score. We’re also thinking of the impact of inflation on the revenue (see Fig. 5). It would be interesting to adapt the box-office to the real value of money according to its release year. Then observe if this changes the pivotal movie selected.

### Further steps: ML approach
We’d like to introduce a ML approach to automate the research of pivotal movies (see Fig. 3). By selecting features that capture the “trend”, and creating a training set with movies identified as pivotal and not (from the distribution analysis approach). Then we’ll fine tune weights of the trend features to have a robust model, and apply it to the whole dataset then possibly reveal more pivotal movies.
Maybe to reduce the complexity of the algorithm, we could analyze the dataset only by a 10-year tranche.
FIRST IDEA OF FEATURES: subset (genre, tropes, voc), release date, box-office, review

## Methods (Paul) ⚙️
Genre Trend Analysis : Genre Trend Analysis begins with the task of identifying trends within the dataset. To this aim the dataset is segmented into subsets of genres such as Adventure, Drama, and War Film for example. A key aspect of this analysis involves plotting the release frequency of films within each genre subset. By closely examining these plots, discernible trends can be identified. To enhance the precision of trend detection, we take a step further by analyzing the variation in the percentage share of film release in a specific genre from one year to the next. For example, we calculate the percentage increase in the market share of Sci-Fi movies in 1980 compared to 1979. A significant percentage increase indicates the emergence of a trend in Sci-Fi movies during that period. 

3.	Finding a list of potential pivotal movie if any. 
Now we know when the trends in specific genre appears. We need to identify movie that could have caused this trend. To procede we need to look at the past of this trend. We know that if a film provoke a trend then others film that want to copy this success we’ll not be release two days after. Indeed, from theses sources we see that make a movie from scratch take 2-3 years. In order to take into account this delay we take a window of 5 years before the trend and look at which film of the same genre of the trend were release the past 5 years. After taking this film subset we need to determine which film was a success. Two criteria can stand for the success of film: revenue and rating. In deed, if a movie make big revenue other film producter will copy because they see that this type of film provoke an interest . Same thing for rating.  But we need to be careful to the analysis of these features because they can be dependant. To clarify this step let’s take an example: we analysed a trend in Sci-fi movie in 1980 in first part, then in this part we will look back at film from 1975 to 1979 and retains film that made big revenue and have good rating compared to other films of the same genre in this period.

Now that we have identified when trends in specific genres emerge, the next step is to pinpoint the movies that could have catalyzed these trends. To accomplish this, we delve into the history of the identified trend. Recognizing that films influencing a trend wouldn't be released immediately, we acknowledge the time it takes to produce a movie from scratch, typically spanning 2-3 years. To account for this time lag, we establish a window of 5 years before the onset of the trend and scrutinize films within the same genre released during this period. Within this subset, our focus narrows down to determining which films were successful. Success is gauged by two criteria: revenue and rating. High revenue indicates commercial success, and a good rating signifies audience appreciation. However, it is crucial to approach this analysis with care, considering the potential interdependence of these features. For instance, in analyzing a Sci-Fi trend in 1980, we retrospectively examine films from 1975 to 1979, selecting those that not only garnered significant revenue but also exhibited high ratings compared to other films in the same genre during that timeframe. This meticulous approach ensures a nuanced understanding of the films contributing to and shaping genre trends.
After narrowing down our list to potential movies, the next crucial step is identifying the pivotal one – a film so successful that it becomes a prototype for others seeking to replicate its success. In this phase, we conduct a thorough analysis of the similarities within our shortlisted movies and those released during the genre trend. This entails assessing plot similarities by evaluating the number of keywords shared between two movies, comparing character types to identify commonalities, and considering cross-genre elements shared by the films. The sixth step, "Comparison with Predecessors," might not suffice on its own. Since a pivotal movie is the catalyst for a trend, it may introduce something novel. To explore this, we replicate the previous analysis but this time by delving into the past to identify films with no apparent similarities. This comprehensive selection and analysis conclude our pivotal movie examination, ultimately revealing the most influential film that initiated the trend. An intriguing avenue for further exploration could involve analyzing the similarities among these pivotal movies to uncover any distinctive and noteworthy aspects that contribute to their trend-setting impact.


## Proposed timeline ⏳
```
.
├── 27.11.23 – Pre-processing and additionnal datasets
│  
├── 04.12.23 – Create subsets for analyzing
│  
├── 08.12.23 - Shape analysis of subsets and range selections
│  
├── 11.12.23 – Pivotal scores and metrics
│  
├── 15.12.23 – Further investigate
│            ├── Extra subsets
│            ├── Tune range selection
│            ├── Tune metrics
│            ├── ML approach
│    
├── 18.12.23 – Data story
│  
├── 20.12.23 – Final touch
│  
├── 22.12.23 – Milestone 3 deadline
.

```

## Organization within the team 👥
| Team Member | Tasks |
|-------------|-------|
| @David       | - Shape analysis. <br> - Extra subsets. |
| @Manuel     | - Additional datasets. <br> - Data story. |
| @Arthur      | - Subsets. <br> - Data story. |
| @Mehdi       | - Visualizations. <br> - Range selection. |
| @Paul        | - Shape analysis.<br> - Pivotal scores and metrics. |

| Team Member | Tasks |
|-------------|-------|
| @David       | - Work on criterion to find data similarity between plots. <br> - Find the similarity criteria for both plot and character types for potential pivotal movie subset. <br> - Analyze the results. |
| @Manuel     | - Extract data from character clusters. <br> - Creation of potential pivotal movie subset. <br> - Analyze the results. |
| @Arthur      | - Find a method to caracterize the success of a movie and method for ranking the most successful movies in a period. <br> - Creation of potential pivotal movie subset. <br> - Analyze the results. |
| @Mehdi       | - Find trends in movie history. <br> - Analyze similarity criteria for cross-genre elements. <br> - Rank the pivotal movie subset to identify the most pivotal film. <br> - Create a data story. |
| @Paul        | - Find trends in movie history. <br> - Find the similarity criteria for both plot and character types for potential pivotal movie subset. <br> - Create a data story. |

## Questions for TAs 📝
- Questions for TAs (optional): Add here any questions you have for us related to the proposed project.
