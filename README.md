# Pivotal Movies

Readme.md (up to 1000 words)

## Abstract (150 words)
- Abstract: A 150 word description of the project idea and goals. What’s the motivation behind your project? What story would you like to tell, and why?

Pivotal movie is a movie that has influenced the industry in the subsequent years of its release, by creating a trend. First, it might be a good representatives in the landscape of movies diversity. But they might also be an important cinematographic reference.
The [CMU Movies Summary Corpus](https://www.cs.cmu.edu/~ark/personas/) contains data of movies release date, genres and box-office that are crucial for our analysis.

Please visit our data story [here]()

## Research Questions 🔎
Research questions we would like to address during the project:
- Which movies are pivotal in history of cinema ?
- How to detect them ?
- Can we bring context to explain why they are pivotal ?
- In which subsets can we detect pivotal movies, with a trend shape ?

## Additional datasets 📈
- Proposed additional datasets (if any): List the additional dataset(s) you want to use (if any), and some ideas on how you expect to get, manage, process, and enrich it/them. Show us that you’ve read the docs and some examples, and that you have a clear idea on what to expect. Discuss data size and format if relevant. It is your responsibility to check that what you propose is feasible.

[Movielens](https://grouplens.org/datasets/movielens/) contains movie reviews (grade from 0 to 5 stars), it will allow us to compute more precise metrics.

[]() contains movies budget and box office.

## Methods (Arthur)⚙️

To determine a pivotal movie, our first approach is to analyze the distribution over time of a subset (eg. genre), and identify whether an unusual shape occurs, such as a bump and high variations.
The idea would be then to select a range of movies prior to the peak, and compare their box offices and reviews. Then, we could draw from this range the pivotal movie that may have produced the particular shape.

We typically recognize an unusual shape if it differs from the baseline. To illustrate this, notice how the subset “Teen” differs from the whole data release date distribution (Fig. 1-2).
For this distribution analysis approach, it is important to select a good range prior to the trend peak. Let’s say the production of a movie takes 1 year, maybe we could select a range of 5 prior years. Otherwise, another method that requires more work and hypothesis would be to identify a bump as a roughly (skewed) gaussian curve. Then we could select a range of 1-2 standard deviations prior to the mean-median-mode. (Fig. 4)

We’ll filter the dataset, to be more precise on the data covered. We decided to focus on the movies produced by the US. We may introduce more filtering later if needed…

There are several ways to create a subset. It has to be relevant enough to analyze a trend. The easier approach is to use genres of movies, but other methods could be interesting to investigate. For example, we could extract vocabulary from a summary (ex: spaceships). By the way, our dataset provides us very interesting substance : Harvard processed NLP which extracts tropes from summaries (type of character in a movie). Thus, we could analyze occurrences of tropes over time, and possibly draw trends.

Another detail to be careful of is the normalization of data. As we can see on Fig. 1, the number of movies released exploded recently. Then a rough analysis of the distribution wouldn’t be robust. To get more interesting results, we’d like to compare and visualize the fraction of movies from a specific subset. A nice visualization would be a stacked plot to combine both number of releases and fractions of subsets. Here, a problem we may encounter is the high number of genres, because that would require too many colors and overload the graph. An idea to solve this issue is to group genres into 5-10 main categories, and have a more readable plot. For example the genre “airplanes and airport” isn’t that representative yet, for a first visualization, however it could be that this category reveals a peak of trend with further analysis…
We’re also thinking of the impact of inflation on the revenue (see Fig. 5). It would be a good thing to adapt the box-office to the real value of money according to its release year.

We’d like to introduce a ML approach to automate the research of pivotal movies (see Fig. 3). By selecting features that capture the “trend”, and creating a training set with movies identified as pivotal and not (from the distribution analysis approach). Then we’ll fine tune weights of the trend features to have a robust model, and apply it to the whole dataset then possibly reveal more pivotal movies.
Maybe to reduce the complexity of the algorithm, we could analyze the dataset only by a 10-year tranche.
FIRST IDEA OF FEATURES: subset (genre, tropes, voc), release date, box-office, review

## Methods (Paul)⚙️
Genre Trend Analysis : Genre Trend Analysis begins with the task of identifying trends within the dataset. To this aim the dataset is segmented into subsets of genres such as Adventure, Drama, and War Film for example. A key aspect of this analysis involves plotting the release frequency of films within each genre subset. By closely examining these plots, discernible trends can be identified. To enhance the precision of trend detection, we take a step further by analyzing the variation in the percentage share of film release in a specific genre from one year to the next. For example, we calculate the percentage increase in the market share of Sci-Fi movies in 1980 compared to 1979. A significant percentage increase indicates the emergence of a trend in Sci-Fi movies during that period. 

3.	Finding a list of potential pivotal movie if any. 
Now we know when the trends in specific genre appears. We need to identify movie that could have caused this trend. To procede we need to look at the past of this trend. We know that if a film provoke a trend then others film that want to copy this success we’ll not be release two days after. Indeed, from theses sources we see that make a movie from scratch take 2-3 years. In order to take into account this delay we take a window of 5 years before the trend and look at which film of the same genre of the trend were release the past 5 years. After taking this film subset we need to determine which film was a success. Two criteria can stand for the success of film: revenue and rating. In deed, if a movie make big revenue other film producter will copy because they see that this type of film provoke an interest . Same thing for rating.  But we need to be careful to the analysis of these features because they can be dependant. To clarify this step let’s take an example: we analysed a trend in Sci-fi movie in 1980 in first part, then in this part we will look back at film from 1975 to 1979 and retains film that made big revenue and have good rating compared to other films of the same genre in this period.

Now that we have identified when trends in specific genres emerge, the next step is to pinpoint the movies that could have catalyzed these trends. To accomplish this, we delve into the history of the identified trend. Recognizing that films influencing a trend wouldn't be released immediately, we acknowledge the time it takes to produce a movie from scratch, typically spanning 2-3 years. To account for this time lag, we establish a window of 5 years before the onset of the trend and scrutinize films within the same genre released during this period. Within this subset, our focus narrows down to determining which films were successful. Success is gauged by two criteria: revenue and rating. High revenue indicates commercial success, and a good rating signifies audience appreciation. However, it is crucial to approach this analysis with care, considering the potential interdependence of these features. For instance, in analyzing a Sci-Fi trend in 1980, we retrospectively examine films from 1975 to 1979, selecting those that not only garnered significant revenue but also exhibited high ratings compared to other films in the same genre during that timeframe. This meticulous approach ensures a nuanced understanding of the films contributing to and shaping genre trends.
After narrowing down our list to potential movies, the next crucial step is identifying the pivotal one – a film so successful that it becomes a prototype for others seeking to replicate its success. In this phase, we conduct a thorough analysis of the similarities within our shortlisted movies and those released during the genre trend. This entails assessing plot similarities by evaluating the number of keywords shared between two movies, comparing character types to identify commonalities, and considering cross-genre elements shared by the films. The sixth step, "Comparison with Predecessors," might not suffice on its own. Since a pivotal movie is the catalyst for a trend, it may introduce something novel. To explore this, we replicate the previous analysis but this time by delving into the past to identify films with no apparent similarities. This comprehensive selection and analysis conclude our pivotal movie examination, ultimately revealing the most influential film that initiated the trend. An intriguing avenue for further exploration could involve analyzing the similarities among these pivotal movies to uncover any distinctive and noteworthy aspects that contribute to their trend-setting impact.


## Proposed timeline ⏳

# Project Timeline

- **27.11.23** – Finding all interesting trends 

- **04.12.23** – Having a subset of pivotal movies
  - Define and gather a subset of pivotal movies
  
- **8.12.23** – Analyze the similarity’s criterion
  - Identify and define criteria for similarity analysis
  
- **11.12.23** – Perform similarity analysis
  - Execute similarity analysis on the subset of movies
  
- **15.12.22** – Find a list of all pivotal movies and start analyzing it
  - Expand the analysis to a comprehensive list of pivotal movies
  
- **18.12.22** – Finalization of project story and cleaning the code
  - Refine the project narrative and clean up codebase
  
- **21.12.22** – Last corrections
  - Address any final corrections or adjustments
  
- **22.12.22** – Milestone 3 deadline
  - Submission of the project for Milestone 3

## Organization within the team 👥
| Team Member | Tasks |
|-------------|-------|
| David       | - Work on criterion to find data similarity between plots. <br> - Find the similarity criteria for both plot and character types for potential pivotal movie subset. <br> - Analyze the results. |
| Manuel     | - Extract data from character clusters. <br> - Creation of potential pivotal movie subset. <br> - Analyze the results. |
| Arthur      | - Find a method to caracterize the success of a movie and method for ranking the most successful movies in a period. <br> - Creation of potential pivotal movie subset. <br> - Analyze the results. |
| Mehdi       | - Find trends in movie history. <br> - Analyze similarity criteria for cross-genre elements. <br> - Rank the pivotal movie subset to identify the most pivotal film. <br> - Create a data story. |
| Paul        | - Find trends in movie history. <br> - Find the similarity criteria for both plot and character types for potential pivotal movie subset. <br> - Create a data story. |

## Questions for TAs 📝
- Questions for TAs (optional): Add here any questions you have for us related to the proposed project.

## Abstract

Pivotal movie is a movie that has influenced the industry in the subsequent years of its release, by creating a trend. In this project, we want to analyse which film of cinema industry can be qualified as pivotal. To determine it, we will analyse features of films 

our first approach is to analyze the distribution over time of a subset (eg. genre), and identify whether an unusual shape occurs, such as a bump and high variations.
The idea would be then to select a range of movies prior to the peak, and compare their box offices and reviews. Then, we could draw from this range the pivotal movie that may have produced the particular shape.
