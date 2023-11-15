# Pivotal Movies

Readme.md (up to 1000 words)

## Abstract (150 words)
- Abstract: A 150 word description of the project idea and goals. What’s the motivation behind your project? What story would you like to tell, and why?

Pivotal movie is a movie that has influenced the industry in the subsequent years of its release, by creating a trend. First, they might be good representatives of the landscape of movies diversity. But they might also be important cinematographic references.

To determine a pivotal movie, our first approach is to analyze the distribution over time of a subset (eg. genre), and identify whether an unusual shape occurs, such as a bump and high variations.
The idea would be then to select a range of movies prior to the peak, and compare their box offices and reviews. Then, we could draw from this range the pivotal movie that may have produced the particular shape.

## Research Questions 🔎
Research questions we would like to address during the project:
- Which movies are pivotal in history of cinema ?
- How to detect them ?
- Can we bring context to explain why are they pivotal ?
- In which subset of movies can we detect pivotal movies, with a trend shape ?

## Additional datasets 📈
- Proposed additional datasets (if any): List the additional dataset(s) you want to use (if any), and some ideas on how you expect to get, manage, process, and enrich it/them. Show us that you’ve read the docs and some examples, and that you have a clear idea on what to expect. Discuss data size and format if relevant. It is your responsibility to check that what you propose is feasible.

To have more precise metrics, we’d like to add another dataset to use movie reviews (grade from 0 to 5 stars). We’d like to use the Movielens dataset that seems complete and clean. https://grouplens.org/datasets/movielens/ 

## Methods ⚙️

We typically recognize an unusual shape if it differs from the baseline. To illustrate this, notice how the subset “Teen” differs from the whole data release date distribution (Fig. 1-2).
For this distribution analysis approach, it is important to select a good range prior to the trend peak. Let’s say the production of a movie takes 1 year, maybe we could select a range of 5 prior years. Otherwise, another method that requires more work and hypothesis would be to identify a bump as a roughly (skewed) gaussian curve. Then we could select a range of 1-2 standard deviations prior to the mean-median-mode. (Fig. 4)

We’ll filter the dataset, to be more precise on the data covered. We decided to focus on the movies produced by the US. We may introduce more filtering later if needed…

There are several ways to create a subset. It has to be relevant enough to analyze a trend. The easier approach is to use genres of movies, but other methods could be interesting to investigate. For example, we could extract vocabulary from a summary (ex: spaceships). By the way, our dataset provides us very interesting substance : Harvard processed NLP which extracts tropes from summaries (type of character in a movie). Thus, we could analyze occurrences of tropes over time, and possibly draw trends.

Another detail to be careful of is the normalization of data. As we can see on Fig. 1, the number of movies released exploded recently. Then a rough analysis of the distribution wouldn’t be robust. To get more interesting results, we’d like to compare and visualize the fraction of movies from a specific subset. A nice visualization would be a stacked plot to combine both number of releases and fractions of subsets. Here, a problem we may encounter is the high number of genres, because that would require too many colors and overload the graph. An idea to solve this issue is to group genres into 5-10 main categories, and have a more readable plot. For example the genre “airplanes and airport” isn’t that representative yet, for a first visualization, however it could be that this category reveals a peak of trend with further analysis…
We’re also thinking of the impact of inflation on the revenue (see Fig. 5). It would be a good thing to adapt the box-office to the real value of money according to its release year.

We’d like to introduce a ML approach to automate the research of pivotal movies (see Fig. 3). By selecting features that capture the “trend”, and creating a training set with movies identified as pivotal and not (from the distribution analysis approach). Then we’ll fine tune weights of the trend features to have a robust model, and apply it to the whole dataset then possibly reveal more pivotal movies.
Maybe to reduce the complexity of the algorithm, we could analyze the dataset only by a 10-year tranche.
FIRST IDEA OF FEATURES: subset (genre, tropes, voc), release date, box-office, review

## Proposed timeline ⏳

## Organization within the team 👥
- Organization within the team: A list of internal milestones up until project Milestone P3.

## Questions for TAs 📝
- Questions for TAs (optional): Add here any questions you have for us related to the proposed project.
