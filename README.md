MOVIE GROSS PREDICTION

Abstract- 
Important factors influencing moviegross are genre, directors, actors, productionhouse that makes the movie but everybodymisses the sentiment of people which play animportant role in determining the success of amovie. In this report we present a model whichwill predict the gross income of any movie whiletaking into consideration attributes whichrepresent people’s views before it is released onthe box office. Using this model we were able topredict gross income of movies which werereleased in the first week of May 2017.


DATA COLLECTION

We first extracted our raw movie data from IMDB website. The data obtained included 5043 records and had 28 attributes which included directors names, actor names, number of critic reviews, duration of movie, director Facebook likes, actors Facebook likes, gross, budget, genre, movie title, number of votes users, imdb link, number of users for review, imdbVotes and score, Awards won by the cast of movie, Production house and some more parameters. Since we needed release date of the movies in our data we extracted that data from OMDB database. OMDB provides an API key to connect to it which in return gets data from IMDB. By providing the imdb id we can extract the movie data for that id. After collecting the required data we removed the unwanted columns from the data and cleaned the Excel data which included unwanted characters. After merging the data we had 43 attributes also since genre was a categorical variable we had to convert it to binary to check individual effect of genre on gross of movie. Since data collection is the most important part of analysis we spent most of our time cleaning the data and removing the null values in Python. As we had data from year 1916 to 2016 initially we started our analysis steps considering all the data but then realized that the past years 1916 data was prune to inflation factors and this data affected our gross model prediction and we were not getting accurate results as the movies in 1916 did not make much gross as they used to have very less budget in those days for movies as compared to the movies released in 20’s. So for our analysis we considered data from year 2010-2016 of recent years with 1510 data rows.

DATA ANALYSIS

Our first step in Analysis started with correlation test in Python. Correlation test helps to find out which parameters are closely related to our predictor variable which is Gross. The value of correlation coefficient varies between +1 and -1 [5]. When the value of the correlation coefficient lies around ± 1, then it is said to be a perfect degree of association between the two variables. As the correlation coefficient value goes towards 0, the relationship between the two variables will be weaker. The positive sign + indicates a positive relationship between the variables and negative sign – indicates a negative relationship between the variables. The sign of correlation is important. We plotted correlation matrix using heat map in Python Fig 1. Also Table 1. depicts the correlation coefficients of variables with gross. Both the heat map and coefficients shows that Gross has very strong correlation with num_critic_for_reviews, num_voted_users, num_user_for_reviews, budget, movie_facebook_likes, imdbVotes. GenreAdventure, Action and Sci-Fi had good correlation with gross of movie compared to other genres that is these genre movies were amongst the top gross earnings. This can be seen in Figure 2. Figure 2: Top Genres and the gross earned by them in Million Dollars Since budget and gross had a very strong correlation we plotted a scatter plot Figure 3. to see their relation. It can be seen that they have a positive correlation and as budget for a movie increases their gross earned also increases. This is true because more the money spent on a movie for cast, promotions, directors, actors more the movie is hit on Box office and they earn high gross. To understand the growth of gross and budget in movie industry plotted the trend over the years 2010-2016 in Figure 4. It is clear that initially in 2010 when the budget of the movies was less gross earned was also less which went on  increasing in successive years later. Figure 1: Heat map depicting correlation matrix of all variables. Dark red indicates strong positive correlation and dark blue indicates strong negative correlation

CONCLUSION

We have predicted the gross of movies using regression model and to do so we have used features such as
budget, IMDB Votes, Number of critics for Review and movie Facebook likes. We selected these features
based on the output of PCA and correlation. PCA selects the number of components such that the amount
of variance that needs to be explained is greater than the percentage specified by components. Features
selected had multicolinearity and therefore we used Ridge regression to predict the gross. Our model was
tested using MSE and explained variance and MSE was huge whereas explained variance was very low.
This was caused because we previously used last 100 years of data which was not inflation adjusted. To
overcome this problem we fitted our model with only 10 years of data to predict gross of future movies.
And then the results were as expected, with small MSE and explained variance close to 1. Therefore this
data of budget, IMDB Votes, Number of critics for Review and movie Facebook likes are proven to be
capable of predicting movie gross with good accuracy. For future work, we plan to take data from more
sources and adjust its inflation to current date and see if that increase the model’s accuracy. We are also
planning to quantify actors and directors to build the feature vector to predict the gross.
