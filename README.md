# Songs-popularity-prediction-with-NLP

# Motivation


The global music market was worth $26bn in 2021 (IFPI, 2022). This number rises each year, which might not be surprising- for many, music could be perceived as an important part of their lives. With technology advancing and moving from ownership concepts, the number of streaming subscribers also increases with time, and accounts for 65% of music industry revenue. (IFPI, 2022).Artists’ revenue can depend on the number of fans attending concerts, streaming statistics, selling licences and on promoting, selling products/services. The revenue is directly correlated with the number of fans (IFPI, 2022), thus if an artist wants to incorporate high profits, one can wonder what makes them popular.
It can be the outcome of a few factors like personality, connections, media appearance, years on stage and music they produce (IFPI, 2022). If focusing solely on music, thus what characteristics of a song make it more successful?
Song features can consist of audio, metadata and lyrics. 

With the use of Natural Language Processing a large amount of lyrics can be analysed. Our goal is to check whether it is possible to be able to predict the rank on the chart solely from the lyrics of a song.


# Research Question

With this paper we want to investigate whether it is possible to create a model capable of predicting the popularity of a specific song, based on its lyrical content. With this purpose two different sub- question have been identified.
Firstly, 20 years of Billboard Hot 100 chart songs’ lyrics are compared with those songs outside the chart but from the same artists to investigate if only focusing on textual features can lead to successfully predicting the presence of a song in a ranking representing the most popular songs. Secondly, in direct connection with the previous point we also want to address an analysis to predict the degree of the popularity and its rank range within the Billboard Hot 100 chart.

# Data Retrieval and Description

Our dataset consisted of 4217 songs in total. Out of those, we labelled 2000 songs as popular and 2217 as not popular. The distinction was made based on the inclusion or exclusion in the Billboard Hot 100 year end charts. The Billboard Hot 100 year-end charts represent the most popular songs in the United States across all genres. The popularity is measured by radio airplay audience impressions,sales data both physical and digital, and streaming activity data. (Billboard, 2022). We collected the lyrics for the Billboard Hot 100 year end charts of the last 20 years which presented our popular songs. After that, we collected songs from the same artists which were not contained in the Billboard Hot 100 year end charts. Those songs represented the not popular songs.
Considering our data collection process for the popular songs, we first sent requests to Wikipedia in order to access the specific page for the Billboard Hot 100 year end charts of each year from 2000- 2021. On each webpage, we scraped the html properties by using the Python library BeautifulSoup to access the table that contained our required information. The responses of our html requests for each year were then stored as a Pandas data frame with the columns “year”, “rank”, “song”, and “artist”. After we collected the data for each year, we exported our data frame as a “.csv” file. The data in this csv file was then used to access the API from Genius (https://docs.genius.com). By calling this API, we were able to add a “lyrics” column to our data which contained the lyrics for each specific song. The process for the unpopular songs left out the Wikipedia scraping and only used the Genius API. We selected artists contained in the popular songs and randomly collected songs from them that were not contained in the Billboard Hot 100 year end charts. The columns for the unpopular songs contained “song”, “artist”, and “lyrics”. The columns “year” and “rank” were not relevant as we only cared about the class “not popular” for this data, and not for a specific chart placement. After finishing our data retrieval process, our dataset consisted of 4217 songs from 1168 unique artists.

# Data Processing
In order to prepare the lyrics for further steps like model building, it is vital to perform data preprocessing. Some of the steps are lowercasing, removing stopwords, lemmatization.
In the dataset there were missing values present. Some of the songs did not have lyrics assigned so we have decided to remove them.
The standard procedure is tokenization, making sure that text is divided into words. Moreover, we converted the lyrics to lowercase. Lower and upper cases are treated differently so it is important to have common cases making it easier for a machine to interpret the content.
Words considered as ‘stop words’ (e.g. the, a, an, there, this) as well as digits were removed as they do not provide much value for any of the further operations we will conduct.
Moreover, punctuation was removed. Those signs are common in the language, thus when working with occurrence and frequency of words, they would affect the result, which is not correct as they can be perceived as redundant. Similarly, symbols like emoji which were wrongly captured by the API were let go. In our analysis, emoji do not bring much value and can be considered as a noise. It might be more sense to keep them when performing sentiment analysis.
Furthermore, we have performed normalisation on the data. The lyrics contained words that are based on one word but with a suffix a new one can be created. It contributes to redundancy as multiple versions are not insightful for model analysis. There are two common ways to obtain the root of the words: lemmatization and stemming. We have proceeded with lemmatization as it outputs words that are a part of the dictionary, contrary to stemming which could return a non-meaningful one.

After preprocessing our data, we had a final dataset of 3801 songs. Out of those, 1901 songs were in the category of popular songs, namely from the Billboard Hot 100 year end charts, and 1900 were in the category of not popular songs. The songs in the dataset were from 1120 unique artists. Table 1 shows the average length of lyrics with and without stopwords. The average length of lyrics is 2156.64 characters or rather 1320.20 characters for lyrics without stop words. Popular songs appear to be longer than non-popular songs on average. In general, we notice a trend of decreasing length of the lyrics over the years as displayed in Figure 2. Considering the distribution of artists and their published songs, Figure 3 shows that most artists are represented with one song only in the dataset. After that, the range of 2 to 7 songs is almost equally distributed. The smallest fraction of artists in our dataset is represented with 8 or more songs.
<img width="718" alt="Screenshot 2023-01-31 at 19 41 06" src="https://user-images.githubusercontent.com/91185911/215852783-950e312c-acd1-44f0-8734-7c59bfdd50b5.png">
<img width="748" alt="Screenshot 2023-01-31 at 19 41 38" src="https://user-images.githubusercontent.com/91185911/215852898-3dd1175e-67c4-42aa-aea5-9bef0fe9c26c.png">
<img width="850" alt="Screenshot 2023-01-31 at 19 41 52" src="https://user-images.githubusercontent.com/91185911/215852948-8c9d2e9c-2ce4-4bae-a2f4-7e9cd5ddb28d.png">



# Modelling Framework 

Three different algorithms were used for the classification task: Logistic Regression, Naive Bayes (Gaussian for binary classification and Multinomial for the multiclass one) and the Random Forest classifier. Another approach was tried, but not included in the final script, because the higher complexity of the model was not reflected in an improvement of the result metric. This involves an embedding through Word2Vec technique and the training of a neural network with two Bidirectional LSTM layers and a dense layer with softmax activation function.
For the presented models, a pipeline was built to have a more neat process. This includes the count vectorizer and tf-idf transformer - which is equivalent to the use of TfidfVectorizer from Sklearn (Pedregosa et al, 2011) and the model in question. For the GaussianNB a FunctionTransformer to transform the train set in a dense form was required before passing the model. On the other hand, the training of the RF is simply done in conjunction with a tf-idf vectorizer.
A Grid Search was carried out for all models but the Logistic Regressor, as the penalties are not fit for all the possible solvers and this would have caused the search to fail. In the case of the NB the parameters involved were the range of n_grams of the vectorizer, the norm and use_idf of the tf-idf transformer, while for the RF classifier the search concerned number of estimators, max features and evaluation criterion.

# Results

For the task 3 different algorithms were used, with the models showing very close results. For what concerns the classification between popular and unpopular songs, Random Forest has outperformed the other models, with an accuracy of 59% and f1-scores of 0.59 and 0.58 for the popular and not- popular class respectively. Nevertheless, results for the other classifiers do not show substantial differences, with the worse classifier - NB- having 51% accuracy and f1-score below 0.55 for both labels.
<img width="607" alt="Screenshot 2023-01-31 at 19 34 02" src="https://user-images.githubusercontent.com/91185911/215851305-010ba38f-96ba-40d0-ae61-e1ef67b0cb4c.png">
<img width="631" alt="Screenshot 2023-01-31 at 19 34 49" src="https://user-images.githubusercontent.com/91185911/215851486-89b4bfc5-2247-49b3-8228-d197324b9a72.png">

When trying to predict the level of popularity (3 brackets; 1 for spots 1-32, 2 for spots 34-66, 3 for spots 67-100) of a Billboard Hot 100 song, Logistic Regression and Random Forest are performing very similarly. They both have a 36% accuracy, but while the first model has f1-scores of 0.36, 0.37, 0.34, the other has f1-scores of 0.33, 0.36, 0.38 for class b_1, b_2, b_3 respectively. On the other hand, in this case the NB model is closer to the performance of the others compared to binary classification, as values are just under 1-2% lower of the best-performing models.

# Conclusion and further remarks

This analysis tries to answer three different research sub questions in relation to the opportunity of using the lyrical content to predict the popularity of a song. For what concerns the possibility of determining whether a song has reached a considerably high popularity- in this case the presence of the track in the yearly Billboard Top 100 - the classification task presents weak results. Indeed, the highest level of accuracy is 59%.
However, it was then inquired if it would be possible to distinguish among different levels of popularity. This was done by applying the same classification algorithms to predict whether a track is ranked among the 30 best songs in the given playlist, or in the 2nd third or 3rd third of the positions. Again, results proved that there is little room for obtaining satisfactory prediction by the exclusive observation of music lyrics.

Given the observation of the results it is clear that the potential expansion of this research in the realm of NLP is quite limited, as music lyrics do not seem to be determinant in evaluating the popularity of a song. However, it could be possible to obtain more meaningful insights by applying other machine learning or deep learning techniques on different aspects of songs. While lyrics represent a relevant component, they are not necessarily fundamental: indeed, it is possible to have instrumental music or tracks where voice is used without expressing actual words. For this reason, other aspects should also be taken into account, both independently and in combination with lyric analysis. This may include numerical audio metadata as well as spectrograms of real audio samples (Pelchat & Gelowitz, 2020).




References:
IFPI (International Federation of the Phonographic Industry) (2022). Global Music Report 2022. IFPI
Pelchat, N., & Gelowitz, C. M. (2020). Neural network music genre classification. Canadian Journal of Electrical and Computer Engineering, 43(3), 170-173
