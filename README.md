# Songs-popularity-prediction-with-NLP

# Motivation


The global music market was worth $26bn in 2021 (IFPI, 2022). This number rises each year, which might not be surprising- for many, music could be perceived as an important part of their lives. With technology advancing and moving from ownership concepts, the number of streaming subscribers also increases with time, and accounts for 65% of music industry revenue. (IFPI, 2022).Artists’ revenue can depend on the number of fans attending concerts, streaming statistics, selling licences and on promoting, selling products/services. The revenue is directly correlated with the number of fans (IFPI, 2022), thus if an artist wants to incorporate high profits, one can wonder what makes them popular.
It can be the outcome of a few factors like personality, connections, media appearance, years on stage and music they produce (IFPI, 2022). If focusing solely on music, thus what characteristics of a song make it more successful?
Song features can consist of audio, metadata and lyrics. 

With the use of Natural Language Processing a large amount of lyrics can be analysed. Our goal is to check whether it is possible to be able to predict the rank on the chart solely from the lyrics of a song.


# Research Question

With this paper we want to investigate whether it is possible to create a model capable of predicting the popularity of a specific song, based on its lyrical content. With this purpose two different sub- question have been identified.
Firstly, 20 years of Billboard Hot 100 chart songs’ lyrics are compared with those songs outside the chart but from the same artists to investigate if only focusing on textual features can lead to successfully predicting the presence of a song in a ranking representing the most popular songs. Secondly, in direct connection with the previous point we also want to address an analysis to predict the degree of the popularity and its rank range within the Billboard Hot 100 chart.

# Modelling Framework 

Three different algorithms were used for the classification task: Logistic Regression, Naive Bayes (Gaussian for binary classification and Multinomial for the multiclass one) and the Random Forest classifier. Another approach was tried, but not included in the final script, because the higher complexity of the model was not reflected in an improvement of the result metric. This involves an embedding through Word2Vec technique and the training of a neural network with two Bidirectional LSTM layers and a dense layer with softmax activation function.
For the presented models, a pipeline was built to have a more neat process. This includes the count vectorizer and tf-idf transformer - which is equivalent to the use of TfidfVectorizer from Sklearn (Pedregosa et al, 2011) and the model in question. For the GaussianNB a FunctionTransformer to transform the train set in a dense form was required before passing the model. On the other hand, the training of the RF is simply done in conjunction with a tf-idf vectorizer.
A Grid Search was carried out for all models but the Logistic Regressor, as the penalties are not fit for all the possible solvers and this would have caused the search to fail. In the case of the NB the parameters involved were the range of n_grams of the vectorizer, the norm and use_idf of the tf-idf transformer, while for the RF classifier the search concerned number of estimators, max features and evaluation criterion.

# Results

For the task 3 different algorithms were used, with the models showing very close results. For what concerns the classification between popular and unpopular songs, Random Forest has outperformed the other models, with an accuracy of 59% and f1-scores of 0.59 and 0.58 for the popular and not- popular class respectively. Nevertheless, results for the other classifiers do not show substantial differences, with the worse classifier - NB- having 51% accuracy and f1-score below 0.55 for both labels.
<img width="607" alt="Screenshot 2023-01-31 at 19 34 02" src="https://user-images.githubusercontent.com/91185911/215851305-010ba38f-96ba-40d0-ae61-e1ef67b0cb4c.png">
<img width="631" alt="Screenshot 2023-01-31 at 19 34 49" src="https://user-images.githubusercontent.com/91185911/215851486-89b4bfc5-2247-49b3-8228-d197324b9a72.png">

When trying to predict the level of popularity (3 brackets; 1 for spots 1-32, 2 for spots 34-66, 3 for spots 67-100) of a Billboard Hot 100 song, Logistic Regression and Random Forest are performing very similarly. They both have a 36% accuracy, but while the first model has f1-scores of 0.36, 0.37, 0.34, the other has f1-scores of 0.33, 0.36, 0.38 for class b_1, b_2, b_3 respectively. On the other hand, in this case the NB model is closer to the performance of the others compared to binary classification, as values are just under 1-2% lower of the best-performing models.


