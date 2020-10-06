# Twitter Sentiments
Machine learning project in the Natural Language Processing (NPL) field that trains a Naive Bayes model to classify hate speech in a tweet. We developed a flask application for this project.

### Dataset

The dataset used for training our models is composed of 31k plus labeled tweets. Tweets are labeled 0 (non-hate), and 1 (hate). Dataset was downloaded from kaggle.com (link below). After exploration of the files we renamed them modelData (for model building and training) and tweetsBank (for testing) to reflect their content and use in our project.

* [Twitter Sentiment Data Source](https://www.kaggle.com/arkhoshghalb/twitter-sentiment-analysis-hatred-speech)
  
## Flow Process:
Our chosen flow for rendering a classification model of a tweet is as follows:
1. User input (tweet )
2. Cleaning
3. Vectorizing 
4. Classifying 

### User Input 
For building our model we used the modelData dataset, we dropped duplicates and non a numbers. 
  
### Cleaning
We used a combinationn of regular expressions (re), natural language toolkit (nltk), and suplemented stop words.

#### Upsampling Minority Class
We upsampled our dataset for building our models. Our original dataset was imbalanced showing a 93% of tweets labeled non-hate and only a 7% of tweets labeled hate. We followed recommendations of the scikit-learn.org for this issue, and obtained our best result with upsampling the minority class method. Other resampling method is penalization of the classifier.

### Vectorizing
We decided to use a bag of words (BOW) for our final model. Although we used BOW feautures in our final flask app, we extracted feautures with both countvectorizer (BOW) and term frequency-inverse document frecuency (TfIdf). 

### Classifying
We created models with Naive Bayes (nb), Random Forest (rf), and support vector machine (SVM) classfiers.

## Results

We used F-1 scores for choosing our model performance. Scores of the models by vectorizer and by classifier are detailed in Table 1 below. All models before resampling scored around the 50% mark, which indicated there was no advantage of using the model since chances of obtaining a classification of hate or non-hate was the same. After re-sampling (up-sampling minority class for nb and rf, and penalizing for svm) scores showed improvement for the nv and rf models, and a decrease in performance for the svm model. Lower performance with SVM model was unexpected since its a popular classifier for sentiment analysis.
Although scores of BOW rf model was the highest, we decided to use the nb model for our app. This is because of the large file size produced by the rf model, which made it dificult to share and upload into our repository.

Table 1
 |Vectorizer|Classifier| F-1 Score Before Re-sampling| F-1 Score After Re-sampling|            
 |:---:|:---:|:---:|:---:|
 |BOW |Naive Bayes| 0.21|0.87|
 |Tf-Idf |Naive Bayes| 0.22|0.87|
 |BOW |Random Forest| 0.58|0.97|
 |Tf-Idf |Random Forest| 0.60|0.72|
 |BOW |SVM| 0.55|0.48|
 |Tf-Idf |SVM| 0.54|0.52|




