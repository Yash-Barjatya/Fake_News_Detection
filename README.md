# Fake_News_Detection

## Problem definition:
The problem is to develop a model that can accurately detect fake news articles from real ones.

## Approach:
There are different approaches one can take, such as supervised learning, unsupervised learning, or deep learning. For fake news detection, supervised learning is a common approach. In supervised learning, the model is trained on labeled data, where the true class of each article (real or fake) is known. The goal is to train a model that can accurately classify new, unseen articles as either real or fake.

## Data collection:
The next step is to collect data that you can use to train and evaluate your model. There are several sources of data you can use, such as online news sources, social media platforms, and news archives. You can use web scraping techniques to collect data from these sources or else simply download the data from the [dataset](/dataset) folder of this repository

## Data exploration:
After data preparation, the next step is to explore the data to gain insights into its characteristics. You can use various libraried like matplotlib ,seaborn ,wordcloud to explore the data and identify any patterns or trends.

## Data preparation and cleaning:
Once you have explored the data, you may identify some data quality issues that need to be addressed. So the next step is to prepare it for analysis. This involves cleaning and preprocessing the data, which includes tasks such as removing duplicates, handling missing values and correcting errors in the data.

## Feature engineering:
Feature engineering involves selecting and transforming the variables (features) in the data that will be used by the machine learning algorithm. 
<i>For the given dataset,i have used only the title and body of news article and obmitted the author column</i>

## Training model:
After feature engineering, the next step is to train the machine learning model. You can use various algorithms to train the model.
<i>I have used both logistic regression , decision trees classifier and PassiveAggressiveClassifier as models </i>

## Model building:
Once the model is trained, the next step is to build the final model. This involves tuning the model hyperparameters and selecting the best performing model.
<i>Out of the three tested model I have used PassiveAggressiveClassifier model as the final model to predict new data</i>

## Model evaluation:
After building the model, you need to evaluate its performance.
<i>I have used the accuracy metric.</i>

## Prediction:
Once you have evaluated the model, you can use it to predict the class of new, unseen articles.

## Getting started with the project:-

- Clone my repository.
- Open CMD in working directory.
- Run `pip install -r requirements.txt`
- Open project in any IDE(Pycharm or VSCode)
- Run `Fake_News_Detector.py`, go to the `http://127.0.0.1:5000/`
- If you want to build your model with the some changes, you can check the `Fake_News_Detection.ipynb`.
- You can check web app is working fine. Sometimes predictions may be wrong.

## Note
- This project is just for learning purpose, don't think, it can do work in real time, because model was trained on historic & limited data.
- For real time building of this kind of system, we need updated dataset and we need to build a model in particular interval of time, because news data can be updated in seconds, so our model should be also updated with the data.

## Disclaimer
- The prediction made by the model may be wrong at times and the author doesn't take gaurantee of the accuracy of prediction of this model on latest news