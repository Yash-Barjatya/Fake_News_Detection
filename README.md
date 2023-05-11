# Fake_News_Detection

## Problem definition:
The problem is to develop a model that can accurately detect fake news articles from real ones.

## Approach:
There are different approaches one can take, such as supervised learning, unsupervised learning, or deep learning. For fake news detection, supervised learning is a common approach. In supervised learning, the model is trained on labeled data, where the true class of each article (real or fake) is known. The goal is to train a model that can accurately classify new, unseen articles as either real or fake.

## Data collection:
The next step is to collect data that you can use to train and evaluate your model. There are several sources of data you can use, such as online news sources, social media platforms, and news archives. You can use web scraping techniques to collect data from these sources or else simply download the data from the [dataset](/dataset) folder of this repository

## Data preparation:
Once you have collected the data, the next step is to prepare it for analysis. This involves cleaning and preprocessing the data, which includes tasks such as removing duplicates, handling missing values, and converting the data into a format that can be used by the machine learning algorithms.

## Data exploration:
After data preparation, the next step is to explore the data to gain insights into its characteristics. You can use various techniques such as visualization and statistical analysis to explore the data and identify any patterns or trends.

## Data preparation and cleaning:
Once you have explored the data, you may identify some data quality issues that need to be addressed. So the next step is to prepare it for analysis. This involves cleaning and preprocessing the data, which includes tasks such as removing duplicates, handling missing values and correcting errors in the data.

## Feature engineering:
Feature engineering involves selecting and transforming the variables (features) in the data that will be used by the machine learning algorithm. For fake news detection, common features include the length of the article, the number of unique words, and the presence of certain keywords or phrases.

## Training model:
After feature engineering, the next step is to train the machine learning model. You can use various algorithms such as logistic regression, decision trees, or random forests to train the model.(I have used both logistic regression and decision trees classifier as model)

## Model building:
Once the model is trained, the next step is to build the final model. This involves tuning the model hyperparameters and selecting the best performing model.

## Model evaluation:
After building the model, you need to evaluate its performance. This involves using metrics such as accuracy, precision, recall, and F1-score to assess the performance of the model.

## Prediction:
Once you have evaluated the model, you can use it to predict the class of new, unseen articles.
