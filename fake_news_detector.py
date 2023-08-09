from flask import Flask, render_template, request
import pickle
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import string

app = Flask(__name__)

# Load model and vectorizer
loaded_model = pickle.load(open('model.pkl', 'rb'))
loaded_tfidfvect =  pickle.load(open('tfidfvect.pkl', 'rb'))
lemmatizer = WordNetLemmatizer()
stpwrds = list(stopwords.words('english'))

# def preprocessText(text):
#     corpus = []
#     review = text
#     review = re.sub(r'https?://\S+|www\.\S+', '', review)  # Remove URLs
#     review = re.sub(r'\W', ' ', review)  # Remove special characters
#     review = re.sub(r'\n', ' ', review)  # Replace newline characters with a space
#     review = re.sub(r'\w*\d\w*', '', review)  # Remove words containing digits
#     review = review.lower()  # Convert to lowercase
#     review = nltk.word_tokenize(review)  # Tokenize
#     for y in review :
#         if y not in stpwrds :
#             corpus.append(lemmatizer.lemmatize(y))
#     review = ' '.join(corpus)
#     text = review  
#     return text
def preprocessText(text):
    review = text[0]
    review = re.sub(r'\W', ' ', review)  # Remove special characters
    review = re.sub(r'\n', ' ', review)  # Replace newline characters with a space
    review = re.sub(r'\w*\d\w*', '', review)  # Remove words containing digits
    review = review.lower()  # Convert to lowercase
    review = re.sub(r'https?://\S+|www\.\S+', '', review)  # Remove URLs
    words = nltk.word_tokenize(review)  # Tokenize
    corpus = [lemmatizer.lemmatize(word) for word in words if word not in stpwrds]
    review = ' '.join(corpus)
    return review


# def fake_news_det(news):
#     input_data = [news]
#     input_data = preprocessText(input_data)
#     vectorized_input_data = loaded_tfidfvect.transform(input_data)
#     prediction = loaded_model.predict(vectorized_input_data)
#     return prediction
def fake_news_det(news):
    preprocessed_news = preprocessText(news)  # Preprocess the input news
    vectorized_input_data = loaded_tfidfvect.transform([preprocessed_news])  # Transform the preprocessed news
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = fake_news_det(message)
        return render_template('index.html', prediction=pred)
    else:
        return render_template('index.html', prediction="Something went wrong")

if __name__ == '__main__':
    app.run(debug=True)