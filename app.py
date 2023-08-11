from flask import Flask, render_template, request
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
app = Flask(__name__)

# Load model and vectorizer
loaded_model = pickle.load(open('models/model.pkl', 'rb'))
loaded_tfidfvect = pickle.load(open('models/tfidfvect.pkl', 'rb'))
lemmatizer = WordNetLemmatizer()
stpwrds = list(stopwords.words('english'))


def preprocessText(text):
    review = text[0]
    review = re.sub(r'\W', ' ', review)  # Remove special characters
    # Replace newline characters with a space
    review = re.sub(r'\n', ' ', review)
    review = re.sub(r'\w*\d\w*', '', review)  # Remove words containing digits
    review = review.lower()  # Convert to lowercase
    review = re.sub(r'https?://\S+|www\.\S+', '', review)  # Remove URLs
    words = nltk.word_tokenize(review)  # Tokenize
    corpus = [lemmatizer.lemmatize(word)
              for word in words if word not in stpwrds]
    review = ' '.join(corpus)
    return review


def fake_news_det(news):
    # print('news', news)
    # Preprocess the input news
    # preprocessed_news = preprocessText(news)
    preprocessed_news = news
    vectorized_input_data = loaded_tfidfvect.transform(
        [preprocessed_news])  # Transform the preprocessed news
    prediction = loaded_model.predict(vectorized_input_data)
    # print('prediction', prediction[0])
    return prediction[0]


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
