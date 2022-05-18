import html
import random
import warnings
from flask import Flask, render_template, request, session
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
import gunicorn


warnings.filterwarnings('ignore')
app = Flask(__name__)
app.secret_key = 'My_Secret_Key'


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')


@app.route('/finalize1', methods=['GET', 'POST'])
def finalize1():
    message = request.form['message']
    session['message'] = message

    return render_template('finalize.html', msg=message)


@app.route('/finalize2', methods=['GET', 'POST'])
def finalize2():
    df = pd.read_csv('spam.csv', encoding='latin-1')
    message = df['text'][random.randint(0, 5571)]
    session['message'] = message
    return render_template('finalize.html', msg=message)


@app.route('/predict', methods=['POST'])
def predict():
    model = pickle.load(open('spam_model.pkl', 'rb'))
    tfidf_model = pickle.load(open('tfidf_model.pkl', 'rb'))
    if request.method == 'POST':
        message = session.get('message')
        msg = message
        msg = html.unescape(msg)
        msg = msg.replace('&gt;', '>').replace('&lt;', '<')
        if message == "":
            return render_template('predict.html', prediction=0, probaScam="0%", msg=msg)
        message = [message]
        dataset = {'message': message}
        data = pd.DataFrame(dataset)

        stop_words = set(stopwords.words('english'))
        data['message'] = data['message'].str.replace(r'[^\w\s]+', '', regex=True)
        data['message'] = data['message'].apply(
            lambda x: ' '.join(term for term in x.split() if term not in stop_words))

        w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
        lemmatizer = nltk.stem.WordNetLemmatizer()

        def lemmatize_text(text):
            return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

        data['message'] = data['message'].apply(lemmatize_text)
        data['message'] = data['message'].apply(lambda x: " ".join(x))

        ss = nltk.SnowballStemmer("english")
        data['message'] = data['message'].apply(lambda x: ' '.join(ss.stem(term) for term in x.split()))

        tfidf_vec = tfidf_model.transform(data["message"])
        tfidf_data = pd.DataFrame(tfidf_vec.toarray())
        my_prediction = model.predict(tfidf_data)
        probaScam = model.predict_proba(tfidf_data)[:, 1]
        probaScam = probaScam[0]
        probaScam = str(probaScam * 100)
        probaScam = probaScam[:4]
        probaScam = (probaScam + '%')

    return render_template('predict.html', prediction=my_prediction, probaScam=probaScam, msg=msg)


if __name__ == '__main__':
    app.run(debug=True)
