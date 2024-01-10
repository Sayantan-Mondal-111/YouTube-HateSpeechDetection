from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np
from hate_prediction import preprocess
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer # initialises an array of frequencies of words.
from hate_prediction import vectorizer
app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/',methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        stop_words = stopwords.words('english')
        comment = request.form['comment']    

        dfnew = pd.DataFrame([comment], columns=['Comment'])

        res= preprocess(dfnew['Comment']).to_numpy()

        comment1_vect = vectorizer.transform(res)

        result_prob = model.predict_proba(comment1_vect)
        output='{0:.{1}f}'.format(result_prob[0][1], 2)
    

        if output<str(0.3):
            return render_template('index.html', comment='Text: {}'.format(comment), pred='Result: NOT A HATE comment.\n Probability of hate is {}.'.format(output))
        else:
            return render_template('index.html', comment='Text: {}'.format(comment), pred='Result: HATE comment.\n Probability of hate is {}.'.format(output))

    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)