from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from string import punctuation
import re
from nltk.corpus import stopwords
import sqlite3
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

nltk.download('stopwords')

set(stopwords.words('english'))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')

@app.route("/signup")
def signup():

    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
    con.commit()
    con.close()
    return render_template("signin.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == 'admin' and password1 == 'admin':
        return render_template("form.html")

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("form.html")
    else:
        return render_template("signup.html")



@app.route('/notebook')
def notebook():
	return render_template('notebook.html')

@app.route('/notebook1')
def notebook1():
	return render_template('NOtebook.html')


@app.route('/form')
def form():
    return render_template('form.html')

model=pickle.load(open('model.pkl','rb'))


@app.route('/predict',  methods=['GET','POST'])
def predict():
    stop_words = stopwords.words('english')

    df = pd.read_csv('data/deceptive-opinion.csv')
    df1 = df[['deceptive', 'text']]
    df1.loc[df1['deceptive'] == 'deceptive', 'deceptive'] = 0
    df1.loc[df1['deceptive'] == 'truthful', 'deceptive'] = 1
    X = df1['text']
    Y = np.asarray(df1['deceptive'], dtype = int)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=109)
    cv = CountVectorizer()
    x = cv.fit_transform(X_train)
    y = cv.transform(X_test)

    
    #convert to lowercase
    text1 = request.form['text1'].lower()

    data = [text1]
    vect = cv.transform(data).toarray()
    pred = model.predict(vect)
    
    if pred == 0:
        return render_template('result.html', prediction_text=pred)

    elif pred ==1:
        text_final = ''.join(c for c in text1 if not c.isdigit())
        
        #remove punctuations
        #text3 = ''.join(c for c in text2 if c not in punctuation)
            
        #remove stopwords    
        processed_doc1 = ' '.join([word for word in text_final.split() if word not in stop_words])

        sa = SentimentIntensityAnalyzer()
        dd = sa.polarity_scores(text=processed_doc1)
        compound = round((1 + dd['compound'])/2, 2)

        return render_template('form.html', prediction_text=pred, final=compound, text1=text_final,text2=dd['pos'],text5=dd['neg'],text4=compound,text3=dd['neu'])

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000, threaded=True)
