import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

df = pd.read_excel("_sentiment analysis.xlsx")

import nltk
nltk.download('stopwords')

stop_words = stopwords.words("english")
ps = PorterStemmer()

def preprocessing(text):
    text = text.lower()
    token = nltk.word_tokenize(text)
    new = []
    for i in token:
      if i not in stop_words and i not in string.punctuation:
        stemmer = ps.stem(i)
        new.append(stemmer)
    return " ".join(new)

df["clean"] = df["Sentence"].apply(preprocessing)

df["clean"]

from sklearn.feature_extraction.text import TfidfVectorizer

vector = TfidfVectorizer()

X = vector.fit_transform(df["clean"])
y = df["Sentiment"]

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 55)

log = LogisticRegression()
log.fit(X_train,y_train)


def preprocess_and_vectorize(text):
    text = text.lower()
    token = nltk.word_tokenize(text)
    new = []
    for i in token:
      if i not in stop_words and i not in string.punctuation:
        stemmer = ps.stem(i)
        new.append(stemmer)
    return " ".join(new)
    processed_data = df["text"].apply(preprocessing)
    def vectorize(processed_data):
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(processed_data)


def pred_text(text):
    text = text.lower()
    text = preprocessing(text)
    vec = vector.transform([text])
    predict = log.predict(vec)
    return "positive " if predict ==1 else "negative"


c = "happy"
pred_text(c)