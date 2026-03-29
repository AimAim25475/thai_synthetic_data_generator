import pandas as pd
# import numpy as np
# import pickle as pk
from pathlib import Path
from pythainlp import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import os

cvec = None
lr = None

_BASE_DIR = Path(__file__).resolve().parents[1]

# model_path = 'models/text_classification.sav'

def text_process(text):

    # Word Tokenize
    final = word_tokenize(text)

    return " ".join(final)

# def load_model():
#     return pk.load(open(model_path, 'rb'))

def prepare_model():
    global cvec, lr
    qa = pd.read_csv(_BASE_DIR / 'CMSK' / 'train.csv')
    qa_df = qa.iloc[:,1:3]
    qa_df['label'] = 'qa_mode'

    chat_df = pd.read_csv(_BASE_DIR / 'CMSK' / 'chitchat_train_data.csv')
    chat_df.rename(columns={'Q':'question','A':'answer'}, inplace=True)
    chat_df['label'] = 'chat_mode'

    df = pd.concat([qa_df,chat_df], axis=0)

    extra_path = (os.getenv("ROUTING_EXTRA_TRAIN_CSV") or "").strip()
    if extra_path:
        try:
            extra_df = pd.read_csv(extra_path)
            # Expect columns: question,label
            if "question" in extra_df.columns and "label" in extra_df.columns:
                extra_df = extra_df[["question", "label"]].copy()
                df = pd.concat([df, extra_df], axis=0, ignore_index=True)
        except Exception:
            # Keep base behavior if extra file is missing/bad.
            pass

    df['text_token'] = df['question'].apply(text_process)

    # Split Train and Test
    X = df[['text_token']]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=25)

    cvec = CountVectorizer(analyzer=lambda x:x.split(' '))
    cvec.fit_transform(X_train['text_token'])

    # Create Bag-Of-Words (BoW)
    train_bow = cvec.transform(X_train['text_token'])

    lr = LogisticRegression()
    lr.fit(train_bow, y_train)

prepare_model()

def predict(text):
    global cvec, lr
    token = text_process(text)
    bow = cvec.transform(pd.Series([token]))
    pred = lr.predict(bow)
    return pred[0] if len(pred) > 0 else ""