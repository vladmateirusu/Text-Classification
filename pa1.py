import numpy as np
import nltk
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

class TextPreprocessor:
    def __init__(self, strategy):
        self.strategy = strategy
        self.stemmer = PorterStemmer()
    
    def preprocess(self, text):
        if self.strategy == 'tokenize':
            tokens = word_tokenize(text)
            return ' '.join(tokens)
        elif self.strategy == 'lowercase':
            return text.lower()
        elif self.strategy == 'stem':
            tokens = word_tokenize(text.lower())
            return ' '.join([self.stemmer.stem(token) for token in tokens])
        
        return text
    
def load_data(file0, file1):
    with open(file0, 'r', encoding='utf-8') as f0:
        class0=[line.strip() for line in f0 if line.strip()]
    with open(file1, 'r', encoding='utf-8') as f1:
        class1=[line.strip() for line in f1 if line.strip()]

    labels0=[0]*len(class0)
    labels1=[1]*len(class1)

    X = class0 + class1
    Y= labels0 + labels1

    return np.array(X), np.array(Y)

def experiment(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

    preprocessing_strategies = ['tokenize', 'lowercase', 'stem']
    
    classifiers = {
        'logistic_reg': LogisticRegression(max_iter=1000, random_state=42),
        'perceptron': Perceptron(max_iter=1000, random_state=42)
    }

    feature_extractors = {
        'word_features': CountVectorizer(max_features=1000),
        'bigram_features': CountVectorizer(ngram_range=(1, 2), max_features=1000)
    }

    results = []