import numpy as np
import nltk
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

class TextPreprocessor:
    def __init__(self, strategy):
        self.strategy = strategy
        self.lemmatizer = WordNetLemmatizer()
    
    def preprocess(self, text):
        if self.strategy == 'tokenize':
            tokens = word_tokenize(text)
            return ' '.join(tokens)

        elif self.strategy == 'lowercase':
            return text.lower()

        elif self.strategy == 'lemma':
            tokens = word_tokenize(text.lower())
            return ' '.join([self.lemmatizer.lemmatize(token) for token in tokens])

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

def experiment():
    X, Y = load_data('synsem0.txt', 'synsem1.txt')

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

    preprocessing_strategies = ['tokenize', 'lowercase', 'lemma']
    
    classifiers = {
        'logistic_reg': LogisticRegression(max_iter=1000, random_state=42),
        'perceptron': Perceptron(max_iter=1000, random_state=42)
    }

    feature_extractors = {
        'word_features': CountVectorizer(
            max_features=5000,
            stop_words=None 
            ),
        'bigram_features': CountVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            stop_words=None 
    )
}

    results = []
    
    for prep_strategy in preprocessing_strategies:
        preprocessor = TextPreprocessor(prep_strategy)
        
        X_train_prep = [preprocessor.preprocess(x) for x in X_train]
        X_test_prep = [preprocessor.preprocess(x) for x in X_test]
        
        for feat_name, vectorizer in feature_extractors.items():

            X_train_vec = vectorizer.fit_transform(X_train_prep)
            X_test_vec = vectorizer.transform(X_test_prep)
            
            for clf_name, classifier in classifiers.items():

                classifier.fit(X_train_vec, Y_train)
                
                train_acc = accuracy_score(Y_train, classifier.predict(X_train_vec))
                test_acc = accuracy_score(Y_test, classifier.predict(X_test_vec))
                
                result = {
                    'preprocessing': prep_strategy,
                    'features': feat_name,
                    'classifier': clf_name,
                    'train_acc': train_acc,
                    'test_acc': test_acc
                }
                results.append(result)
                
                print(f"\nPreprocessing: {prep_strategy:10s} | Features: {feat_name:15s} | Classifier: {clf_name:15s}")
                print(f"  Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
    
    best_result = max(results, key=lambda x: x['test_acc'])
    

