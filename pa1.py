import numpy as np
import nltk
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

class TextPreprocessor:
    def __init__(self, strategy):
        self.strategy = strategy
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
    
    def preprocess(self, text):
        if self.strategy == 'tokenize':
            tokens = word_tokenize(text)
            return ' '.join(tokens)
        elif self.strategy == 'lowercase':
            return text.lower()
        elif self.strategy == 'stem':
            tokens = word_tokenize(text.lower())
            return ' '.join([self.stemmer.stem(token) for token in tokens])
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

def experiment_sentiment(f):
    X, Y = load_data('synsem0_file.txt', 'synsem1_file.txt')

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

    preprocessing_strategies = ['tokenize', 'lowercase', 'stem']
    
    feature_extractors = {
        'word_50': CountVectorizer(max_features=50),
        'word_100': CountVectorizer(max_features=100),
        'bigram_50': CountVectorizer(ngram_range=(1, 2), max_features=50),
        'bigram_100': CountVectorizer(ngram_range=(1, 2), max_features=100),
        'word_100_binary': CountVectorizer(max_features=100, binary=True),
        'word_100_stop': CountVectorizer(max_features=100, stop_words='english'),
        'word_100_min2': CountVectorizer( max_features=100, min_df=2, max_df=0.9 ),
        'char_3_5_200': CountVectorizer( analyzer='char', ngram_range=(3, 5), max_features=200 )
    }
    classifiers = {
        'logistic_L2_C0.01': LogisticRegression(penalty='l2', C=0.01, max_iter=500, random_state=42),
        'logistic_L2_C0.1': LogisticRegression(penalty='l2', C=0.1, max_iter=500, random_state=42),
        'logistic_L2_C1.0': LogisticRegression(penalty='l2', C=1.0, max_iter=500, random_state=42),
        'logistic_L2_C10': LogisticRegression( penalty='l2', C=10.0, max_iter=500, random_state=42 ),
        'logistic_L1_C0.01': LogisticRegression(penalty='l1', solver='liblinear', C=0.01, max_iter=500, random_state=42),
        'logistic_L1_C0.10': LogisticRegression(penalty='l1', solver='liblinear', C=0.1, max_iter=500, random_state=42),
        'logistic_L1_C1.0': LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=500, random_state=42),
        'logistic_L1_C10': LogisticRegression(penalty='l1', solver='liblinear', C=10.0, max_iter=500, random_state=42),
        'perceptron_L2': Perceptron(penalty='l2', alpha=0.0001, max_iter=500, random_state=42),
        'perceptron_L1': Perceptron(penalty='l1', alpha=0.0001, max_iter=500, random_state=42),
        'perceptron_none': Perceptron(penalty=None, max_iter=500, random_state=42),
        'perceptron_ES': Perceptron( penalty='l2', alpha=0.0001, max_iter=1000, early_stopping=True, validation_fraction=0.1, random_state=42 ),
        'perceptron_eta0_0.01': Perceptron( eta0=0.01, max_iter=100, random_state=42 ),
        'perceptron_eta0_0.1': Perceptron( eta0=0.1, max_iter=100, random_state=42 ),
        'perceptron_eta0_1.0': Perceptron( eta0=1.0, max_iter=100, random_state=42 ),
        'perceptron_iter_10': Perceptron( max_iter=10, random_state=42 ),
        'perceptron_iter_50': Perceptron( max_iter=50, random_state=42 ),
        'perceptron_iter_500': Perceptron( max_iter=500, random_state=42 ),
        'perceptron_L2_alpha1e-5': Perceptron( penalty='l2', alpha=1e-5, max_iter=100, random_state=42 ),
        'perceptron_L2_alpha1e-3': Perceptron( penalty='l2', alpha=1e-3, max_iter=100, random_state=42 ),
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
                
                
                print(f"\nPreprocessing: {prep_strategy}")
                print(f"Classifier: {clf_name}")
                print(f"Features: {feat_name}")
                print(f"Train Acc: {train_acc}")
                print(f"Test Acc: {test_acc}")
                print("\n")
                
                f.write(f"Preprocessing: {prep_strategy}\n")
                f.write(f"Classifier: {clf_name}\n")
                f.write(f"Features: {feat_name}\n")
                f.write(f"Train Acc: {train_acc}\n")
                f.write(f"Test Acc: {test_acc}\n\n")
    
    best_result = max(results, key=lambda x: x['test_acc'])
    
    return results, best_result

def experiment_morphphon(f):
    
    X, Y = load_data('morphphon0_file.txt', 'morphphon1_file.txt')
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )
    
    preprocessing_strategies = ['tokenize', 'lowercase', 'stem']
    
    feature_extractors = {
        'word_50': CountVectorizer(max_features=50),
        'word_100': CountVectorizer(max_features=100),
        'bigram_50': CountVectorizer(ngram_range=(1, 2), max_features=50),
        'bigram_100': CountVectorizer(ngram_range=(1, 2), max_features=100),
        'word_100_binary': CountVectorizer(max_features=100, binary=True),
        'word_100_stop': CountVectorizer(max_features=100, stop_words='english'),
        'word_100_min2': CountVectorizer( max_features=100, min_df=2, max_df=0.9 ),
        'char_3_5_200': CountVectorizer( analyzer='char', ngram_range=(3, 5), max_features=200 )
    }
    classifiers = {
        'logistic_L2_C0.01': LogisticRegression(penalty='l2', C=0.01, max_iter=500, random_state=42),
        'logistic_L2_C0.1': LogisticRegression(penalty='l2', C=0.1, max_iter=500, random_state=42),
        'logistic_L2_C1.0': LogisticRegression(penalty='l2', C=1.0, max_iter=500, random_state=42),
        'logistic_L2_C10': LogisticRegression( penalty='l2', C=10.0, max_iter=500, random_state=42 ),
        'logistic_L1_C0.01': LogisticRegression(penalty='l1', solver='liblinear', C=0.01, max_iter=500, random_state=42),
        'logistic_L1_C0.10': LogisticRegression(penalty='l1', solver='liblinear', C=0.1, max_iter=500, random_state=42),
        'logistic_L1_C1.0': LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=500, random_state=42),
        'logistic_L1_C10': LogisticRegression(penalty='l1', solver='liblinear', C=10.0, max_iter=500, random_state=42),
        'perceptron_L2': Perceptron(penalty='l2', alpha=0.0001, max_iter=500, random_state=42),
        'perceptron_L1': Perceptron(penalty='l1', alpha=0.0001, max_iter=500, random_state=42),
        'perceptron_none': Perceptron(penalty=None, max_iter=500, random_state=42),
        'perceptron_ES': Perceptron( penalty='l2', alpha=0.0001, max_iter=1000, early_stopping=True, validation_fraction=0.1, random_state=42 ),
        'perceptron_eta0_0.01': Perceptron( eta0=0.01, max_iter=100, random_state=42 ),
        'perceptron_eta0_0.1': Perceptron( eta0=0.1, max_iter=100, random_state=42 ),
        'perceptron_eta0_1.0': Perceptron( eta0=1.0, max_iter=100, random_state=42 ),
        'perceptron_iter_10': Perceptron( max_iter=10, random_state=42 ),
        'perceptron_iter_50': Perceptron( max_iter=50, random_state=42 ),
        'perceptron_iter_500': Perceptron( max_iter=500, random_state=42 ),
        'perceptron_L2_alpha1e-5': Perceptron( penalty='l2', alpha=1e-5, max_iter=100, random_state=42 ),
        'perceptron_L2_alpha1e-3': Perceptron( penalty='l2', alpha=1e-3, max_iter=100, random_state=42 ),
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

                print(f"\nPreprocessing: {prep_strategy}")
                print(f"Classifier: {clf_name}")
                print(f"Features: {feat_name}")
                print(f"Train Acc: {train_acc}")
                print(f"Test Acc: {test_acc}")
                print("\n")
                
                f.write(f"Preprocessing: {prep_strategy}\n")
                f.write(f"Classifier: {clf_name}\n")
                f.write(f"Features: {feat_name}\n")
                f.write(f"Train Acc: {train_acc}\n")
                f.write(f"Test Acc: {test_acc}\n\n")
    
    best_result = max(results, key=lambda x: x['test_acc'])
    
    return results, best_result


if __name__ == "__main__":
    with open('results.txt', 'w', encoding='utf-8') as f:
        f.write("Sentiment Analysis Results\n")
        f.write("=" * 50 + "\n\n")
        results, best = experiment_sentiment(f)
        f.write("\nBest config for Sentiment Analysis:\n")
        f.write(f"  Preprocessing: {best['preprocessing']}\n")
        f.write(f"  Features: {best['features']}\n")
        f.write(f"  Classifier: {best['classifier']}\n")
        f.write(f"  Test Accuracy: {best['test_acc']}\n\n")
        
        f.write("Morphophonology Results\n")
        f.write("=" * 50 + "\n\n")
        results_morphphon, best_morphphon = experiment_morphphon(f)
        f.write("\nBest config for Morphophonology:\n")
        f.write(f"  Preprocessing: {best_morphphon['preprocessing']}\n")
        f.write(f"  Features: {best_morphphon['features']}\n")
        f.write(f"  Classifier: {best_morphphon['classifier']}\n")
        f.write(f"  Test Accuracy: {best_morphphon['test_acc']}\n")
    
    print("Results have been written to results.txt")
