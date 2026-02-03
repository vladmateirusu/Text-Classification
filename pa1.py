import numpy as np
import nltk
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import train_test_split, cross_val_score
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, log_loss
from scipy.stats import entropy
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC

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
        elif self.strategy == 'remove_stopwords':
            stop_words = set(stopwords.words('english'))
            tokens = word_tokenize(text.lower())
            return ' '.join([t for t in tokens if t.lower() not in stop_words])
        
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
    X, Y = load_data('synsem0.txt', 'synsem1.txt')

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,Y, test_size=0.2, random_state=42, stratify=Y
        )

    preprocessing_strategies = ['tokenize', 'lowercase', 'stem', 'lemma', 'remove_stopwords']
    
    feature_extractors = {
        'tfidf_100': TfidfVectorizer(max_features=100),
        'word_100': CountVectorizer(max_features=100),
        'bigram_100': CountVectorizer(ngram_range=(1, 2), max_features=100),
        'char_3_5_count': CountVectorizer(analyzer='char', ngram_range=(3, 5), max_features=200),
        'char_3_5_binary': CountVectorizer(analyzer='char', ngram_range=(3, 5), binary=True, max_features=200),
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
        
        'linearSVC_L2_C0.01': LinearSVC(penalty='l2', C=0.01, max_iter=1000, random_state=42, dual=True),
        'linearSVC_L2_C0.1': LinearSVC(penalty='l2', C=0.1, max_iter=1000, random_state=42, dual=True),
        'linearSVC_L2_C1.0': LinearSVC(penalty='l2', C=1.0, max_iter=1000, random_state=42, dual=True),
        'linearSVC_L2_C10': LinearSVC(penalty='l2', C=10.0, max_iter=1000, random_state=42, dual=True),
        'linearSVC_L2_C100': LinearSVC(penalty='l2', C=100.0, max_iter=1000, random_state=42, dual=True),
        'linearSVC_L1_C0.01': LinearSVC(penalty='l1', C=0.01, max_iter=1000, random_state=42, dual=False),
        'linearSVC_L1_C0.1': LinearSVC(penalty='l1', C=0.1, max_iter=1000, random_state=42, dual=False),
        'linearSVC_L1_C1.0': LinearSVC(penalty='l1', C=1.0, max_iter=1000, random_state=42, dual=False),
        'linearSVC_L1_C10': LinearSVC(penalty='l1', C=10.0, max_iter=1000, random_state=42, dual=False),
    }
    
    results = []
    
    
    for prep_strategy in preprocessing_strategies:
        preprocessor = TextPreprocessor(prep_strategy)
        
        X_train_prep = [preprocessor.preprocess(x) for x in X_train]
        X_test_prep = [preprocessor.preprocess(x) for x in X_test]
        
        for feat_name, vectorizer in feature_extractors.items():

            for clf_name, classifier in classifiers.items():

                #Pipeline prevents feature leakage
                pipeline = Pipeline([
                    ('vectorizer', vectorizer),
                    ('classifier', classifier)
                ])

                # Coss val
                cv_scores = cross_val_score(
                    pipeline,
                    X_train_prep,
                    Y_train,
                    cv=5
                )
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()

                # train
                pipeline.fit(X_train_prep, Y_train)

                # accuracy
                train_preds = pipeline.predict(X_train_prep)
                test_preds = pipeline.predict(X_test_prep)

                train_acc = accuracy_score(Y_train, train_preds)
                test_acc = accuracy_score(Y_test, test_preds)

                # log loss and entropy
                if hasattr(pipeline.named_steps['classifier'], "predict_proba"):
                    test_probs = pipeline.predict_proba(X_test_prep)
                    test_logloss = log_loss(Y_test, test_probs)
                    pred_entropies = [entropy(prob) for prob in test_probs]
                    mean_entropy = np.mean(pred_entropies)
                else:
                    test_logloss = None
                    mean_entropy = None


                result = {
                    'preprocessing': prep_strategy,
                    'features': feat_name,
                    'classifier': clf_name,
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'test_logloss': test_logloss,
                    'mean_entropy': mean_entropy
                }
                results.append(result)
                
                
                print(f"\nPreprocessing: {prep_strategy}")
                print(f"Classifier: {clf_name}")
                print(f"Features: {feat_name}")
                print(f"Train Acc: {train_acc:.4f}")
                print(f"Test Acc: {test_acc:.4f}")
                print(f"CV Mean: {cv_mean:.4f} (+/- {cv_std:.4f})")
                print(f"Test Log Loss: {test_logloss}")
                print(f"Mean Entropy: {mean_entropy}")
                print("\n")
                
    best_acc = max(results, key=lambda x: x['test_acc'])

    best_logloss = min(
        [r for r in results if r['test_logloss'] is not None],
        key=lambda x: x['test_logloss']
    )

    return results, best_acc, best_logloss

def experiment_morphphon(f):
    
    X, Y = load_data('morphphon0.txt', 'morphphon1.txt')
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )
    
    preprocessing_strategies = ['tokenize', 'lowercase', 'stem', 'lemma', 'remove_stopwords']
    
    feature_extractors = {
        'tfidf_100': TfidfVectorizer(max_features=100),
        'word_100': CountVectorizer(max_features=100),
        'bigram_100': CountVectorizer(ngram_range=(1, 2), max_features=100),
        'char_3_5_count': CountVectorizer(analyzer='char', ngram_range=(3, 5), max_features=200),
        'char_3_5_binary': CountVectorizer(analyzer='char', ngram_range=(3, 5), binary=True, max_features=200),
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
        
        'linearSVC_L2_C0.01': LinearSVC(penalty='l2', C=0.01, max_iter=1000, random_state=42, dual=True),
        'linearSVC_L2_C0.1': LinearSVC(penalty='l2', C=0.1, max_iter=1000, random_state=42, dual=True),
        'linearSVC_L2_C1.0': LinearSVC(penalty='l2', C=1.0, max_iter=1000, random_state=42, dual=True),
        'linearSVC_L2_C10': LinearSVC(penalty='l2', C=10.0, max_iter=1000, random_state=42, dual=True),
        'linearSVC_L2_C100': LinearSVC(penalty='l2', C=100.0, max_iter=1000, random_state=42, dual=True),
        'linearSVC_L1_C0.01': LinearSVC(penalty='l1', C=0.01, max_iter=1000, random_state=42, dual=False),
        'linearSVC_L1_C0.1': LinearSVC(penalty='l1', C=0.1, max_iter=1000, random_state=42, dual=False),
        'linearSVC_L1_C1.0': LinearSVC(penalty='l1', C=1.0, max_iter=1000, random_state=42, dual=False),
        'linearSVC_L1_C10': LinearSVC(penalty='l1', C=10.0, max_iter=1000, random_state=42, dual=False),
    }
    
    results = []
    
    for prep_strategy in preprocessing_strategies:
        preprocessor = TextPreprocessor(prep_strategy)
        
        X_train_prep = [preprocessor.preprocess(x) for x in X_train]
        X_test_prep = [preprocessor.preprocess(x) for x in X_test]
        
        for feat_name, vectorizer in feature_extractors.items():

            for clf_name, classifier in classifiers.items():

                #Pipeline prevents feature leakage
                pipeline = Pipeline([
                    ('vectorizer', vectorizer),
                    ('classifier', classifier)
                ])

                # Coss val
                cv_scores = cross_val_score(
                    pipeline,
                    X_train_prep,
                    Y_train,
                    cv=5
                )
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()

                # train
                pipeline.fit(X_train_prep, Y_train)

                # accuracy
                train_preds = pipeline.predict(X_train_prep)
                test_preds = pipeline.predict(X_test_prep)

                train_acc = accuracy_score(Y_train, train_preds)
                test_acc = accuracy_score(Y_test, test_preds)

                # log loss and entropy
                if hasattr(pipeline.named_steps['classifier'], "predict_proba"):
                    test_probs = pipeline.predict_proba(X_test_prep)
                    test_logloss = log_loss(Y_test, test_probs)
                    pred_entropies = [entropy(prob) for prob in test_probs]
                    mean_entropy = np.mean(pred_entropies)
                else:
                    test_logloss = None
                    mean_entropy = None


                result = {
                    'preprocessing': prep_strategy,
                    'features': feat_name,
                    'classifier': clf_name,
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'test_logloss': test_logloss,
                    'mean_entropy': mean_entropy
                }

                results.append(result)

                print(f"\nPreprocessing: {prep_strategy}")
                print(f"Classifier: {clf_name}")
                print(f"Features: {feat_name}")
                print(f"Train Acc: {train_acc:.4f}")
                print(f"Test Acc: {test_acc:.4f}")
                print(f"CV Mean: {cv_mean:.4f} (+/- {cv_std:.4f})")
                print(f"Test Log Loss: {test_logloss}")
                print(f"Mean Entropy: {mean_entropy}")
                print("\n")
    
    best_acc = max(results, key=lambda x: x['test_acc'])

    best_logloss = min(
     [r for r in results if r['test_logloss'] is not None],
     key=lambda x: x['test_logloss']
    )

    return results, best_acc, best_logloss



if __name__ == "__main__":
    with open('results.txt', 'w', encoding='utf-8') as f:
        f.write("Sentiment Analysis Results\n")
        f.write("=" * 50 + "\n\n")
        results, best_acc, best_logloss = experiment_sentiment(f)
        f.write("\nBest test accuracy for Sentiment Analysis:\n")
        f.write(f"  Preprocessing: {best_acc['preprocessing']}\n")
        f.write(f"  Features: {best_acc['features']}\n")
        f.write(f"  Classifier: {best_acc['classifier']}\n")
        f.write(f"  Train Accuracy: {best_acc['train_acc']:.4f}\n")
        f.write(f"  Test Accuracy: {best_acc['test_acc']:.4f}\n")
        if best_acc['test_logloss'] is None:
            f.write("  Test Log Loss: N/A\n")
        else:
            f.write(f"  Test Log Loss: {best_acc['test_logloss']:.4f}\n")
        f.write(f"  CV Mean: {best_acc['cv_mean']:.4f} (+/- {best_acc['cv_std']:.4f})\n")
        if best_acc['mean_entropy'] is not None:
            f.write(f"  Mean Entropy: {best_acc['mean_entropy']:.4f}\n\n")
        else:
            f.write("  Mean Entropy: N/A\n\n")

        f.write("\nBest test log loss for Sentiment Analysis:\n")
        f.write(f"  Preprocessing: {best_logloss['preprocessing']}\n")
        f.write(f"  Features: {best_logloss['features']}\n")
        f.write(f"  Classifier: {best_logloss['classifier']}\n")
        f.write(f"  Train Accuracy: {best_logloss['train_acc']:.4f}\n")
        f.write(f"  Test Accuracy: {best_logloss['test_acc']:.4f}\n")
        if best_logloss['test_logloss'] is None:
            f.write("  Test Log Loss: N/A\n")
        else:
            f.write(f"  Test Log Loss: {best_logloss['test_logloss']:.4f}\n")
        f.write(f"  CV Mean: {best_logloss['cv_mean']:.4f} (+/- {best_logloss['cv_std']:.4f})\n")
        if best_logloss['mean_entropy'] is not None:
            f.write(f"  Mean Entropy: {best_logloss['mean_entropy']:.4f}\n\n")
        else:
            f.write("  Mean Entropy: N/A\n\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("Alliteration Detection Results (3+ words with same starting letter)\n")
        f.write("=" * 70 + "\n\n")
        results_morphphon, best_morphphon, best_logloss_morphphon = experiment_morphphon(f)
        f.write("\nBest test accuracy for Alliteration Detection:\n")
        f.write(f"  Preprocessing: {best_morphphon['preprocessing']}\n")
        f.write(f"  Features: {best_morphphon['features']}\n")
        f.write(f"  Classifier: {best_morphphon['classifier']}\n")
        f.write(f"  Train Accuracy: {best_morphphon['train_acc']:.4f}\n")
        f.write(f"  Test Accuracy: {best_morphphon['test_acc']:.4f}\n")
        if best_morphphon['test_logloss'] is None:
            f.write("  Test Log Loss: N/A\n")
        else:
            f.write(f"  Test Log Loss: {best_morphphon['test_logloss']:.4f}\n")
        f.write(f"  CV Mean: {best_morphphon['cv_mean']:.4f} (+/- {best_morphphon['cv_std']:.4f})\n")
        if best_morphphon['mean_entropy'] is not None:
            f.write(f"  Mean Entropy: {best_morphphon['mean_entropy']:.4f}\n\n")
        else:
            f.write("  Mean Entropy: N/A\n\n")
        
        f.write("\nBest test log loss for Alliteration Detection:\n")
        f.write(f"  Preprocessing: {best_logloss_morphphon['preprocessing']}\n")
        f.write(f"  Features: {best_logloss_morphphon['features']}\n")
        f.write(f"  Classifier: {best_logloss_morphphon['classifier']}\n")
        f.write(f"  Train Accuracy: {best_logloss_morphphon['train_acc']:.4f}\n")
        f.write(f"  Test Accuracy: {best_logloss_morphphon['test_acc']:.4f}\n")
        if best_logloss_morphphon['test_logloss'] is None:
            f.write("  Test Log Loss: N/A\n")
        else:
            f.write(f"  Test Log Loss: {best_logloss_morphphon['test_logloss']:.4f}\n")
        f.write(f"  CV Mean: {best_logloss_morphphon['cv_mean']:.4f} (+/- {best_logloss_morphphon['cv_std']:.4f})\n")
        if best_logloss_morphphon['mean_entropy'] is not None:
            f.write(f"  Mean Entropy: {best_logloss_morphphon['mean_entropy']:.4f}\n\n")
        else:
            f.write("  Mean Entropy: N/A\n\n")