import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
import nltk
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline

from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pickle

def load_data(database_filepath):
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('stopwords')
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('SELECT * FROM DisasterResponse', engine)
    X = df['message'].values
    Y = df.iloc[:, 4:39].values
    category_names = df.iloc[:, 4:39].columns.tolist()
    return X,Y, category_names


def tokenize(text):
    # normalization 
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ",text)
    
    # Tokenize text
    words = word_tokenize(text)
    
    # remove stop words
    words = [w for w in words if w not in stopwords.words('english')]
    
    # stemming and lemmatization
    words = [PorterStemmer().stem(w) for w in words]
    words = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    # remove leading/trailing whitespace
    words = [w.strip() for w in words]
    
    return words


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())) #   ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])

    parameters = {
        'clf__estimator__max_features': [1],
#         'clf__estimator__min_samples_leaf': [2, 3],
#         'clf__estimator__min_samples_split': [2, 3],
#       'clf__estimator__n_estimators': [100, 200]
    }

    cv = GridSearchCV(pipeline, param_grid = parameters) 

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    
    Y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        labels = np.unique(Y_test[:,i])
        confusion_mat = confusion_matrix(Y_test[:,i], Y_pred[:,i], labels=labels)
        accuracy = (Y_pred[:,i] == Y_test[:,i]).mean()

    print("Labels:", labels)
    print("Accuracy:", accuracy)
    print(i, classification_report(Y_test[:,i], Y_pred[:,i],labels=labels))  
    print("Confusion Matrix:\n", confusion_mat)
    
    


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()