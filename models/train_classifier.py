import sys
import pandas as pd
import nltk

nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_data(database_filepath):
    """
    Load SQL database file as pandas DataFrame

    :param database_filepath: str, path to SQL database file
    :return: X: pandas Dataframe for training features [message text and message genre]
             y: pandas DataFrame with category messages
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM messages", engine)
    X = df[["message", "genre"]]
    y = df.iloc[:,4:]

    return X, y


def tokenize(text):
    """
    :param text: str, text sms message
    :return: remove punctuation, lower case all letters and tokenize message
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]

    return clean_tokens


### Helper functions for building sklearn pipeline

def get_message_col(X):
    """
    Return only message column
    """
    return X["message"]


def get_genre_col(X):
    """
    Return only genre column
    """
    return X["genre"]


def get_dummies(X):
    """
    one-hot encode the categorical variables
    """
    return pd.get_dummies(X)

###


def build_model():
    """
    Return sklearn pipeline with Random Forrest Classifier
    """

    # Take message collumn and tokenize it
    text_pipeline = Pipeline([
        ('get_text', FunctionTransformer(get_message_col, validate=False)),
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer())
    ])

    # one-hot-encode "genre" column
    genre_pipeline = Pipeline([
        ('get_genre', FunctionTransformer(get_genre_col, validate=False)),
        ('get_dummies', FunctionTransformer(get_dummies, validate=False))
    ])

    # join both transformations into single DataFrame
    features = FeatureUnion([
        ('text_pipeline', text_pipeline),
        ('genre_pipeline', genre_pipeline)
    ])

    # join data transformation with Multi Output Random Forest Classifier
    pipeline = Pipeline([
        ('features', features),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=5, min_samples_split=6), n_jobs=-1))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test):
    """
    :param model: sklearn trained model
    :param X_test: pandas test DataFrame
    :param Y_test: test data labels

    print out accuracy, precision, recall and fą scores for all message categories
    """

    # predict on test data
    Y_pred = model.predict(X_test)
    for idx in range(Y_test.columns.shape[0]):
        test = Y_test[Y_test.columns[idx]].values
        pred = Y_pred[:,idx]

        # calculate different scores
        accu = accuracy_score(test, pred)
        prec = precision_score(test, pred)
        reca = recall_score(test, pred)
        f1_s = f1_score(test, pred)
        # print results
        print(f"{Y_test.columns[idx]:22s} | accuracy: {accu:.2f} | precision: {prec:.2f} | recall: {reca:.2f} | f1 score: {f1_s:.2f}")


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

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