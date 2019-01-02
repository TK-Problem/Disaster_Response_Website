import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
from sklearn.externals import joblib
from sklearn.metrics import f1_score
from sqlalchemy import create_engine


app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals

    # data for visualizing distribution of message genres
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # data for visualizing message type counts
    column_names = df.columns.values[4:]
    msg_counts = [df[col].sum() for col in column_names]
    # remove "_" sign from column names
    column_names = [col.replace("_", " ") for col in column_names]

    # data for visualizing f1-scores. f-1 scores were manually calculated in separate notebook to speed up loading
    f1_scores = [0.9440867008480549, 0.82632495645683002, 0.51572327044025157, 0.86649227110582649,0.6937914406268838,
                 0.68376911692155895, 0.5807067812798471,  0.53881278538812782, 0.68460388639760839, 0.0,
                 0.81836596893990554, 0.85008485762775787, 0.79045226130653268, 0.63455149501661123,
                 0.65126512651265134, 0.51105651105651106, 0.67723342939481268, 0.71466106148187081,
                 0.68756916267060131, 0.62208151958844482, 0.67961165048543681, 0.72727272727272729,
                 0.65586034912718194, 0.46153846153846156, 0.50526315789473686, 0.43137254901960792,
                 0.45925925925925926, 0.60595238095238091, 0.87656013274655509, 0.79565807327001359,
                 0.84541601606067363, 0.54123711340206193, 0.88956654323676754, 0.68932038834951459,
                 0.64872657376261411, 0.80427283854456433]
    
    # create visuals
    graphs = [
       {
            'data': [
                Bar(
                    x=column_names,
                    y=f1_scores,
                    marker=dict(color='rgba(55, 128, 191, 0.7)',
                                line=dict(color='rgba(80, 80, 92, 1.0)', width=2))
                )
            ],

            'layout': {
                'title': 'F1-scores for different message types',
                'yaxis': {
                    'title': "F1-score"
                },
                'margin': dict(b=150),
                'xaxis': {
                    'title': None
                }
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    marker=dict(color='rgba(55, 128, 191, 0.7)',
                                line=dict(color='rgba(80, 80, 92, 1.0)', width=2))
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'margin': dict(b=150),
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
         {
            'data': [
                Bar(
                    x=column_names,
                    y=msg_counts,
                    marker=dict(color='rgba(55, 128, 191, 0.7)',
                                line=dict(color='rgba(80, 80, 92, 1.0)', width=2))
                )
            ],

            'layout': {
                'title': 'Distribution of Message Types',
                'height': 600,
                'yaxis': {
                    'title': "Count"
                },
                'margin': dict(b=150),
                'xaxis': {
                    'title': None
                }
            }
        },
        {
            'data': [
                Pie(
                    labels=column_names,
                    values=msg_counts,
                    hoverinfo="label+percent+name",
                    textinfo='none'
                )
            ],

            'layout': {
                'title': 'Distribution of Message Types',
                'height': 875
            }
        },
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()