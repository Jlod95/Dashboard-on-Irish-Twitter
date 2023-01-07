import pandas as pd
import snscrape.modules.twitter as sntwitter
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
import tqdm
from tqdm import tqdm

# create dataframe containing date, user and tweet
queries = ["Fianna Fail min_faves:150 until:2022-12-31 since:2020-01-01",
           "Fine Gael min_faves:150 until:2022-12-31 since:2020-01-01",
           "Sinn Fein min_faves:150 until:2022-12-31 since:2020-01-01"]
tweets = []

for query in queries:
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        tweets.append([tweet.date, tweet.username, tweet.content, " ".join(query.split()[:2])])

df_tweets = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet', "Party"])

# group by month of the year
df_tweets['Month'] = df_tweets['Date'].dt.strftime('%Y-%m')

# create model
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# create polarity function that passes through the model
def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores).tolist()
    return scores

# create 3 columns for our scores
df_tweets = df_tweets.assign(Negative = None, Neutral = None, Positive = None)

for i, row in tqdm(df_tweets.iterrows(), total=len(df_tweets)):
    try:
      text = row['Tweet']
      scores = polarity_scores_roberta(text)
      df_tweets.loc[i, 'Negative'] = scores[0]
      df_tweets.loc[i, 'Neutral'] = scores[1]
      df_tweets.loc[i, 'Positive'] = scores[2]
    except RuntimeError:
        print(f'Broke for id {i}')

df_tweets = df_tweets.astype({'Negative':'float','Neutral':'float','Positive':'float'}) # convert them to floats

# find average by month for each scores
df_month = df_tweets.groupby(['Month', 'Party']).mean()
df_month = df_month.pivot_table(index = 'Month',
                                columns = 'Party',
                                values = ['Negative', 'Neutral', 'Positive'])
df_month.columns = [' '.join(col) for col in df_month.columns.values]

import pandas as pd
import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, Output, Input, dash_table, html


# Create the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

dff = df_tweets[['Tweet', 'Party', 'Negative', 'Neutral', 'Positive', 'Month']]

# Create a checklist of options using the DataFrame column names
checklist = html.Div([
    html.Div(
    dcc.Checklist(
        options=['Negative Fianna Fail', 'Negative Fine Gael', 'Negative Sinn Fein',
                 'Neutral Fianna Fail', 'Neutral Fine Gael', 'Neutral Sinn Fein',
                 'Positive Fianna Fail', 'Positive Fine Gael', 'Positive Sinn Fein'],
        value=['Negative Fianna Fail', 'Negative Fine Gael', 'Negative Sinn Fein'],
        id='checklist',
        style={'backgroundColor': '#0a244d',
               'borderRadius': '15px',
               'alignItems': 'center','justify-content': 'center'}
    ),
        style={'width' : '200px'}
    )
])
# Create the dropdown
dropdown = html.Div([
    html.Label('Select a Month:',style={'display': 'flex', 'alignItems': 'center'}),
    html.Div(dcc.Dropdown(id='status-dropdown',
                          options=[{'label': html.Span(value,
                                                       style = {'color' : '#0a244d', 'plot_bgcolor' : '#0a244d'}), 'value': value} for value in dff['Month'].unique()],
                          value='2020-01',
                          style={'width':'200px'})
             )
], style={'display': 'flex', 'justify-content': 'center'})

# Create a Graph component to display the data
graph = dcc.Graph(id='graph')

# Create the table
table = dash_table.DataTable(
    id='table',
    data=dff.to_dict('records'),
    columns=[{'id': c, 'name': c} for c in dff.columns],
    style_cell={'backgroundColor': '#0a244d', 'font_color' : 'white'},
    style_data ={'color' : 'white',
                 'textAlign' : "left",
                 'whiteSpace': 'normal',
                 'height': 'auto'},
    style_header={'textAlign': 'center'}
)

# Create layout of app
app.layout = html.Div([
    dbc.Row(html.H1('Public Perception of Party', style={'text-align': 'center', 'fontSize' : 24}),style={'padding-top': 20}),
    dbc.Row([
        dbc.Col(checklist, width=2, style={'display': 'flex', 'alignItems': 'center'}),
        dbc.Col(graph, width=10)
    ], style={'padding': 0}),
    dbc.Row(dbc.Col(dropdown)),
    dbc.Row(dbc.Col(table))
    ])

# Define a callback function that updates the graph
@app.callback(Output('graph', 'figure'),
              [Input('checklist', 'value')])

def update_graph(selected_columns):
  df_selected = df_month[selected_columns]
    # Generate a figure object with the selected data
  figure = {
      'data': [{'x': df_selected.index, 'y': df_selected[col], 'name': col} for col in df_selected.columns],
      'layout': {'xaxis':{'color' : 'white'}, 'yaxis' : {'color' : 'white'},
                 'legend': {'font' : {'color' : 'white'}},
                 'paper_bgcolor' : '#222222',
                 'plot_bgcolor' : '#0a244d'}
  }
  return figure

# Define the callback function
@app.callback(Output('table', 'data'),
              [Input('status-dropdown', 'value')])

def update_table(status):
    filtered_df = dff[dff['Month'] == status]
    # Return the filtered DataFrame as a list of dictionaries
    return filtered_df.to_dict('records')

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

