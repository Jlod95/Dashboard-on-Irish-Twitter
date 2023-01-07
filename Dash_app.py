
import pandas as pd
import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, Output, Input, dash_table, html

df_tweets = pd.read_csv('df_tweets.csv')
df_month = pd.read_csv('df_month.csv')

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

