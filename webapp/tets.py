


import dash
import dash_bootstrap_components as dbc
from dash_bootstrap_components._components.Col import Col
from dash_bootstrap_components._components.Row import Row
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import dash_table
import base64
import datetime
import io
from flask_caching import Cache
import pandas as pd
import uuid
import time
import os

from app import app, filecache_dir, cache
#from pages import page1, page2, methods



upload_data = dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select File')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    )

# slider = html.Div([
#     html.Label("Noise: ", style={'display':'inline-block'}),
#     dcc.Slider(id='slider', min=0, max=10, step=1, value=0),
# ])

# *** Top-level app layout ***

def serve_layout():
    print('Calling serve_layout')
    session_id = str(uuid.uuid4())
    layout =  html.Div(children=[

        html.Div(session_id, id='session-id'), #style={'display': 'none'}),
        html.Div(id='filecache_marker', style={'display': 'none'}),

        upload_data,

        #html.Div(id='data-table-div'),

        #slider,

        #dcc.Graph(id='two-column-graph'),

        # needed to load relevant CSS/JS
        #html.Div(dt.DataTable(rows=[{}]),style={'display': 'none'})
    ])
    return layout

app.layout = serve_layout


def parse_table(contents, filename):
    '''
    Parse uploaded tabular file and return dataframe.
    '''
    print('Calling parse_table')
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        raise
    print('Read dataframe:')
    print(df.columns)
    print(df)
    return df


def write_dataframe(session_id, df):
    '''
    Write dataframe to disk, for now just as CSV
    For now do not preserve or distinguish filename;
    user has one file at once.
    '''
    print('Calling write_dataframe')
    # simulate reading in big data with a delay
    #time.sleep(SIMULATE_WRITE_DELAY)
    filename = os.path.join(filecache_dir, session_id)
    df.to_pickle(filename)

# cache memoize this and add timestamp as input!
@cache.memoize()
def read_dataframe(session_id, timestamp):
    '''
    Read dataframe from disk, for now just as CSV
    '''
    print('Calling read_dataframe')
    filename = os.path.join(filecache_dir, session_id)
    df = pd.read_pickle(filename)
    # simulate reading in big data with a delay
    print('** Reading data from disk **')
    #time.sleep(SIMULATE_READ_DELAY)
    return df

# # if there were few slider values, we could conceivably
# # call this function with them all during data load,
# # especially if this could be parallelised
# @cache.memoize()
# def add_noise(series, slider_value):
#     '''
#     Add noise to column.
#     '''
#     # simulate complex transform with a delay
#     print('** Calculating data transform **')
#     #time.sleep(SIMULATE_TRANSFORM_DELAY)
#     noise = np.random.randn(len(series))*slider_value
#     return series+noise

@app.callback(
    Output('filecache_marker', 'children'),
    [Input('upload-data', 'contents'),
        Input('upload-data', 'filename'),
        Input('upload-data', 'last_modified')],
    [State('session-id', 'children')])
def save_file(contents, filename, last_modified, session_id):
    # write contents to file
    print('Calling save_file')
    print('New last_modified would be',last_modified)
    if contents is not None:
        print('contents is not None')
        # Simulate large file upload with sleep
        #time.sleep(SIMULATE_UPLOAD_DELAY)
        df = parse_table(contents, filename)
        write_dataframe(session_id, df)
        return str(last_modified) # not str()?

# # could remove last_modified state
# # but want either it or filecache timestamp as input to read_dataframe
# @app.callback(Output('data-table-div', 'children'),
#                 [Input('filecache_marker', 'children')],
#                 [State('upload-data', 'last_modified'),
#                 State('session-id','children')])
# def update_table(filecache_marker, timestamp, session_id):
#     print('Calling update_table')
#     if filecache_marker is not None:
#         print('filecache marker is not None')
#         print('filecache marker:',filecache_marker)
#         try:
#             df = read_dataframe(session_id, timestamp)
#         except Exception as e:
#             # Show exception
#             return str(e)
#         output = [dt.DataTable(rows=df.to_dict('records'))]
#         return output


# @app.callback(Output('two-column-graph', 'figure'),
#                 [Input('filecache_marker', 'children'),
#                 Input('slider','value')],
#                 [State('upload-data', 'last_modified'),
#                 State('session-id','children')])
# def update_graph(filecache_marker, slider_value, timestamp, session_id):
#     ''' Plot first column against second '''
#     print('Calling update_graph')
#     # For now no dtype checking!
#     # For now no error checking either
#     if filecache_marker is None:
#         raise ValueError('No data yet') # want PreventUpdate
#     df = read_dataframe(session_id, timestamp)
#     y_noised = add_noise(df.iloc[:,1], slider_value)
#     traces = [go.Scatter(x=df.iloc[:,0], y=y_noised,
#                 mode='markers', marker=dict(size=10, opacity=0.7),
#                 text=df.index)]
#     figure = {
#         'data': traces,
#         'layout': {
#             'title': 'Graph',
#             'xaxis': {'title': df.columns[0]},
#             'yaxis': {'title': df.columns[1]},
#             'hovermode': 'closest',
#         }
#     }
#     return figure


if __name__ == '__main__':
    app.run_server(debug=True)