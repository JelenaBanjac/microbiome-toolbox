from dash.dependencies import Input, Output, State
from app import app


@app.callback(Output('page-0-content', 'children'),
              [Input('page-0-dropdown', 'value')])
def page_1_dropdown(value):
    return 'You have selected "{}"'.format(value)
