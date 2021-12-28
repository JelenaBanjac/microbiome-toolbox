# from dash import html as dhc
from dash import dcc
from dash import html as dhc

page_1_layout = dhc.Div([
    dhc.H1('Page 1'),
    dcc.Dropdown(
        id='page-0-dropdown',
        options=[{'label': i, 'value': i} for i in ['LA', 'NYC', 'MTL']],
        value='LA'
    ),
    dhc.Div(id='page-0-content'),
    dhc.Br(),
    dcc.Link('Go to Page 2', href='/page-2'),
    dhc.Br(),
    dcc.Link('Go back to home', href='/'),
])
