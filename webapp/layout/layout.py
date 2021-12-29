from dash import html
from dash import dcc
from dash import html as dhc
from layout.navbar.navbar import navbar


content = html.Div(id="page-content")

layout = dhc.Div(
    [
        dcc.Location(id="url", refresh=False),
        # dcc.Store(data=session_id, id='session-id'),
        dcc.Store(id='microbiome-dataset-location'),
        dcc.Store(id='microbiome-trajectory-location'),
        navbar,
        content,
    ]
)
