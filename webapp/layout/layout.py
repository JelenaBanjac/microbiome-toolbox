from dash import dcc
from dash import html as dhc
from layout.navbar.navbar import navbar

# content = html.Div(id="page-content")
from pages.home import home
from pages.page1 import page1
from pages.page2 import page2
from pages.page3 import page3
from pages.page4 import page4
from pages.page5 import page5
from pages.page6 import page6

layout = dhc.Div(
    [
        dcc.Location(id="url", refresh=False),
        # dcc.Store(data=session_id, id='session-id'),
        dcc.Store(id="microbiome-dataset-location"),
        dcc.Store(id="microbiome-trajectory-location"),
        navbar,
        home.layout,
        page1.layout,
        page2.layout,
        page3.layout,
        page4.layout,
        page5.layout,
        page6.layout,
        dhc.Div(id="page-not-found"),
    ]
)
