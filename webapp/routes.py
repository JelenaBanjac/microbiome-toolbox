from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_html_components as dhc
# from pages import page1, page2, page3, page4, page5, page6

from app import app
from utils.constants import (
    home_location,
    page1_location,
    page2_location,
    page3_location,
    page4_location,
    page5_location,
    page6_location,
)
from pages.home import home
from pages.page0 import page0


@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")],
)
def render_page_content(pathname):
    print(pathname)
    if pathname == home_location:
        return home.layout
    elif pathname == page1_location:
        return page0.layout
    # elif pathname == page1_location:
    #     return page1.layout
    # elif pathname == page2_location:
    #     return page2.layout
    # elif pathname == page3_location:
    #     return page3.layout
    # elif pathname == page4_location:
    #     return page4.layout
    # elif pathname == page5_location:
    #     return page5.layout
    # elif pathname == page6_location:
    #     return page6.layout
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            dhc.H1("404: Not found", className="text-danger"),
            dhc.Hr(),
            dhc.P(f"The pathname {pathname} was not recognized..."),
        ]
    )
