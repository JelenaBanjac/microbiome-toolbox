from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
# from dash import html as dhc
from dash import html as dhc
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
from pages.page1 import page1
from pages.page2 import page2
from pages.page3 import page3
from pages.page4 import page4
from pages.page5 import page5
from pages.page6 import page6


@app.callback(
    Output("home-layout", "hidden"),
    Output("page-1-layout", "hidden"),
    Output("page-2-layout", "hidden"),
    Output("page-3-layout", "hidden"),
    Output("page-4-layout", "hidden"),
    Output("page-5-layout", "hidden"),
    Output("page-6-layout", "hidden"),
    Output("page-not-found", "children"),
    [Input("url", "pathname")],
)
def render_page_content(pathname):
    print(pathname)
    if pathname == home_location:
        return False, True, True, True, True, True, True, ""
    elif pathname == page1_location:
        return True, False, True, True, True, True, True, ""
    elif pathname == page2_location:
        return True, True, False, True, True, True, True, ""
    elif pathname == page3_location:
        return True, True, True, False, True, True, True, ""
    elif pathname == page4_location:
        return True, True, True, True, False, True, True, ""
    elif pathname == page5_location:
        return True, True, True, True, True, False, True, ""
    elif pathname == page6_location:
        return True, True, True, True, True, True, False, ""
    # If the user tries to reach a different page, return a 404 message
    return True, True, True, True, True, True, True, dbc.Jumbotron(
        [
            dhc.H1("404: Not found", className="text-danger"),
            dhc.Hr(),
            dhc.P(f"The pathname {pathname} was not recognized..."),
        ]
    )
