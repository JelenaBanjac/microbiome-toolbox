import dash_bootstrap_components as dbc
from dash import html as dhc
from dash import dcc
from utils.constants import home_location

navbar = dbc.Navbar(
    dbc.Container(
        dcc.Link(
            dbc.NavbarBrand("Microbiome Toolbox", className="ml-2"),
            href=home_location,
        )
    ),
    color="dark",
    dark=True,
    className="mb-5",
)
