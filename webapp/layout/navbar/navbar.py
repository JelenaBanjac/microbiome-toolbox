import dash_bootstrap_components as dbc
from dash import dcc
from dash import html as dhc
from utils.constants import home_location

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink([
                "Known issues",
                dhc.I(
                    title="Known issues",
                    className="fa fa-bug",
                    style={"marginLeft": "10px"},
                ),
        ], href="https://github.com/JelenaBanjac/microbiome-toolbox/issues", target="_blank",)),
        dbc.NavItem(dbc.NavLink([
                "Cite paper",
                dhc.I(
                    title="Cite paper",
                    className="fa fa-quote-left",
                    style={"marginLeft": "10px"},
                ),
        ], href="https://www.biorxiv.org/content/10.1101/2022.02.14.479826v1", target="_blank",)),
    ],
    brand="Microbiome Toolbox",
    brand_href=home_location,
    color="dark",
    dark=True,
)
