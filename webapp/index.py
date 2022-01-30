from app import app, server
from environment.settings import APP_DEBUG, APP_HOST, APP_PORT, DEV_TOOLS_PROPS_CHECK
from pages.home.home_callbacks import *
from pages.page1.page1_callbacks import *
from pages.page2.page2_callbacks import *
from pages.page3.page3_callbacks import *
from pages.page4.page4_callbacks import *
from pages.page5.page5_callbacks import *
from pages.page6.page6_callbacks import *
from routes import render_page_content

if __name__ == "__main__":
    app.run_server(
        host=APP_HOST,
        port=APP_PORT,
        debug=APP_DEBUG,
        dev_tools_props_check=DEV_TOOLS_PROPS_CHECK,
    )
