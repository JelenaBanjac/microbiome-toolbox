from dash.dependencies import Input, Output, State
from app import app


# def parse_contents(contents, filename, date):
#     # content_type, content_string = contents.split(',')

#     # decoded = base64.b64decode(content_string)
#     # try:
#     #     if 'csv' in filename:
#     #         # Assume that the user uploaded a CSV file
#     #         df = pd.read_csv(
#     #             io.StringIO(decoded.decode('utf-8')))
#     #     elif 'xls' in filename:
#     #         # Assume that the user uploaded an excel file
#     #         df = pd.read_excel(io.BytesIO(decoded))
#     # except Exception as e:
#     #     print(e)
#     #     return html.Div([
#     #         'There was an error processing this file.'
#     #     ])

#     return "Hello"

# @app.callback(Output('output-data-upload', 'children'),
#               Input('upload-data', 'contents'),
#               State('upload-data', 'filename'),
#               State('upload-data', 'last_modified'))
# def update_output(list_of_contents, list_of_names, list_of_dates):
#     if list_of_contents is not None:
#         children = [
#             parse_contents(c, n, d) for c, n, d in
#             zip(list_of_contents, list_of_names, list_of_dates)]
#         return children


@app.callback(Output('output-data-upload', 'children'),
              Input('upload-default-data', 'n_clicks'),
              )
def update_output(n_clicks):
    
    return n_clicks
    
