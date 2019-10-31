# Importing Libraries

import dash_core_components as dcc
import dash
from dash.dependencies import Input, Output,State
import dash_bootstrap_components as dbc
import dash_html_components as html
import base64
from base64 import decodestring
import numpy as np
import os
import urllib
import stock
import dash_table
from datetime import datetime
import json
import pandas as pd









# open output file for reading
with open('one.txt', 'r') as filehandle:
    basicList = json.load(filehandle)


from flask import Flask, Response
import cv2



# external JS
external_scripts = [
    'https://www.google-analytics.com/analytics.js',
    "https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js",
    {'src': 'https://cdn.polyfill.io/v2/polyfill.min.js'},
    {
        'src': 'https://cdnjs.cloudflare.com/ajax/libs/lodash.js/4.17.10/lodash.core.js',
        'integrity': 'sha256-Qqd/EfdABZUcAxjOkMi8eGEivtdTkh3b65xCZL4qAQA=',
        'crossorigin': 'anonymous'
    }
]

# external CSS stylesheets
external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    {
        'href': 'https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css',
        'rel': 'stylesheet',
        'integrity': 'sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO',
        'crossorigin': 'anonymous'
    },

    {
        'href': 'https://fonts.googleapis.com/css?family=Varela',
        'rel': 'stylesheet',
        'integrity': 'sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO',
        'crossorigin': 'anonymous'
    }

    
]



# Making app as an object of dash class. Invoking Dash function.

# Provide your name and content.



app = dash.Dash(__name__,meta_tags=[
    {
        'name': 'description',
        'content': 'My description'
    },
    {
        'http-equiv': 'X-UA-Compatible',
        'content': 'IE=edge'
    }
],
                external_scripts=external_scripts,
                external_stylesheets=external_stylesheets,
                static_folder='assets')


server = app.server
app.scripts.config.serve_locally = True
app.css.config.serve_locally = True


# Your webpage title

app.title = 'Hackathon 	&#x1F525;'

app.config['suppress_callback_exceptions']=True




'''

Here to write <div> , <h1>, ,<p>, etc. tags we need to specify them as html.Div, html.H1, html.P (As we are importing this tags from html library)


'''




# Your main layout

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

PLOTLY_LOGO = "/"



app.config['suppress_callback_exceptions']=True






# Update the index

@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':

        # Default path is / . It will render Page_1_layout
        
        return page_1_layout

    else:
        return []
    # You could also return a 404 "URL not found" page here



# Root page

page_1_layout = html.Div([


        # Creating a Navbar

        dbc.Navbar(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src="assets/keras-logo-small-wb-1.png", height="30px")),
                        dbc.Col(dbc.NavbarBrand("Hackathon", className="ml-4")),
                    ],
                    align="center",
                    no_gutters=True,
                ),
                href="/",
            ),

           
            
            dbc.NavbarToggler(id="navbar-toggler"),
           
        ],
        color="dark",      
        dark=True,
        sticky="top",
    ),


    
     html.Div([    
     html.Div([


        html.Div([

         # Inside this class we are making dcc.Upload which will upload our image.
            
       dbc.Card(
            [
                dbc.CardHeader("Hackathon \u2b50"),
                dbc.CardBody(
                    [

                dcc.Dropdown(
                    id="dropdown",
                    options=basicList,
                    value='MTL'
                ),          
                           
               html.Div(html.P("Mode"),style={"margin-top":'3%'}),

              dcc.RadioItems(

                    id='radio',
    options=[
        {'label': 'Basic', 'value': '0'},
        {'label': 'Standard', 'value': '1'},
        {'label': 'Super', 'value': '2'}],
    value='0',
),  

                 html.Button('Submit', id='button'),



              ]), ]
            
        ),],style={ 'margin-top':'3%'}),   


         html.Div(id='output-image',style={"margin-top":"10px","margin-":"10px"}),
        
    ],className='ten columns offset-by-one'),
     ]),

])





# Each time you give input as image in id ('upload-data') this function app.callback will fire and give you output in id ('output-image')

@app.callback(Output('output-image', 'children'),
              [Input('button','n_clicks')],
              [dash.dependencies.State('dropdown', 'value'),
               dash.dependencies.State('radio', 'value'),
               dash.dependencies.State('dropdown', 'label')])
def update_graph_interactive_image(click,value,vals,labe):


        if value!="":

           
            zyes=int(str(vals))

            print("\n This is the value:",value,"\n")
            s=stock.predictData(value,5,zyes)
            curr=str(datetime.now()).split(" ")[0]

            v=[curr]

            for i in range(5):
                st=curr.split("-")[0]+"-"+curr.split("-")[1]+"-"+str(int(curr.split("-")[-1])+i+1)
                v.append(st)

            print(s)

            sss= dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': v, 'y': s, 'type': 'line', 'name': value},
            ],
            'layout': {
                'title': labe
            }
        }
    )
            sss=[sss]
            
            
            
            return sss







if __name__ == '__main__':
    app.run_server(debug=True)

