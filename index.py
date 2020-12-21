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
from datetime import date
from dateutil.relativedelta import relativedelta
import validate

from json2html import *
import dash_dangerously_set_inner_html
from yahoofinancials import YahooFinancials


colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}



# open output file for reading
with open('one1.txt', 'r') as filehandle:
    basicList = json.load(filehandle)




from flask import Flask, Response



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

##meta_tags=[
##    {
##        'name': 'description',
##        'content': 'My description'
##    },
##    {
##        'http-equiv': 'X-UA-Compatible',
##        'content': 'IE=edge'
##    }
##]


app = dash.Dash(__name__,    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
                external_scripts=external_scripts,
                external_stylesheets=external_stylesheets)


server = app.server
#app.scripts.config.serve_locally = True
#app.css.config.serve_locally = True


# Your webpage title

app.title = 'Stock Predictor | An open source tool to predict future stock prices using machine learning. &#x1F4CA;'

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

    elif pathname=='/validate':

        return page_2_layout

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
                        dbc.Col(html.Img(src="assets/keras-logo-small-wb-1.png", height="50px")),
                        dbc.Col(dbc.NavbarBrand("StockBasket", className="ml-4",style={"color":"#00ff00"})),
                        dbc.Col(dbc.NavLink("Validate",href="/validate", className="ml-4",style={"color":"#fff","font-size": "135%"})),
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
                dbc.CardHeader("Stock Predictor \ud83d\udcc8"),
                dbc.CardBody(
                    [


                html.Div([
                html.Div([
                dcc.Dropdown(
                    id="dropdown",
                    options=basicList,
                    value=''
                ), ],className="four columns"),         
                ],className="twelve columns",style={'margin-bottom':"2%"}),

                html.Div(html.P("Type"),style={"margin-top":'1%','margin-bottom':"1%"}),
                dcc.RadioItems(

                    id='radio1',
    options=[
        {'label': 'Closing Price Prediction', 'value': '0'},
        {'label': 'Volume Prediction', 'value': '1'}],
    value='0',
    inputStyle={"margin-right": "10px","margin-left": "10px"},
    labelStyle={'display': 'inline-block'}                
),  
               html.Div(html.P("Mode"),style={"margin-top":'3%'}),

              dcc.RadioItems(

                    id='radio',
    options=[
        {'label': 'Basic', 'value': '0'},
        {'label': 'Standard', 'value': '1'},
        {'label': 'Super', 'value': '2'}],
    value='0',
    inputStyle={"margin-right": "20px"}                
),  

                 html.Button('Submit', id='button',style={"margin-top":"2%","margin-bottom":"1%"}),



              ]), ]
            
        ),],style={ 'margin-top':'3%'}),   


         html.Div(id='output-image',style={"margin-top":"10px","margin-":"10px"}),
        
    ],className='ten columns offset-by-one'),
     ]),

])





# Each time you give input as image in id ('upload-data') this function app.callback will fire and give you output in id ('output-image')



@app.callback(Output('output_image_table_1', 'children'),
              [Input('dropdown2','value')])
def update_graph_interactive_image(value):

    print(value,'\n\n')
    value,label,stockLabel=int(value.split("$$$$$")[1]),value.split("$$$$$")[0],value.split("$$$$$")[2]


    if value==0:

        df=stock.getTrendingNews(stockLabel+" India")
        print(df)
        if len(df)==0:
            return [

                html.Div([
                
                html.Div(html.H5("Sorry! No trending news exist for this stock."))

                ],className="twelve columns", style={"margin-top":"1%","padding":"0.5%"})
                ]

        else:

            l=[]
            temp=[]

            for i in range(len(df)):
                
                imgUrl=df['Image'][i]
                url=df['URL'][i]
                ps=df['Title'][i]
                date=str(df['Date'][i]).split(" ")[0]
                card = dbc.Card(
        [
            dbc.CardImg(src=imgUrl, top=True),
            dbc.CardBody(
                [
                    html.H4(date, className="card-title"),
                    html.P( ps),
                    html.A(html.Button('Get News'),href=url)
                ]
            ),
        ],
        style={"width": "52%"},

    )
                temp.append(html.Div(card,className="six columns"))

                if i%2!=0 and i!=0:
                    l.append(html.Div(children=temp,className="row"))
                    l.append(html.Div())
                    temp=[]

                if (i%2==0 and i==len(df)-1):
                    l.append(card)
                    l.append(html.Div())
                  
                    
                    

            print(l)
       
            return [html.Div(html.H5(label),style={"padding":"2.6%","margin-top":"3.8%"}),html.Div(children=l,className="twelve columns offset-by-one",style={"padding":"5%"})]

                
        
    else:
      
        string=stock.getWordCloud(stockLabel+" India")
        if len(string)==0:
            return [
                html.Div([
                html.Div(html.H5("Sorry! No trending news exist for this stock to make the word cloud."))


                ],className="twelve columns", style={"margin-top":"1%","padding":"0.5%"})]

        else:
            img1=string[0]
            cards=dbc.Card(
    [
        dbc.CardBody(
            [html.H5("Word Cloud")]
        ),
        dbc.CardImg(
            src=(
                'data:image/jpg;base64,{}'.format(img1)
            )
        ),
         dbc.CardBody(
            [
                html.P(
                  "Word Cloud of Recent News of "+stockLabel
                ),
            ]
        ),
    ],
    style={"width": "69%"},
)
            print("success1")
            l=[]
            l.append(cards)
            return [html.Div(children=l,className="nine columns offset-by-three",style={"margin-top":"3%"})]



@app.callback(Output('output_image_table', 'children'),
              [Input('dropdown1','value')])
def update_graph_interactive_image(value):
    value,label,stock=int(value.split("$$$$$")[1]),value.split("$$$$$")[0],value.split("$$$$$")[2]

    if value==0:
        yahoo_financials = YahooFinancials(stock)
        input = yahoo_financials.get_financial_stmts('annual', 'income')
        htmls=json2html.convert(json = input, table_attributes="id=\"info-table\" class=\"table-inverse\"")
        uc=dash_dangerously_set_inner_html.DangerouslySetInnerHTML(str(htmls))


        


    elif value==1:
        yahoo_financials = YahooFinancials(stock)
        input = yahoo_financials.get_financial_stmts('annual', 'balance')
        htmls=json2html.convert(json = input, table_attributes="id=\"info-table\" class=\"table-inverse\"")
        uc=dash_dangerously_set_inner_html.DangerouslySetInnerHTML(str(htmls))


    elif value==2:
        yahoo_financials = YahooFinancials(stock)
        input = yahoo_financials.get_financial_stmts('quarterly', 'cash')
        htmls=json2html.convert(json = input, table_attributes="id=\"info-table\" class=\"table-inverse\"")
        uc=dash_dangerously_set_inner_html.DangerouslySetInnerHTML(str(htmls))


    elif value==3:
        yahoo_financials = YahooFinancials(stock)
        input = yahoo_financials.get_stock_quote_type_data()
        htmls=json2html.convert(json = input, table_attributes="id=\"info-table\" class=\"table-inverse\"")
        uc=dash_dangerously_set_inner_html.DangerouslySetInnerHTML(str(htmls))

    print(uc)


    return [

            html.Div([
            html.Div(html.H5(label+" Report of "+stock),style={"padding":"2%"}),
            html.Div([uc],style={"overflow-x":"auto","overflow-y":"auto","height":"750px"})
            ])

        ]




    

@app.callback(Output('output-image', 'children'),
              [Input('button','n_clicks')],
              [dash.dependencies.State('dropdown', 'value'),
               dash.dependencies.State('radio', 'value'),
               dash.dependencies.State('radio1', 'value')])
def update_graph_interactive_image(click,value,vals,radio1):


        if value!="":

           
            zyes=int(str(vals))

            print("\n This is the value:",value.split("$$$$$")[0],"\n",value.split("$$$$$")[1])

            if int(radio1)==0:
                s=stock.predictData(value.split("$$$$$")[0],5,zyes,value.split("$$$$$")[1])
            else:
                s=stock.predictDataByVolume(value.split("$$$$$")[0],5,zyes,value.split("$$$$$")[1])
                
            #curr=str(datetime.now()).split(" ")[0]
            curr = date.today()
            print(s)

            v=[]
            s1=[]
            i=0
            j=0
            flag=1
            while(flag==1):
                end =  date.today() + relativedelta(days=+i)
                if (end.weekday()==5) or (end.weekday()==6):
                    if i==0 or i==1:
                        s1.append(s[j])
                    else:
                        s1.append(s[j-1])
                    v.append(end)
                    i+=1
                    

                else:
                    if ((i==1) or (i==2)) and (j==0):
                        j+=1
                    s1.append(s[j])
                    v.append(end)
                    i+=1
                    j+=1

                if j==5:
                    flag=0
                    break
                    
##  
##            for i in range(0,5):
##                end =  date.today() + relativedelta(days=+i)
##                
##                if (end.weekday()==5 or end.weekday()==6):
##
##                else:
##                    
##                v.append(end)
            

            print(s1,v)

            if int(radio1)==0:
                sss= dcc.Graph(
            id='example-graph',
            figure={
                'data': [
                    {'x': v, 'y': s1, 'type': 'line', 'name': "Prediction for "+value.split("$$$$$")[0], 'marker': {
                   'color': ['#00ff00']*i
               }, 'line': {'width': 2, 'color':'#7FDBFF'}},
                ],
                'layout': {
                    'title': value.split("$$$$$")[1],
                     'plot_bgcolor': colors['background'],
                    'paper_bgcolor': colors['background'],
                    'font': {
                        'color': colors['text']
                    
                },
                    'xaxis':{'title':'Date'},
                'yaxis':{'title':'Price in INR'}
            }
            }
        )
            else:
                 sss= dcc.Graph(
            id='example-graph',
            figure={
                'data': [
                    {'x': v, 'y': s1, 'type': 'bar', 'name': value,'marker':dict(color='#00EF00')},
                ],
                'layout': {
                    'title': value.split("$$$$$")[1],
                     'plot_bgcolor': colors['background'],
                    'paper_bgcolor': colors['background'],
                    'font': {
                        'color': colors['text']
                    
                },
                    'xaxis':{'title':'Date'},
                'yaxis':{'title':'Volume of Stocks Traded'}
            }
            }
        )
                
            sss=[sss]


                     
            card=dbc.Card(
            [
                dbc.CardHeader(""),
                dbc.CardBody(
                    [

                     
                     html.Div(id='graphsOutput',children=sss)


              ]), ]
            
        )
            card1= dbc.Card(
            [
                dbc.CardHeader(""),
                dbc.CardBody(
                    [

                     
                     html.Div(html.H4(value.split("$$$$$")[1]+" Statistics")),
                     html.Div([
                dcc.Dropdown(
                    id="dropdown1",
                    options=[{"label":"Annual Income Statement","value":"Annual Income Statement Data$$$$$0$$$$$"+value.split("$$$$$")[0]},{"label":"Annual Balance Sheet","value":"Annual Balance Sheet Data$$$$$1$$$$$"+value.split("$$$$$")[0]},{"label":"Quarterly Cash Flow","value":"Quarterly Cash Flow Statement Data$$$$$2$$$$$"+value.split("$$$$$")[0]},{"label":"Stock Quote Data","value":"Stock Quote Data$$$$$3$$$$$"+value.split("$$$$$")[0]}][::-1],
                    value='',style={"margin-bottom":"2%"}),],className="three columns",style={"margin-bottom":"2%"} ),         
                ],className="twelve columns"),

                html.Div(id="output_image_table"),
       
              ])


            card2= dbc.Card(
            [
                dbc.CardHeader(""),
                dbc.CardBody(
                    [

                     
                     html.Div(html.H4("Trending News and Word Cloud of "+value.split("$$$$$")[1])),
                     html.Div([
                dcc.Dropdown(
                    id="dropdown2",
                    options=[{"label":"Trending News","value":"Trending News$$$$$0$$$$$"+value.split("$$$$$")[1]},{"label":"Word Cloud of News","value":"Word Cloud$$$$$1$$$$$"+value.split("$$$$$")[1]}],
                    value='',style={"margin-bottom":"2%"}),],className="three columns",style={"margin-bottom":"2%"} ),
                     
                     

                html.Div(id="output_image_table_1"),
       
              ])])

            

            c3= html.Div([

                    card,
                    card1,
                    card2


                ]),


            print(c3)
            
            
            return c3


#### Page 2 Layout ####        

page_2_layout = html.Div([


        # Creating a Navbar

        dbc.Navbar(
        [   
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src="assets/keras-logo-small-wb-1.png", height="50px")),
                        dbc.Col(dbc.NavbarBrand("StockBasket", className="ml-4",style={"color":"#00ff00"})),
                        dbc.Col(dbc.NavLink("Validate",href="/validate", className="ml-4",style={"color":"#fff","font-size": "135%"})),
                        
                    ],
                    align="center",
                    no_gutters=True,
                ),
                href="/",
            ),

           
            
            dbc.NavbarToggler(id="navbar-toggler-10"),
           
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
                dbc.CardHeader("Stock Predictor \ud83d\udcc8"),
                dbc.CardBody(
                    [


                html.Div([
                html.Div([
                dcc.Dropdown(
                    id="dropdown10",
                    options=basicList,
                    value=''
                ), ],className="four columns"),         
                ],className="twelve columns"),

                html.Div(html.P("Type"),style={"margin-top":'4%','margin-bottom':"1%"}),
                dcc.RadioItems(

                    id='radio11',
    options=[
        {'label': 'Closing Price Prediction', 'value': '0'},
        {'label': 'Volume Prediction', 'value': '1'}],
    value='0',
    inputStyle={"margin-right": "10px","margin-left": "10px"},
    labelStyle={'display': 'inline-block'}                
),              
               
               html.Div(html.P("Mode"),style={"margin-top":'3%','margin-bottom':"2%"}),

              dcc.RadioItems(

                    id='radio10',
    options=[
        {'label': 'Basic', 'value': '0'},
        {'label': 'Standard', 'value': '1'},
        {'label': 'Super', 'value': '2'}],
    value='0',
    inputStyle={"margin-right": "20px"}                
),  

                 html.Button('Submit', id='button10',style={"margin-top":"2%","margin-bottom":"1%"}),



              ]), ]
            
        ),],style={ 'margin-top':'3%'}),   


         html.Div(id='output-image-10',style={"margin-top":"10px","margin-":"10px"}),
        
    ],className='ten columns offset-by-one'),
     ]),

])



@app.callback(Output('output-image-10', 'children'),
              [Input('button10','n_clicks')],
              [dash.dependencies.State('dropdown10', 'value'),
               dash.dependencies.State('radio10', 'value'),
               dash.dependencies.State('radio11', 'value')])
def update_graph_interactive_image(click,value,vals,radio1):

    if value!="":

           
        zyes=int(str(vals))

        print("\n This is the value:",value.split("$$$$$")[0],"\n",value.split("$$$$$")[1])

        if int(radio1)==0:
            s=validate.predictData(value.split("$$$$$")[0],5,zyes,value.split("$$$$$")[1])

            s,d=s[0],s[1]
            print(s,d)
            sss= dcc.Graph(
            id='example-graph',
            figure={
                'data': [
                    {'x': d[0], 'y': s, 'type': 'line', 'name': "Predicted"+" "+value.split('$$$$$')[0], 'marker': {
                   'color': ['#00ff00']*5
               }, 'line': {'width': 2, 'color':'#7FDBFF'}},

                 {'x': d[0], 'y': d[1], 'type': 'line', 'name': "Actual"+" "+value.split('$$$$$')[0], 'marker': {
                   'color': ['#0000ff']*5
               }, 'line': {'width': 2, 'color':'#00FF00'}},   
                ],
                'layout': {
                    'title': value.split("$$$$$")[1],
                     'plot_bgcolor': colors['background'],
                    'paper_bgcolor': colors['background'],
                    'font': {
                        'color': colors['text']
                    
                },
                    'xaxis':{'title':'Date'},
                'yaxis':{'title':'Price in INR'}
            }
            }
        )
            
            print([dbc.Card(
    [
         dbc.CardBody(
            [
               sss
            ]
        ),
    ],
    style={"width": "100%"},
)])
            return [dbc.Card(
    [
         dbc.CardBody(
            [
               sss
            ]
        ),
    ],
    style={"width": "100%"},
)]


        else:
            s=validate.predictDataByVolume(value.split("$$$$$")[0],5,zyes,value.split("$$$$$")[1])

            s,d=s[0],s[1]
            print(s,d)
            sss= dcc.Graph(
            id='example-graph',
            figure={
                'data': [
                    {'x': d[0], 'y': s, 'type': 'bar', 'name': "Predicted"+" "+value.split('$$$$$')[0], 'marker': {
                   'color': ['#00Ef00']*5
               },},

                 {'x': d[0], 'y': d[1], 'type': 'bar', 'name': "Actual"+" "+value.split('$$$$$')[0], 'marker': {'color': ['#0000Ef']*5 }}
                
                ],
                'layout': {
                    'title': value.split("$$$$$")[1],
                     'plot_bgcolor': colors['background'],
                    'paper_bgcolor': colors['background'],
                    'font': {
                        'color': colors['text']
                    
                },
                    'xaxis':{'title':'Date'},
                'yaxis':{'title':'Volume of Stocks Traded'}
            }
            }
        )
            print([dbc.Card(
    [
         dbc.CardBody(
            [
               sss
            ]
        ),
    ],
    style={"width": "100%"},
)])
            return [dbc.Card(
    [
         dbc.CardBody(
            [
               sss
            ]
        ),
    ],
    style={"width": "100%"},
)]
                        
                    
                    
                    
                    
                    










if __name__ == '__main__':
    app.run_server(debug=False)

