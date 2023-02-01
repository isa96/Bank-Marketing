import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
import dash_table
from dash.dependencies import Input, Output, State
import pickle
import numpy as np

loadModel_log = pickle.load(open('pipe_logreg_bank.sav', 'rb'))
loadModel_xgb = pickle.load(open('xgb_bank.sav', 'rb'))
loadModel_tpot = pickle.load(open('pipe_tpot.sav', 'rb'))
loadModel_tpot2 = pickle.load(open('pipe_tpot2.sav', 'rb'))

job_val = ['hm','srv','adm','bc','tech','rtr','mn','unm','se','unk','ent','stu']
mar_val = ['mar', 'sin', 'div', 'unk']
edu_val = ['4y','hs','6y','9y','pc','unk','ud','il']

bank = pd.read_csv('bank_dash.csv')
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

def generate_table(dataframe, page_size = 10, n_job = '', n_marital = '', n_edu = ''):
    if n_job == '':
        dataframe = dataframe
    else:
        dataframe = dataframe[dataframe['Job'] == n_job]
    if n_marital == '':
        dataframe = dataframe
    else:
        dataframe = dataframe[dataframe['Marital Status'] == n_marital]
    if n_edu == '':
        dataframe = dataframe
    else:
        dataframe = dataframe[dataframe['Education'] == n_edu]

    return dash_table.DataTable(
        id = 'data-table',
        columns = [{
            'name': i,
            'id': i
        } for i in dataframe.columns],
        data = dataframe.to_dict('records'),
        page_action = 'native',
        page_current = 0,
        page_size = page_size,
        style_table = {'overflowX': 'scroll'}
    )

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1('Final Project: Bank Marketing'),
    html.P('Created by: Listia'),
    dcc.Tabs(children = 
        [dcc.Tab(value = 'Tab1', label = 'Data Frame Bank', children = [
        html.Div([
            html.Div([
                html.P('Job'),
                dcc.Dropdown(value = '',
                id = 'filter-job',
                options = [{'label' : i, 'value': i} for i in bank['Job'].unique()])
            ], className = 'col-3'),
            
            html.Div([
                html.P('Marital'),
                dcc.Dropdown(value = '',
                id = 'filter-marital',
                options = [{'label' : i, 'value': i} for i in bank['Marital Status'].unique()])
            ], className = 'col-3'),

            html.Div([
                html.P('Education'),
                dcc.Dropdown(value = '',
                id = 'filter-education',
                options = [{'label' : i, 'value': i} for i in bank['Education'].unique()])
            ], className = 'col-3'),
            html.Div([
                html.P('Max Row'),

                dcc.Input(id='filter-row', type='number', value = 10)
                ], className = 'col-3'),
             
        ], 
        className = 'row'),
        html.Br(),
        html.Div(children = [
            html.Center(html.Button('Filter', id='filter'))
            ]),
        
        html.Br(),
       
        html.Div(id = 'div_table',
            children = [generate_table(bank)])
            ]
            )
            ,
        dcc.Tab(label = 'Bar-Chart', value = 'tab-satu', children=
             [html.Div(children =[
                html.Div(children =[
                html.P('Choose X-axis'),    
                dcc.Dropdown(id = 'x-axis-1', options = [{'label': i, 'value': i} for i in bank.select_dtypes('object').columns], 
                value = 'y')
                ], className = 'col-3')
                ], className = 'row'),

                html.Div([
                ## Graph Bar
                dcc.Graph(
                id = 'graph-bar',
                figure ={
                    'data' : [
                        {'x': bank.groupby('Job')['y'].count().index, 'y': bank[bank['y'] == 'no'].groupby('Job')['y'].count(), 'type': 'bar', 'name' :'Target No'},
                        {'x': bank.groupby('Job')['y'].count().index, 'y': bank[bank['y'] == 'yes'].groupby('Job')['y'].count(), 'type': 'bar', 'name' : 'Target Yes'},
                    ], 
                    'layout': {'title': 'Bar Chart'}  
                    }
                    )])]),

        dcc.Tab(label ='Pie-Chart', value = 'tab-tiga', children =[
                html.Div(dcc.Dropdown(id ='pie-dropdown', options = [{'label': i, 'value': i} for i in bank.select_dtypes('object').columns], value = 'Job'), 
                className = 'col-3'),
                html.Div([
                dcc.Graph(
                id = 'graph-pie',
                figure ={
                    'data' : [
                        go.Pie(labels = ['{}'.format(i) for i in list(bank['Job'].unique())], 
                                        values = [round((len(bank[(bank['y'] == 'yes') & (bank['Job'] == i)])/ len(bank[bank['Job'] == i]))*100,2)
                                         for i in list(bank['Job'].unique())],
                                         
                                        sort = False)
                    ], 
                        'layout': {'title': 'Acquired Customers Pie Chart'}}
                    )])   
            ]),
            
        dcc.Tab(value = 'Tab4', label = 'Prototype Model', children = [
            html.Div([
            html.Div([
                html.P('Age'),
                dcc.Input(id='cust-age', value = '56', type = 'number')],
                className = 'col-3')
            ,
            html.Div([
                html.P('Job'),
                dcc.Dropdown(value = 'hm',
                id = 'filter-job-2',
                options = [{'label' : i, 'value': j} for i, j in zip(bank['Job'].unique(), job_val)])
            ], className = 'col-3'),
            
            html.Div([
                html.P('Marital'),
                dcc.Dropdown(value = 'sin',
                id = 'filter-marital-2',
                options = [{'label' : i, 'value': j} for i, j in zip(bank['Marital Status'].unique(), mar_val)])
            ], className = 'col-3'),

            html.Div([
                html.P('Education'),
                dcc.Dropdown(value = '4y',
                id = 'filter-education-2',
                options = [{'label' : i, 'value': j} for i, j in zip(bank['Education'].unique(), edu_val)])
            ], className = 'col-3'),
             
        ], 
        className = 'row'),
        html.Br(),

        html.Div([
            html.Div([
                html.P('Euribor 3M'),
                dcc.Input(id='euribor', value = '3', type = 'number')],
                className = 'col-3'),

            html.Div([
                html.P('Consumer Confidence Index'),
                dcc.Input(id='conf-idx', value = '-36', type = 'number')],
                className = 'col-3'),

            html.Div([
                html.P('Consumer Price Index'),
                dcc.Input(id='price-idx', value = '94', type = 'number')],
                className = 'col-3'),

            html.Div([
                html.P('Bank Strategy'),
                dcc.Dropdown(value = 'Balanced',
                id = 'filter-bank-strategy',
                options = [{'label' : 'Balanced', 'value': 'Balanced'},
                    {'label' : 'Aggresive', 'value': 'Aggresive'},
                    {'label' : 'Conservative', 'value': 'Conservative'}])
            ], className = 'col-3')],
        className = 'row')
                ,
        html.Br(),

        html.Div([
            html.Center(html.Div(children = [
                html.Button('Predict', id='predict')
            ], className = 'col-3')),
            html.Br(),
            html.Div(id = 'result-predict', children = 
                html.Center(html.H1('Please fill all the value'))
            )
            ])
            ]),
        ###PROTOTYPE 2
        dcc.Tab(value = 'Tab5', label = 'Prototype Model 2', children = [
        
        html.Div([
            html.Div([
                html.P('Age'),
                dcc.Input(id='cust-agev2', value = '27', type = 'number')],
                className = 'col-3')
            ,
            html.Div([
                html.P('Job'),
                dcc.Dropdown(value = 'adm',
                id = 'filter-job-v2',
                options = [{'label' : i, 'value': j} for i, j in zip(bank['Job'].unique(), job_val)])
            ], className = 'col-3'),
            
            html.Div([
                html.P('Marital'),
                dcc.Dropdown(value = 'sin',
                id = 'filter-marital-v2',
                options = [{'label' : i, 'value': j} for i, j in zip(bank['Marital Status'].unique(), mar_val)])
            ], className = 'col-3'),

            html.Div([
                html.P('Education'),
                dcc.Dropdown(value = '0',
                id = 'filter-education-v2',
                options = [{'label' : 'None', 'value': ''},
                    {'label' : 'Basic 4 Year', 'value': '0'},
                    {'label' : 'Basic 6 Year', 'value': '1'},
                    {'label' : 'Basic 9 Year', 'value': '2'},
                    {'label' : 'High School', 'value': '3'},
                {'label' : 'University Degree', 'value': '4'},
                {'label' : 'Professional Course', 'value': '3'},
                {'label' : 'Unknown', 'value': '4'},
                {'label' : 'Illiterate', 'value': 'Illiterate'}])
            ], className = 'col-3'),
             
        ], 
        className = 'row'),
        
        html.Br(),
        html.Div([
                html.Div([
                html.P('Default'),
                dcc.Dropdown(value = 'No',
                id = 'defaultv2',
                options = [
                    {'label' : 'No', 'value': 'No'},
                    {'label' : 'Unknown', 'value': 'Unknown'}])
            ], className = 'col-3'),

            html.Div([
                html.P('Contacted via'),
                dcc.Dropdown(value = 'Cellular',
                id = 'contactv2',
                options = [
                    {'label' : 'Cellular', 'value': 'Cellular'},
                    {'label' : 'Telephone', 'value': 'Telephone'}])
            ], className = 'col-3'),
            html.Div([
                html.P('Month contacted'),
                dcc.Dropdown(value = 'Apr',
                id = 'monthv2',
                options = [{'label' : i, 'value': i} for i in bank['Month'].unique()])
            ], className = 'col-3'),

            html.Div([
                html.P('Day contacted'),
                dcc.Dropdown(value = 'Mon',
                id = 'dayv2',
                options = [{'label' : i, 'value': i} for i in bank['Day of Week'].unique()])
            ], className = 'col-3'),  
        ], className = 'row'),

        html.Br(),

        html.Div([
                html.Div([
                html.P('Calls in previous campaign:'),
                dcc.Input(id='previousv2', value = '0', type = 'number')],
                className = 'col-3'),

                html.Div([
                html.P('Calls in current campaign:'),
                dcc.Input(id='campaignv2', value = '1', type = 'number')],
                className = 'col-3'),
        ], className = 'row'),

        html.Br(),        
        html.Div([
            html.Div([
                html.P('Euribor 3M'),
                dcc.Input(id='euriborv2', value = '2', type = 'number')],
                className = 'col-3'),

            html.Div([
                html.P('Consumer Confidence Index'),
                dcc.Input(id='conf-idxv2', value = '-36', type = 'number')],
                className = 'col-3'),

            html.Div([
                html.P('Consumer Price Index'),
                dcc.Input(id='price-idxv2', value = '93', type = 'number')],
                className = 'col-3'),
        ], className = 'row')
                ,
        html.Br(),

        html.Div([
            html.Center(html.Div(children = [
                html.Button('Predict', id='predictv2')
            ], className = 'col-3')),
            html.Br(),

            html.Div(id = 'result-predictv2', children = 
                html.Center(html.H1('Please fill all the value'))
            )
        ])
        ])],
    #Tabs content style
    content_style = {
        'fontFamily': 'Arial',
        'borderBottom': '1px solid #d6d6d6',
        'borderLeft': '1px solid #d6d6d6',
        'borderRight': '1px solid #d6d6d6',
        'padding': '44px'
    })
],
    style = {
    'maxwidth' : '1200px',
    'margin': '20 px'})

@app.callback(
    Output(component_id = 'div_table', component_property = 'children'),
    [Input(component_id = 'filter', component_property = 'n_clicks')],
    [State(component_id = 'filter-row', component_property = 'value'),
    State(component_id = 'filter-job', component_property = 'value'),
    State(component_id = 'filter-marital', component_property = 'value'),
    State(component_id = 'filter-education', component_property = 'value')
    ])

def update_table(n_clicks, row, job, marital, education):
    children = [generate_table(bank, row, job, marital, education)]
    return children


@app.callback(
    Output(component_id = 'graph-bar', component_property = 'figure'),
    [Input(component_id = 'x-axis-1', component_property = 'value')])

def create_graph_bar(x1):
    figure = {
                    'data' : [
                        {'x': bank.groupby(x1)['y'].count().index, 'y': bank[bank['y'] == 'no'].groupby(x1)['y'].count(), 'type': 'bar', 'name' : 'Target No'},
                        {'x': bank.groupby(x1)['y'].count().index, 'y': bank[bank['y'] == 'yes'].groupby(x1)['y'].count(), 'type': 'bar', 'name' : 'Target Yes'}
                    ], 
                    'layout': {'title': 'Bar Chart'}  
                    }
    return figure

@app.callback(
    Output(component_id = 'graph-pie', component_property = 'figure'),
    [Input(component_id = 'pie-dropdown', component_property = 'value')]
)
def create_graph_pie(x):
    figure = {
                    'data' : [
                        go.Pie(labels = ['{}'.format(i) for i in list(bank[x].unique())], 
                                values = [round((len(bank[(bank['y'] == 'yes') & (bank[x] == i)])/ len(bank[bank[x] == i]))*100,2)
                                 for i in list(bank[x].unique())],
                                 
                                 sort = False)
                    ], 
                        'layout': {'title': 'Acquired Customers Pie Chart'}}

    return figure                    

@app.callback(
Output(component_id = 'result-predict', component_property = 'children'),
[Input(component_id = 'predict', component_property = 'n_clicks')],
[State(component_id = 'cust-age', component_property = 'value'),
State(component_id = 'filter-job-2', component_property = 'value'),
State(component_id = 'filter-marital-2', component_property = 'value'),
State(component_id = 'filter-education-2', component_property = 'value'),
State(component_id = 'euribor', component_property = 'value'),
State(component_id = 'conf-idx', component_property = 'value'),
State(component_id = 'price-idx', component_property = 'value'),
State(component_id = 'filter-bank-strategy', component_property = 'value')])

def prototype_model (n_clicks, age, job, marital, education, 
euribor, conf_idx, price_idx, bank_strategy):

##Initiate the content -- job
    df_job = {'adm': 0, 'bc': 0, 'tech': 0, 'srv': 0, 'mn': 0, 'ent': 0, 'hm': 0, 'se': 0, 
    'unm': 0, 'stu': 0, 'rtr': 0, 'unk': 0}

    for i in df_job:
        if job == i:
            df_job[i] = 1

##Initiate the content -- Marital status
    df_mar = {'sin': 0, 'mar': 0, 'div': 0, 'unk': 0}
    
    for i in df_mar:
        if marital == i:
            df_mar[i] = 1

##Initiate the content -- Education
    df_edu = {'4y': 0, '6y': 0, '9y': 0,'hs': 0, 'ud': 0, 'pc': 0, 'unk':0, 'il': 0}

    for i in df_edu:
        if education == i:
            df_edu[i] = 1

## Model prediction
    if bank_strategy == 'Aggresive':
#for log_reg
        dat = pd.DataFrame(data = [(age, euribor, price_idx, conf_idx, df_job['adm'],
            df_job['bc'], df_job['ent'], df_job['hm'], df_job['mn'], 
            df_job['rtr'], df_job['se'], df_job['srv'], df_job['stu'], df_job['tech'], 
            df_job['unm'], df_job['unk'], df_mar['div'], df_mar['mar'], df_mar['sin'], 
            df_mar['unk'], df_edu['4y'], df_edu['6y'], 
            df_edu['9y'], df_edu['hs'], df_edu['il'], 
            df_edu['pc'], df_edu['ud'], df_edu['unk'])])

        probaYes = loadModel_log.predict_proba(dat)[0][1]

#for xgb 
    elif bank_strategy == 'Conservative':
        dat = pd.DataFrame(data = [(int(age), float(euribor), float(price_idx), float(conf_idx), df_job['adm'],
            df_job['bc'], df_job['ent'], df_job['hm'], df_job['mn'], 
            df_job['rtr'], df_job['se'], df_job['srv'], df_job['stu'], df_job['tech'], 
            df_job['unm'], df_job['unk'], df_mar['div'], df_mar['mar'], df_mar['sin'], 
            df_mar['unk'], df_edu['4y'], df_edu['6y'], 
            df_edu['9y'], df_edu['hs'], df_edu['il'], 
            df_edu['pc'], df_edu['ud'], df_edu['unk'])], 
        columns = ['Age', 'Euribor 3M', 'Consumer Price Index', 'Consumer Confidence Index', 
        'Job_Admin.', 'Job_Blue-collar', 'Job_Entrepreneur', 'Job_Housemaid', 'Job_Management', 
        'Job_Retired', 'Job_Self-employed', 'Job_Services', 'Job_Student', 'Job_Technician', 
        'Job_Unemployed', 'Job_Unknown', 'Marital Status_Divorced', 'Marital Status_Married', 
        'Marital Status_Single', 'Marital Status_Unknown', 'Education_Basic 4y', 'Education_Basic 6y', 
        'Education_Basic 9y', 'Education_High school', 'Education_Illiterate', 
        'Education_Professional course', 'Education_University degree', 'Education_Unknown'])	

        probaYes = loadModel_xgb.predict_proba(dat)[0][1]

    elif bank_strategy == 'Balanced':
        dat = pd.DataFrame(data = [(age, euribor, price_idx, conf_idx, df_job['adm'],
            df_job['bc'], df_job['ent'], df_job['hm'], df_job['mn'], 
            df_job['rtr'], df_job['se'], df_job['srv'], df_job['stu'], df_job['tech'], 
            df_job['unm'], df_job['unk'], df_mar['div'], df_mar['mar'], df_mar['sin'], 
            df_mar['unk'], df_edu['4y'], df_edu['6y'], 
            df_edu['9y'], df_edu['hs'], df_edu['il'], 
            df_edu['pc'], df_edu['ud'], df_edu['unk'])])

        probaYes = loadModel_tpot.predict_proba(dat)[0][1]
        
    if probaYes < 0.5:
        decision = 'DO NOT CALL THIS CUSTOMER'
    else: 
        decision = 'CALL THIS CUSTOMER'
    result = 'The probability of this customer to accept the bank offer is : {} --- {}'.format(probaYes, decision)
    return result

@app.callback(
Output(component_id = 'result-predictv2', component_property = 'children'),
[Input(component_id = 'predictv2', component_property = 'n_clicks')],
[State(component_id = 'cust-agev2', component_property = 'value'),
State(component_id = 'filter-job-v2', component_property = 'value'),
State(component_id = 'filter-marital-v2', component_property = 'value'),
State(component_id = 'filter-education-v2', component_property = 'value'),
State(component_id = 'contactv2', component_property = 'value'),
State(component_id = 'monthv2', component_property = 'value'),
State(component_id = 'dayv2', component_property = 'value'),
State(component_id = 'defaultv2', component_property = 'value'),
State(component_id = 'campaignv2', component_property = 'value'),
State(component_id = 'previousv2', component_property = 'value'),
State(component_id = 'euriborv2', component_property = 'value'),
State(component_id = 'conf-idxv2', component_property = 'value'),
State(component_id = 'price-idxv2', component_property = 'value')
])

def prototype_model2(n_clicks, age2, job2, marital2, education2, 
contact2, month2, day2, default2, campaign2, previous2,
    euribor2, conf_idx2, price_idx2):

##Initiate the content -- job
    df_job = {'adm': 0, 'bc': 0, 'srv': 0, 'ent': 0, 'unm': 0, 'stu': 0, 'rtr': 0}

    for i in df_job:
        if job2 == i:
            df_job[i] = 1

##Initiate the content -- Marital status
    df_mar = {'sin': 0, 'mar': 0}

    for i in df_mar:
        if marital2 == i:
            df_mar[i] = 1

##Initiate the content -- Default
    if default2 == 'No':
        def_no = 1
        def_unk = 0
    else:
        def_no = 0
        def_unk = 1

## Initiate the content -- contact
    if contact2 == 'Cellular':
        con_cel = 1
        con_tel = 0
    else:
        con_cel = 0
        con_tel = 1

## Initiate the content --  month
    df_mon = {'Apr': 0, 'Dec': 0, 'Jul': 0, 'Mar': 0,'May': 0, 'Oct': 0, 'Sep': 0}

    for i in df_mon:
        if month2 == i:
            df_mon[i] = 1

    ## Initiate the content -- Day of week
    df_day = {'Mon': 0, 'Thu': 0}

    for i in df_day:
        if day2 == i:
            df_day[i] = 1

    data = np.array([df_job['adm'], df_job['bc'], df_job['ent'], df_job['ent'], 
        df_job['srv'], df_job['stu'], df_job['unm'], df_mar['mar'], df_mar['sin'], 
        def_no, def_unk, con_cel, con_tel, df_mon['Apr'], df_mon['Dec'], 
        df_mon['Jul'], df_mon['Mar'], df_mon['May'], df_mon['Oct'], 
        df_mon['Sep'], df_day['Mon'], df_day['Thu'], age2, education2, campaign2, 
        previous2, price_idx2, conf_idx2, euribor2]).reshape(1,-1)

    probaYes = loadModel_tpot2.predict_proba(data)[0][1]
    if probaYes < 0.5:
        decision = 'DO NOT CALL THIS CUSTOMER.'
    else: 
        decision = 'CALL THIS CUSTOMER'
    result = 'The probability of this customer to accept the bank offer is : {} --- {}'.format(probaYes, decision)
    return result

if __name__ == '__main__':
    app.run_server(debug=True)