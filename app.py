import os
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
from numpy import inf
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Input, Output
import statsmodels.api as sm

df = pd.read_csv("dataset2.csv")

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

X = df.columns[2:-1]
x = df[X]
data  = x.corr()
data  = np.array(data)
fig1 = go.Heatmap(x=X, y=X,z = data, type = "heatmap",colorscale='inferno')
data = [fig1]
fig1 = go.Figure(data = data)
fig1.show()



df['X'] = np.log(df["Density"])
df[df['X'] == -inf] = 0
df['X'] = np.nan_to_num(df['X'])
df['Y'] = np.log(df["Growth Ratio"])
df[df['Y'] == -inf] = 0
df['Y'] = np.nan_to_num(df['Y'])

dataset = df[['X','Y']]
fig2 = px.scatter(x = dataset.Y, y = dataset.X,trendline = "ols")
fig2.update_layout(
    title="Log(Density) vs Log(Growth Ratio)",
    xaxis_title="log(Growth Ratio)",
    yaxis_title="log(Density)",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)
fig2.show()

df['X'] = np.log(df["Population"])
df[df['X'] == -inf] = 0
df['X'] = np.nan_to_num(df['X'])
df['Y'] = np.log(df["Growth Ratio"])
df[df['Y'] == -inf] = 0
df['Y'] = np.nan_to_num(df['Y'])
dataset = df[['X','Y']]
fig3 = px.scatter(x = dataset.Y, y = dataset.X,trendline = "ols")
fig3.update_layout(
    title="Log(Population) vs Log(Growth Ratio)",
    xaxis_title="log(Growth Ratio)",
    yaxis_title="log(Population)",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)
fig3.show()

df['X'] = np.log(df["Area"])
df[df['X'] == -inf] = 0
df['X'] = np.nan_to_num(df['X'])
df['Y'] = np.log(df["Growth Ratio"])
df[df['Y'] == -inf] = 0
df['Y'] = np.nan_to_num(df['Y'])
dataset = df[['X','Y']]
fig4 = px.scatter(x = dataset.Y, y = dataset.X,trendline = "ols")
fig4.update_layout(
    title="Log(Area) vs Log(Growth Ratio)",
    xaxis_title="log(Growth Ratio)",
    yaxis_title="log(Area)",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)
fig4.show()

df['X'] = np.log(df["Air quality"])
df[df['X'] == -inf] = 0
df['X'] = np.nan_to_num(df['X'])
df['Y'] = np.log(df["Growth Ratio"])
df[df['Y'] == -inf] = 0
df['Y'] = np.nan_to_num(df['Y'])
dataset = df[['X','Y']]
fig5 = px.scatter(x = dataset.Y, y = dataset.X,trendline = "ols")
fig5.update_layout(
    title="Log(Air quality) vs Log(Growth Ratio)",
    xaxis_title="log(Growth Ratio)",
    yaxis_title="log(Air quality)",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)
fig5.show()

df['X'] = np.log(df["Water accessibility"])
df[df['X'] == -inf] = 0
df['X'] = np.nan_to_num(df['X'])
df['Y'] = np.log(df["Growth Ratio"])
df[df['Y'] == -inf] = 0
df['Y'] = np.nan_to_num(df['Y'])
dataset = df[['X','Y']]
fig6 = px.scatter(x = dataset.Y, y = dataset.X,trendline = "ols")
fig6.update_layout(
    title="Log(Water Accessibility) vs Log(Growth Ratio)",
    xaxis_title="log(Growth Ratio)",
    yaxis_title="log(Water accessibility)",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)
fig6.show()

df['X'] = np.log(df["Confirmed"])
df[df['X'] == -inf] = 0
df['X'] = np.nan_to_num(df['X'])
df['Y'] = np.log(df["Growth Ratio"])
df[df['Y'] == -inf] = 0
df['Y'] = np.nan_to_num(df['Y'])
dataset = df[['X','Y']]
fig7 = px.scatter(x = dataset.Y, y = dataset.X,trendline = "ols")
fig7.update_layout(
    title="log(Confirmed) vs Log(Growth Ratio)",
    xaxis_title="log(Growth Ratio)",
    yaxis_title="log(Confirmed)",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)
fig7.show()

df['X'] = np.log(df["Active"])
df[df['X'] == -inf] = 0
df['X'] = np.nan_to_num(df['X'])
df['Y'] = np.log(df["Growth Ratio"])
df[df['Y'] == -inf] = 0
df['Y'] = np.nan_to_num(df['Y'])
dataset = df[['X','Y']]
fig8 = px.scatter(x = dataset.Y, y = dataset.X,trendline = "ols")
fig8.update_layout(
    title="Log(Active) vs Log(Growth Ratio)",
    xaxis_title="Log(Growth Ratio)",
    yaxis_title="Log(Active)",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)
fig8.show()

df['X'] = (df["Population"])
df[df['X'] == -inf] = 0
df['X'] = np.nan_to_num(df['X'])
df['Y'] = np.log(df["Confirmed"])
df[df['Y'] == -inf] = 0
df['Y'] = np.nan_to_num(df['Y'])
dataset = df[['X','Y']]
fig9 = px.scatter(x = dataset.Y, y = dataset.X,trendline = "ols")
fig9.update_layout(
    title="Population vs Log(Confirmed)",
    xaxis_title="log(Confirmed)",
    yaxis_title="Population",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)
fig9.show()

df['X'] = np.log(df["Density"])
df[df['X'] == -inf] = 0
df['X'] = np.nan_to_num(df['X'])
df['Y'] = np.log(df["Confirmed"])
df[df['Y'] == -inf] = 0
df['Y'] = np.nan_to_num(df['Y'])
dataset = df[['X','Y']]
fig10 = px.scatter(x = dataset.Y, y = dataset.X,trendline = "ols")
fig10.update_layout(
    title="log(Density) vs Log(Confirmed)",
    xaxis_title="log(Confirmed)",
    yaxis_title="log(Density)",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)
fig10.show()

df['X'] = (df["Area"])
df[df['X'] == -inf] = 0
df['X'] = np.nan_to_num(df['X'])
df['Y'] = np.log(df["Confirmed"])
df[df['Y'] == -inf] = 0
df['Y'] = np.nan_to_num(df['Y'])
dataset = df[['X','Y']]
fig11 = px.scatter(x = dataset.Y, y = dataset.X,trendline = "ols")
fig11.update_layout(
    title="Area vs Log(Confirmed)",
    xaxis_title="log(Confirmed)",
    yaxis_title="Area",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)
fig11.show()

df['X'] = (df["Air quality"])
df[df['X'] == -inf] = 0
df['X'] = np.nan_to_num(df['X'])
df['Y'] = np.log(df["Confirmed"])
df[df['Y'] == -inf] = 0
df['Y'] = np.nan_to_num(df['Y'])
dataset = df[['X','Y']]
fig12 = px.scatter(x = dataset.Y, y = dataset.X,trendline = "ols")
fig12.update_layout(
    title="Air quality vs Log(Confirmed)",
    xaxis_title="log(Confirmed)",
    yaxis_title="Air quality",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)
fig12.show()

df['X'] = (df["Water accessibility"])
df[df['X'] == -inf] = 0
df['X'] = np.nan_to_num(df['X'])
df['Y'] = np.log(df["Confirmed"])
df[df['Y'] == -inf] = 0
df['Y'] = np.nan_to_num(df['Y'])
dataset = df[['X','Y']]
fig13 = px.scatter(x = dataset.Y, y = dataset.X,trendline = "ols")
fig13.update_layout(
    title="Water accessibility vs Log(Confirmed)",
    xaxis_title="log(Confirmed)",
    yaxis_title="Water accesibility",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)
fig13.show()

df['X'] = np.log(df["Active"])
df[df['X'] == -inf] = 0
df['X'] = np.nan_to_num(df['X'])
df['Y'] = np.log(df["Growth Ratio"])
df[df['Y'] == -inf] = 0
df['Y'] = np.nan_to_num(df['Y'])
dataset = df[['X','Y']]
fig14 = px.scatter(x = dataset.Y, y = dataset.X,trendline = "ols")
fig14.update_layout(
    title="log(Active) vs Log(Confirmed)",
    xaxis_title="log(Confirmed)",
    yaxis_title="log(Active)",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)
fig14.show()

df['X'] = (df["Growth Ratio"])
df[df['X'] == -inf] = 0
df['X'] = np.nan_to_num(df['X'])
df['Y'] = np.log(df["Active"])
df[df['Y'] == -inf] = 0
df['Y'] = np.nan_to_num(df['Y'])
dataset = df[['X','Y']]
fig15 = px.scatter(x = dataset.Y, y = dataset.X,trendline = "ols")
fig15.update_layout(
    title="Growth Ratio vs Log(Active)",
    xaxis_title="log(Active)",
    yaxis_title="Growth Ratio",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)
fig15.show()

df['X'] = (df["Water accessibility"])
df[df['X'] == -inf] = 0
df['X'] = np.nan_to_num(df['X'])
df['Y'] = np.log(df["Active"])
df[df['Y'] == -inf] = 0
df['Y'] = np.nan_to_num(df['Y'])
dataset = df[['X','Y']]
fig16 = px.scatter(x = dataset.Y, y = dataset.X,trendline = "ols")
fig16.update_layout(
    title="Water accessibility vs Log(Active)",
    xaxis_title="log(Active)",
    yaxis_title="Water accessibility",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)
fig16.show()

df['X'] = (df["Air quality"])
df[df['X'] == -inf] = 0
df['X'] = np.nan_to_num(df['X'])
df['Y'] = np.log(df["Active"])
df[df['Y'] == -inf] = 0
df['Y'] = np.nan_to_num(df['Y'])
dataset = df[['X','Y']]
fig17 = px.scatter(x = dataset.Y, y = dataset.X,trendline = "ols")
fig17.update_layout(
    title="Air quality vs Log(Active)",
    xaxis_title="log(Active)",
    yaxis_title="Air quality",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)
fig17.show()

df['X'] = (df["Area"])
df[df['X'] == -inf] = 0
df['X'] = np.nan_to_num(df['X'])
df['Y'] = np.log(df["Active"])
df[df['Y'] == -inf] = 0
df['Y'] = np.nan_to_num(df['Y'])
dataset = df[['X','Y']]
fig18 = px.scatter(x = dataset.Y, y = dataset.X,trendline = "ols")
fig18.update_layout(
    title="Area vs Log(Active)",
    xaxis_title="log(Active)",
    yaxis_title="Area",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)
fig18.show()

df['X'] = (df["Density"])
df[df['X'] == -inf] = 0
df['X'] = np.nan_to_num(df['X'])
df['Y'] = np.log(df["Active"])
df[df['Y'] == -inf] = 0
df['Y'] = np.nan_to_num(df['Y'])
dataset = df[['X','Y']]
fig19 = px.scatter(x = dataset.Y, y = dataset.X,trendline = "ols")
fig19.update_layout(
    title="Density vs Log(Active)",
    xaxis_title="log(Active)",
    yaxis_title="Density",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)
fig19.show()

df['X'] = np.log(df["Density"])
df[df['X'] == -inf] = 0
df['X'] = np.nan_to_num(df['X'])
df['Y'] = np.log(df["Active"])
df[df['Y'] == -inf] = 0
df['Y'] = np.nan_to_num(df['Y'])
dataset = df[['X','Y']]
fig20 = px.scatter(x = dataset.Y, y = dataset.X,trendline = "ols")
fig20.update_layout(
    title="Log(Density) vs Log(Active)",
    xaxis_title="log(Active)",
    yaxis_title="Log(Density)",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)
fig20.show()

df['X'] = (df["Population"])
df[df['X'] == -inf] = 0
df['X'] = np.nan_to_num(df['X'])
df['Y'] = np.log(df["Active"])
df[df['Y'] == -inf] = 0
df['Y'] = np.nan_to_num(df['Y'])
dataset = df[['X','Y']]
fig21 = px.scatter(x = dataset.Y, y = dataset.X,trendline = "ols")
fig21.update_layout(
    title="Population vs Log(Active)",
    xaxis_title="log(Active)",
    yaxis_title="Population",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)
fig21.show()

df['X'] = np.log(df["Residential"])
df[df['X'] == -inf] = 0
df['X'] = np.nan_to_num(df['X'])
df['Y'] = np.log(df["Active"])
df[df['Y'] == -inf] = 0
df['Y'] = np.nan_to_num(df['Y'])
dataset = df[['X','Y']]
fig22 = px.scatter(x = dataset.Y, y = dataset.X,trendline = "ols")
fig22.update_layout(
    title="log(residential mobility percentage) vs Log(Active)",
    xaxis_title="log(Active)",
    yaxis_title="log(Residential mobility percentage)",
    font=dict(
        family="Courier New, monospace",
        size=14,
        color="#7f7f7f"
    )
)
fig22.show()

from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app.layout = html.Div(
        html.Div([

            html.Div(
                html.H1('Dependency plots')
            ),

            html.Div(
                dcc.Dropdown(
                        options=[
                                 {'label': 'Correlation Heatmap', 'value': 1},
                            {'label':'Log(Density) vs Log(Growth Ratio)', 'value': 2},
                            {'label':'Log(Population) vs Log(Growth Ratio)', 'value': 3},
                            {'label':'Log(Area) vs Log(Growth Ratio)', 'value': 4},
                            {'label':'Log(Air quality) vs Log(Growth Ratio)', 'value': 5},
                            {'label':'Log(Water Accessibility) vs Log(Growth Ratio)', 'value': 6},
                            {'label':'log(Confirmed) vs Log(Growth Ratio)', 'value': 7},
                            {'label':'Log(Active) vs Log(Growth Ratio)', 'value': 8},
                            {'label':'Population vs Log(Confirmed)', 'value': 9},
                            {'label':'log(Density) vs Log(Confirmed)', 'value': 10},
                            {'label':'Area vs Log(Confirmed)', 'value': 11},
                            {'label':'Air quality vs Log(Confirmed)', 'value': 12},
                            {'label':'Water accessibility vs Log(Confirmed)', 'value': 13},
                            {'label':'log(Active) vs Log(Confirmed)', 'value': 14},
                            {'label':'Growth Ratio vs Log(Active)', 'value': 15},
                            {'label':'Water accessibility vs Log(Active)', 'value': 16},
                            {'label':'Air quality vs Log(Active)', 'value': 17},
                            {'label':'Area vs Log(Active)', 'value': 18},
                            {'label':'Density vs Log(Active)', 'value': 19},
                            {'label':'Log(Density) vs Log(Active)', 'value': 20},
                            {'label':'Population vs Log(Active)', 'value': 21},
                            {'label':'log(residential mobility percentage) vs Log(Active)', 'value': 22}

                        ],
                    id = 'dropdown',
                    placeholder="Select your option",

                ),
                style = {
                        'width' : '40%'
                }
            ),

            html.Div(
                html.Div(id='dropdown-output-1'),
            ),

            html.Div(
                html.Div(id='dropdown-output-2'),
            ),
            html.Div(
                html.Div(id='dropdown-output-3'),
            ),
            html.Div(
                html.Div(id='dropdown-output-4'),
            ),
            html.Div(
                html.Div(id='dropdown-output-5'),
            ),
            html.Div(
                html.Div(id='dropdown-output-6'),
            ),
            html.Div(
                html.Div(id='dropdown-output-7'),
            ),
            html.Div(
                html.Div(id='dropdown-output-8'),
            ),
            html.Div(
                html.Div(id='dropdown-output-9'),
            ),
            html.Div(
                html.Div(id='dropdown-output-10'),
            ),
            html.Div(
                html.Div(id='dropdown-output-11'),
            ),
            html.Div(
                html.Div(id='dropdown-output-12'),
            ),
            html.Div(
                html.Div(id='dropdown-output-13'),
            ),
            html.Div(
                html.Div(id='dropdown-output-14'),
            ),
            html.Div(
                html.Div(id='dropdown-output-15'),
            ),
            html.Div(
                html.Div(id='dropdown-output-16'),
            ),
            html.Div(
                html.Div(id='dropdown-output-17'),
            ),
            html.Div(
                html.Div(id='dropdown-output-18'),
            ),
            html.Div(
                html.Div(id='dropdown-output-19'),
            ),
            html.Div(
                html.Div(id='dropdown-output-20'),
            ),
            html.Div(
                html.Div(id='dropdown-output-21'),
            ),
            html.Div(
                html.Div(id='dropdown-output-22'),
            )
            ])
    )  
@app.callback(Output('dropdown-output-1', 'children'), [Input('dropdown', 'value')])
def display_content(value=1):
   if(value==1):
    return html.Div([dcc.Graph(figure = fig1),html.P("This is a correlation heatmap showing the correlation coefficient between the different Human Factors")])
@app.callback(Output('dropdown-output-2', 'children'), [Input('dropdown', 'value')])
def display_content(value=2):
   if(value==2):
    return html.Div([dcc.Graph(figure = fig2),html.P("This is a regression plot between log(Density) and log(Growth Ratio), showing how the growth ratio depends on the population density. Since population density of a place is comparatively larger than the growth ratio of COVID-19, we take log() of the parameters to normalize the data and get a good regression plot.")])
@app.callback(Output('dropdown-output-3', 'children'), [Input('dropdown', 'value')])
def display_content(value=3):
   if(value==3):
    return html.Div([dcc.Graph(figure = fig3),html.P("This is a regression plot between log(Population) and log(Growth Ratio), showing how the growth ratio depends on population count. Since population count is way too big compared to growth ratio of COVID-19, we take the log() of the parameters to normalize the data and get a good regression plot.")])
@app.callback(Output('dropdown-output-4', 'children'), [Input('dropdown', 'value')])
def display_content(value=4):
   if(value==4):
    return html.Div([dcc.Graph(figure = fig4),html.P("This is a regression plot between log(Area) and log(Growth Ratio), showing how the growth ratio depends on area of a District. Since area is way too big compared to growth ratio, we take log() of the parameters to normalize the data and get a good regresion plot.")])
@app.callback(Output('dropdown-output-5', 'children'), [Input('dropdown', 'value')])
def display_content(value=5):
   if(value==5):
    return html.Div([dcc.Graph(figure = fig5),html.P("This is a regression plot between log(Air quality) and log(Growth ratio), showing how the growth rate depends on the air quality index of a district. Since air quality index is comparatively larger than growth ratio, we take log() of the parameters for a good regression plot. ")])
@app.callback(Output('dropdown-output-6', 'children'), [Input('dropdown', 'value')])
def display_content(value=6):
   if(value==6):
    return html.Div([dcc.Graph(figure = fig6),html.P("This is a regression plot between log(water accessibility) and log(growth ratio), showing how the growth ratio depends on the potable water accessibility percentage. Since potable water accessibility percentage is comparatively higher than growth ratio, we take log() of the parameters for a good regression plot. ")])
@app.callback(Output('dropdown-output-7', 'children'), [Input('dropdown', 'value')])
def display_content(value=7):
   if(value==7):
    return html.Div([dcc.Graph(figure = fig7),html.P("This is a regression plot between log(Confirmed) and log(growth ratio) to show the dependency of the number of Confirmed cases and growth ratio.")])
@app.callback(Output('dropdown-output-8', 'children'), [Input('dropdown', 'value')])
def display_content(value=8):
   if(value==8):
    return html.Div([dcc.Graph(figure = fig8),html.P("This is a regression plot between log(Active) and log(growth ratio) to show how the number of active cases depends on the growth ratio.")])
@app.callback(Output('dropdown-output-9', 'children'), [Input('dropdown', 'value')])
def display_content(value=9):
   if(value==9):
    return html.Div([dcc.Graph(figure = fig9),html.P("This is a linear-log regression between population count of a district and confirmed cases of COVID-19. This graph shows how the factor population size affects the daily number of confirmed cases.")])
@app.callback(Output('dropdown-output-10', 'children'), [Input('dropdown', 'value')])
def display_content(value=10):
   if(value==10):
    return html.Div([dcc.Graph(figure = fig10),html.P("This is a regression plot between log(Density) and log(Confirmed), to show how the population density of districts affects the number of confirmed cases. Since population density is comparatively large compared to number of Confirmed cases, we take log() of the parameters in order to normalise tha data for a good regression plot.")])
@app.callback(Output('dropdown-output-11', 'children'), [Input('dropdown', 'value')])
def display_content(value=11):
   if(value==11):
    return html.Div([dcc.Graph(figure = fig11),html.P("This is a linear-log regression plot between area and confirmed cases, to show how the area of any district affects the number of confirmed cases.")])
@app.callback(Output('dropdown-output-12', 'children'), [Input('dropdown', 'value')])
def display_content(value=12):
   if(value==12):
    return html.Div([dcc.Graph(figure = fig12),html.P("This is a linear-log regression plot between air quality index and confirmed cases, showing how the number of confirmed cases depends with the air quality index of a district.")])
@app.callback(Output('dropdown-output-13', 'children'), [Input('dropdown', 'value')])
def display_content(value=13):
   if(value==13):
    return html.Div([dcc.Graph(figure = fig13),html.P("This ia a linear-log regression plot between potable water accessibility and confirmed cases, showing how the number of confirmed cases dpeneds with potable water accsesbility precentage.")])
@app.callback(Output('dropdown-output-14', 'children'), [Input('dropdown', 'value')])
def display_content(value=14):
   if(value==14):
    return html.Div([dcc.Graph(figure = fig14),html.P("This is a regresssion plot showing the very basic relationship between the active cases and the confirmed cases. Since the number of active and confirmed cases are different for different district we normalize the data by taking log() of the parameters to get a good regression graph.")])
@app.callback(Output('dropdown-output-15', 'children'), [Input('dropdown', 'value')])
def display_content(value=15):
   if(value==15):
    return html.Div([dcc.Graph(figure = fig15),html.P("This is a linear-log regression plot between growth ratio and active cases, to show how the growth ratio depends on the active cases. Since number of active cases is comparatively larger than the growth ratio, we take log() of active cases.")])
@app.callback(Output('dropdown-output-16', 'children'), [Input('dropdown', 'value')])
def display_content(value=16):
   if(value==16):
    return html.Div([dcc.Graph(figure = fig16),html.P("This is a linear-log regression plot between potable water accessibility and number of active cases, to show how potable water accessibility affects the number of active cases in a district.")])
@app.callback(Output('dropdown-output-17', 'children'), [Input('dropdown', 'value')])
def display_content(value=17):
   if(value==17):
    return html.Div([dcc.Graph(figure = fig17),html.P("This is a linear-log regression plot between air quality index and number of active cases to show how the number of active cases depends on the air quality of a district.")])
@app.callback(Output('dropdown-output-18', 'children'), [Input('dropdown', 'value')])
def display_content(value=18):
   if(value==18):
    return html.Div([dcc.Graph(figure = fig18),html.P("This is a linear-log regression plot between area and active cases to show how the number of active cases depends upon the area.")])
@app.callback(Output('dropdown-output-19', 'children'), [Input('dropdown', 'value')])
def display_content(value=19):
   if(value==19):
    return html.Div([dcc.Graph(figure = fig19),html.P("This is a linear-log regression between population density and active cases.")])
@app.callback(Output('dropdown-output-20', 'children'), [Input('dropdown', 'value')])
def display_content(value=20):
   if(value==20):
    return html.Div([dcc.Graph(figure = fig20),html.P("This is a regression plot between log(Population density) an log(Active cases). this plot shows how the trend of the number of active cases is affected by the Population density of a District. Since population density is way to big compared to number of active cases, we take log() of the parameters to get a good regression plot.")])
@app.callback(Output('dropdown-output-21', 'children'), [Input('dropdown', 'value')])
def display_content(value=21):
   if(value==21):
    return html.Div([dcc.Graph(figure = fig21),html.P("This is a linear-log rgression between population count and number of active cases, to show how the trend of the number of active cases is affected by the popultaion count of a district.")])
@app.callback(Output('dropdown-output-22', 'children'), [Input('dropdown', 'value')])
def display_content(value=22):
    if(value==22):
     return html.Div([dcc.Graph(figure = fig22),html.P('This is a regression plot between log(residential mobility percentage) and log(active cases). This plot simply shows how staying at home affects the trend in the rise of active cases.')])
if __name__ == '__main__':
    app.run_server(debug=True)