import os
import sys
import dash
import dash_core_components as dcc
import dash_html_components as html

from calculator import *

rid_path = r'rm.csv'
room_df = pd.read_csv(rid_path)
rooms = []
for rid in room_df['Room']:
    rooms.append({'label': rid, 'value': rid})

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server
activities = [{'label':'Lecture', 'value':'Lecture'}, {'label':'Studying', 'value':'Studying'}, {'label':'Singing', 'value':'Singing'}, {'label':'Social Event', 'value':'Social Event'}, {'label':'Exercising', 'value':'Exercising'}]
server = app.server
activities = [{'label':'Lecture', 'value':'Lecture'}, {'label':'Studying', 'value':'Studying'}, {'label':'Singing', 'value':'Singing'}, {'label':'Social', 'value':'Social'}, {'label':'Exercising', 'value':'Exercising'}]
app.layout = html.Div([
    html.H6("Event Information"),
    html.Div(["RoomID: ",
              dcc.Dropdown(id='room-dropdown', value='test', options=rooms)]), #placeholder rooms
    html.Br(),
    html.Div(["Duration of Event (min): ",
              dcc.Input(id='time-input', value = 0, type='number')]),
    html.Br(),
    html.Div(["Number of Occupants: ",
              dcc.Input(id='occupant-input', value = 0, type='number')]),
    html.Br(),
    html.Div(["Activity: ",
              dcc.Dropdown(id='activity-dropdown', value ='test', options=activities)]),
    html.Br(),
    html.Br(),
    html.Div(["Masks: ",
              dcc.RadioItems(id = 'masks-radio', value = 0,
    options=[
        {'label': 'Masks', 'value': 1},
        {'label': 'No Masks', 'value': 0},
    ], labelStyle={'display': 'inline-block'}
)  ]),
    html.Button('Reset', id='reset-button'),
    html.Br(),
    html.Button('Go', id = 'go-button', n_clicks = 0),
    html.Br(),
    html.Div(id = 'calc-output', children = 'Enter values to calculate risk')

])

@app.callback(
    dash.dependencies.Output('calc-output', 'children'),
    [dash.dependencies.Input('go-button', 'n_clicks')],
    [dash.dependencies.Input('masks-radio', 'value')],
    [dash.dependencies.Input('activity-dropdown', 'value')],
    [dash.dependencies.Input('room-dropdown', 'value')],
    [dash.dependencies.Input('vav-dropdown', 'value')],
    [dash.dependencies.State('time-input', 'value')],
    [dash.dependencies.State('occupant-input', 'value')]
)
def update_calc(n_clicks, activity_dropdown, room_input, time_input, occupant_input):
    if n_clicks >= 1:
        comp_ir = ui_calc(activity_dropdown, room_input, time_input, occupant_input, rid_path)
        total_inf = int(occupant_input * comp_ir) 
        # Update return according to professor
        to_return = 'The risk of holding a(n) {} event for {} minutes in {} is {}%, given the most recent infection rates. With {} occupants, it is likely that {} occupant(s) will be infected.'.format(activity_dropdown, 
                                                                                                                                time_input, 
                                                                                                                                room_input, 
                                                                                                                                round((comp_ir * 100),2), 
                                                                                                                                occupant_input,
                                                                                                                                total_inf)
        return to_return
    else:
        return 'Enter Values to get risk calculation'

if __name__ == '__main__':
    app.run_server(debug=True)
