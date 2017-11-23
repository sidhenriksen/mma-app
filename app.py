import numpy as np
import dash,pickle,requests
import dash_core_components as dcc
import dash_html_components as html

import mltools
from mltools import FightClassifier,Transformer,CustomScaler

app = dash.Dash()

app.suppress_callback_exceptions=True
app.config.update(
    {'routes_pathname_prefix':'',
     'requests_pathname_prefix':''})
server = app.server

app.css.append_css({
   'external_url': (
       'app-style.css'
   )
})


attributeList = ['wins','losses','dob','stance','height','weight','reach']

with open('classifier.save','rb') as f:
    comps = pickle.load(f)

classifier = FightClassifier(classifier=comps['classifier'])

#classifier = mltools.train_classifier()

fighterList = np.sort(list(classifier.fighters.keys()))
def generate_stats_table(fighter):

    fighter = mltools.strip_name(fighter)
    stats=classifier.fighters[fighter]
    units = {'height':'cm','weight':'kg','reach':'cm'}
    stringFormat = {}
    for key in attributeList:
        if key not in ['wins','losses','cumtime','dob','stance']:
            sf = r'%.2f'
        elif key in ['wins','losses','cumtime']:
            sf = r'%i'
        else:
            unit = units[key] if key in units else ''
            sf = r'%s'+unit

        stringFormat[key] = sf
                        

    stats = {key:value for key,value in stats.items() if \
                 key in attributeList}
    tableHeaders = [
        html.Tr([
            html.Th('Attribute'),
            html.Th('Value')
            ])
    ]

    tdCss = {'width':'45%','vertical-align':'top','text-align':'left'}
    tableContent = [
        html.Tr([
            html.Td(attribute.capitalize(),style=tdCss),
            html.Td(stringFormat[attribute]%stats[attribute],style=tdCss)
        ]) for attribute in attributeList
    ]

    return tableContent



layout = html.Div([
    html.Div([
        ## Fighter 1
        html.Div([
            html.Label('Fighter 1: '),
            dcc.Dropdown(id='fighter-1',
                             options = [{'label':classifier.fighters[k]['name'],
                                         'value': k} for k in fighterList],
                             value='georgesstpierre'),
            html.Img(width=150,id='pic-f1'),
            html.Table(id='stats-table-1')
        ],style={'border':'1px solid black;'}),
        
        ## Fighter 2
        html.Div([
            html.Label('Fighter 2: '),
            dcc.Dropdown(id='fighter-2',
                             options = [{'label':classifier.fighters[k]['name'],
                                         'value': k} for k in fighterList],
                             value='michaelbisping'),
                         
            html.Img(width=150,id='pic-f2'),
            html.Table(id='stats-table-2')
        ])

        ],style={'columnCount':2}),
        
    html.Div([
        html.P(id='paragraph-outcome')
    ])
])

app.layout = layout


@app.callback(
    dash.dependencies.Output('stats-table-1','children'),
    [dash.dependencies.Input('fighter-1','value')])
def update_stats_table1(fighter):
    return generate_stats_table(fighter)

@app.callback(
    dash.dependencies.Output('stats-table-2','children'),
    [dash.dependencies.Input('fighter-2','value')])
def update_stats_table2(fighter):
    return generate_stats_table(fighter)

@app.callback(
    dash.dependencies.Output('paragraph-outcome','children'),
    [dash.dependencies.Input('fighter-1','value'),
     dash.dependencies.Input('fighter-2','value')])
def update_outcome(fighter1,fighter2):

    
    p = classifier.predict_fight(fighter1,fighter2)*100
    
    winner = fighter1 if p < 50 else fighter2
    displayWinner = classifier.fighters[winner]['name']
    p = 100-p if p < 50 else p
    
    v = '%.2f'%p +'% chance that {} wins'.format(displayWinner)

    return v

@app.callback(
    dash.dependencies.Output('pic-f1','src'),
    [dash.dependencies.Input('fighter-1','value')])
def update_pic1(fighter):

    picUrl = 'https://sidhenriksen.com/mma/images/%s.png'%fighter

    request = requests.get(picUrl)
    if request.status_code == 404:
        picUrl = 'https://sidhenriksen.com/mma/images/unknown.png'

    return picUrl

@app.callback(
    dash.dependencies.Output('pic-f2','src'),
    [dash.dependencies.Input('fighter-2','value')])
def update_pic2(fighter):

    picUrl = 'https://sidhenriksen.com/mma/images/%s.png'%fighter

    request = requests.get(picUrl)
    if request.status_code == 404:
        picUrl = 'https://sidhenriksen.com/mma/images/unknown.png'

    return picUrl
        
    
if __name__ == "__main__":

    app.run_server(debug=True)
