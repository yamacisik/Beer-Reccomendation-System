import dash
import dash_core_components as dcc
import dash_html_components as html
import pickle
import numpy as np
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

beers = pickle.load( open( "beers.pickle", "rb" ) )
available_beers=beers.values()
beers2 = {v: k for k, v in beers.items()}


Q=np.load("Q.npy")
P=np.load("P.npy")



def add_user(Q,P,user,lr=0.007,beta=0.001,epochs=10000):
    user_hat=np.zeros(len(user))
    n,m=Q.shape
    Q_n=np.random.rand(m)
    z_i=np.where(user>0)
    for epoch in range(epochs):
        for u in range(len(z_i)):
            i=z_i[0][u]
            Q_n=Q_n+(2*lr*(user[i]-np.dot(Q_n,P[:,i]))*P[:,i]).T-beta*Q_n
    for i in range(len(P.T)):
        user_hat[i]=np.dot(Q_n,P.T[i])   
    return user_hat


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Dropdown(
                id='beer1',
                options=[{'label': i, 'value': i} for i in available_beers],
                value='Two Hearted Ale by Bell\'s Brewery - Eccentric Café & General Store'
    ),
    html.Div(id='output-a'),

    dcc.RadioItems(
                id='beer1_rating',
                options=[{'label': i, 'value': i} for i in ['1', '2','3','4','5']],
                value='3',
                labelStyle={'display': 'inline-block'}
            ),


    dcc.Dropdown(
                id='beer2',
                options=[{'label': i, 'value': i} for i in available_beers],
                value='Two Hearted Ale by Bell\'s Brewery - Eccentric Café & General Store'
    ),
    html.Div(id='output-b'),

    dcc.RadioItems(
                id='beer2_rating',
                options=[{'label': i, 'value': i} for i in ['1', '2','3','4','5']],
                value='3',
                labelStyle={'display': 'inline-block'}
            ),

    html.Div(id='output-c'),
   
   
])


@app.callback(
    dash.dependencies.Output('output-a', 'children'),
    [dash.dependencies.Input('beer1', 'value')],)
def callback_a(dropdown_value):
    return 'Please indicate your rating for "{}"'.format(dropdown_value)

@app.callback(
    dash.dependencies.Output('output-b', 'children'),
    [dash.dependencies.Input('beer2', 'value')])
def callback_b(dropdown_value):
    return 'Please indicate your rating for "{}"'.format(dropdown_value)

@app.callback(
    dash.dependencies.Output('output-c', 'children'),
    [dash.dependencies.Input('beer1', 'value'),dash.dependencies.Input('beer1_rating', 'value'),
    dash.dependencies.Input('beer2', 'value'),dash.dependencies.Input('beer2_rating', 'value')])
def callback_c(beer1,beer1_rating,beer2,beer2_rating):
    user=np.zeros(P.shape[1])
    user[beers2[beer1]]=beer1_rating
    user[beers2[beer2]]=beer2_rating
    your_rating=add_user(Q,P,user,lr=0.007,beta=0.001,epochs=1000)
    preference=(-your_rating).argsort()[:10]
    final=[]
    for i in preference:
        if i != beers2[beer1] and beers2[beer2]:
            final.append(i)
    final=np.array(final)
    res=np.random.choice(final)
    return 'Based on your choices you would also like "{}"'.format(beers[res])






if __name__ == '__main__':
    app.run_server(debug=True)