import pandas as pd
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px
from dash.dependencies import Input, Output
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import shap
from scipy.stats import chi2
from factor_analyzer import FactorAnalyzer



# ----------------- Lectura de Datos ----------------------
df= pd.read_csv("diabetes.csv") 
df_descripcion= df.describe().round(3)


scaler = StandardScaler()
X_km_stand = scaler.fit_transform(df[['altura','peso','IMC','cintura','caderas','prop_cin_cad']])
X_km_stand = pd.DataFrame(X_km_stand)
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_km_stand)
df["KmeansCorporales"]=kmeans.labels_
df['KmeansCorporales'].replace({0:"Grupo 1", 1:"Grupo 2"}, inplace=True)

scaler = StandardScaler()
X_km_stand = scaler.fit_transform(df[['colesterol','glucosa','hdl_chol','prop_col_hdl','ps_sistolica','ps_diastolica']])
X_km_stand = pd.DataFrame(X_km_stand)
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_km_stand)
df["KmeansSanguineas"]=kmeans.labels_
df['KmeansSanguineas'].replace({0:"Grupo 1", 1:"Grupo 2"}, inplace=True)

df.replace({"female":0, "male": 1}, inplace=True)
X = df.drop(["diabetes", "KmeansCorporales", "KmeansSanguineas"], axis=1)



# ----------------- Diccionario Colores --------------------

colorPrimary= "#4c9be8"
colorBackground= "#0f2537"

# ---------------------- Gráficos --------------------------

figCorr= px.imshow(df.replace({"Diabetes":1,"No_diabetes":0}).corr(), range_color=[-1,1],
               color_continuous_scale=px.colors.sequential.RdBu, 
               height=600)


figCorr.update_layout(paper_bgcolor="#0f2537", font_color="white")



figScatterMatrix = px.scatter_matrix(df, color="diabetes",
                                     dimensions= ['colesterol', 'glucosa', 'hdl_chol', 'prop_col_hdl', 'edad',
                                                  'altura', 'peso', 'IMC', 'ps_sistolica', 'ps_diastolica', 'cintura',
                                                  'caderas', 'prop_cin_cad'],
                                     height=1300)

figScatterMatrix.update_layout(paper_bgcolor="#0f2537", font_color="white")



figScatterMatrixCorp= px.scatter_matrix(df, dimensions=['altura','peso','IMC','cintura','caderas','prop_cin_cad'],
                        color="KmeansCorporales", height=800,width=1400)

figScatterMatrixCorp.update_layout(paper_bgcolor="#0f2537", font_color="white")

figSunburstCorp= px.sunburst(df, path=['diabetes', 'KmeansCorporales'],
                  width=500, height=500)

figSunburstCorp.update_layout(paper_bgcolor="#0f2537", font_color="white")


figScatterMatrixSang= px.scatter_matrix(df, dimensions=['colesterol','glucosa','hdl_chol','prop_col_hdl','ps_sistolica','ps_diastolica'],
                        color="KmeansSanguineas", height=800,width=1400)

figScatterMatrixSang.update_layout(paper_bgcolor="#0f2537", font_color="white")

figSunburstSang= px.sunburst(df, path=['diabetes', 'KmeansSanguineas'],
                  width=500, height=500)

figSunburstSang.update_layout(paper_bgcolor="#0f2537", font_color="white")



# -------------------- Aplicación ------------------------

app = dash.Dash(external_stylesheets=[dbc.themes.SUPERHERO])
server= app.server

app.layout = html.Div(children=[
    
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Adrián Landaverde Nava", href="#")),
            dbc.NavItem(dbc.NavLink("Jesús Rojas Reyes", href="#")),
            dbc.NavItem(dbc.NavLink("Naomi Padilla Mora", href="#")),
            dbc.NavItem(dbc.NavLink("Sabrina Nicole Rodriguez Salgado", href="#")),
            ],
        
        brand="Adelanto Reto",
        brand_href="#",
        color="primary",
        dark=True,
        fluid=True),
    
    html.Br(),
    
    dbc.Col([
        
        dbc.Tabs([
            dbc.Tab([
                
                html.Br(),
                
                    
        
                html.H2("Estadística Descriptiva de las variables"),
                
                dbc.Table.from_dataframe(df_descripcion, striped=True, bordered=True, 
                                         hover=True,index=True, dark=True),
                
                html.Br(),
                
                dbc.Row([dbc.Col([html.H3("Gráfico de Correlación"),
                        dcc.Graph(id="Correlacion", figure=figCorr)], width=5),
                         
                         dbc.Col([html.H4("Boxplot por Variable"),
                             dcc.Dropdown(id='var_hist',
                                   options=['colesterol', 'glucosa', 'hdl_chol', 'prop_col_hdl', 'edad',
                                            'altura', 'peso', 'IMC', 'ps_sistolica', 'ps_diastolica', 'cintura',
                                            'caderas', 'prop_cin_cad'],
                                   value='colesterol',
                                   style={"color":"black"}
                                ),
                                  dcc.Graph(id="Histograma"),
                                  ],width=5)
                
                         ]),
                
                html.Br(),
                
                html.H3("Matriz de dispersión de los Datos"),
                
                dcc.Graph(id="ScatterMatrix", figure=figScatterMatrix)
                ], label="Estadística Descriptiva",
                style={'margin-right': '5px', 'margin-left': '5px'}),
            
            dbc.Tab([
                html.Br(),
                dbc.Accordion([
                dbc.AccordionItem([
                    
                    html.H3("Matriz de Dispersión de las variables corpales agrupadas"),
                    
                    dcc.Graph(id="ClusterMatrix", figure=figScatterMatrixCorp),
                    
                    html.H3("Comparación de Clustering con Clasificación"),
                    
                    dcc.Graph(id="ClusterSunburst", figure=figSunburstCorp)
                    ], title="Variables Corporales"),
                
                dbc.AccordionItem([
                    
                    html.H3("Matriz de Dispersión de las variables sanguíneas agrupadas"),
                    
                    dcc.Graph(id="ClusterMatrix2", figure=figScatterMatrixSang),
                    
                    html.H3("Comparación de Clustering con Clasificación"),
                    
                    dcc.Graph(id="ClusterSunburst2", figure=figSunburstSang)
                    ], title="Variables Sanguíneas")
                
                ])], 
                label="Clustering",
                style={'margin-right': '5px', 'margin-left': '5px'}),
            
            dbc.Tab([
                html.Br(),
                
                dbc.Row([

                    dbc.Col(html.H3("Selección de Variables para Reducción:"), width=5),

                    dbc.Col(dcc.Dropdown(
                    options=['colesterol', 'glucosa', 'hdl_chol', 'prop_col_hdl', 'edad',
                                            'altura', 'peso', 'IMC', 'ps_sistolica', 'ps_diastolica', 'cintura',
                                            'caderas', 'prop_cin_cad', 'diabetes'],
                    value=['colesterol','glucosa', 'hdl_chol','edad','IMC',
                            'ps_sistolica', 'ps_diastolica', 'prop_cin_cad'],
                            style={"color":"black", "width":"80%"},multi=True, id="Dropdown_PCAVar", 
                ), width=7)
                ]),
                

                html.Br(),
                
                html.H4("Varianza explicada por cada componente"),
                
                dcc.Graph(id="Varianza"),

                dbc.Row([
                    
                    dbc.Col(html.H3("Selección de Cantidad de Componentes:"), width=5),
                
                    dbc.Col(dcc.Dropdown(
                        options=list(np.arange(2,15)),value=4,
                        style={"color":"black", "width":"30%"}, id="Dropdown_PCACom", ), width=2)]), 

                html.Br(),
                

                html.H4("Matriz de dispersión de los datos"),
                
                dcc.Graph(id="MatrixPCA"),

                html.H3("Graficación de los componentes"),

                dbc.Row([

                    dbc.Col(html.H4("Primera componente"), width=3),

                    dbc.Col(dcc.Dropdown(id='dd_componente1',
                                   options=['Componente 1','Componente 2','Componente 3','Componente 4',
                                   'Componente 5','Componente 6','Componente 7','Componente 8','Componente 9',
                                   'Componente 10','Componente 11','Componente 12','Componente 13','Componente 14'],
                                   value='Componente 1',
                                   style={"color":"black"}
                                ), width=3),

                    dbc.Col(html.H4("Segunda componente"), width=3),

                    dbc.Col(dcc.Dropdown(id='dd_componente2',
                                   options=['Componente 1','Componente 2','Componente 3','Componente 4',
                                   'Componente 5','Componente 6','Componente 7','Componente 8','Componente 9',
                                   'Componente 10','Componente 11','Componente 12','Componente 13','Componente 14'],
                                   value='Componente 2',
                                   style={"color":"black"}
                                ), width=3),
                ]),

                html.Br(),

                dbc.Row([

                    dbc.Col([

                        html.H4("Score Plot"),

                        dcc.Graph(id="ScorePlot"),

                        html.H4("Biplot"),

                        dcc.Graph(id="Biplot")

                    ], width=6),

                    dbc.Col([

                        html.H4("Loading Plot"),

                        dcc.Graph(id="LoadingPlot"), 

                        html.H4("Outlier Plot"),

                        dcc.Graph(id="MahalanobisPlot")

                    ], width=6)

                ]),

                html.H3("Análisis factorial"),

                html.Br(),

                dbc.Row([
                    dbc.Col(width=1),

                    dbc.Col(html.Div(id='TablaFactor'),width=10),

                    dbc.Col(width=1)
                ])

                
                ], label="Reducción de Dimensiones"),
            
            dbc.Tab([
                
                html.Br(),

                html.H3("Variables a usar para el modelo"),

                dcc.Dropdown(
                    options=['colesterol', 'glucosa', 'hdl_chol', 'prop_col_hdl', 'edad',
                                            'altura', 'peso', 'IMC', 'ps_sistolica', 'ps_diastolica', 'cintura',
                                            'caderas', 'prop_cin_cad'],
                    value=["glucosa","cintura","prop_cin_cad","IMC","peso","prop_col_hdl"],
                            style={"color":"black", "width":"80%"},multi=True, id="Dropdown_MLVars"),

                html.Br(),

                dbc.Row([

                    dbc.Col([

                        html.H3("Matriz de Confusión Entrenamiento"),

                        dcc.Graph(id="ConfusionTrain"),

                        html.H3("Valores SHAP"),

                        dbc.Row([
                            
                            dbc.Col(html.H4("Número de Dato:"), width=4),

                            dbc.Col(dcc.Dropdown(
                                options=list(np.arange(0,len(df))),value=1,
                                style={"color":"black"}, id="Dropdown_shapN"),width=2)
                                
                                ]),

                        dcc.Graph(id="waterfall")
                    ], width=6),

                    dbc.Col([

                        html.H3("Matriz de Confusión Validación"),

                        dcc.Graph(id="ConfusionTest")

                    ], width=6)
                ])
                
                
                
                ], label="Modelo Machine Learning")
            ])
        
    
    
    ],style={'margin-right': '20px', 'margin-left': '20px'}),
    

])

# -------------------- Interactividad ------------------------

@app.callback(Output('Histograma', 'figure'),
              [Input('var_hist', 'value')])
def histograma_variables(variable):
    figHist= px.box(df, x=str(variable), color="diabetes",
             width= 800, height=600)
    figHist.update_layout(paper_bgcolor="#0f2537", font_color="white")

    return figHist

@app.callback(Output('Varianza', 'figure'),
              [Input('Dropdown_PCAVar', 'value')])
def varianza_pca(variables):
    X= df[variables].replace({"No_diabetes":0,"Diabetes":1})
    scaler = StandardScaler()
    X_stand= scaler.fit_transform(X)
    pcaCompleto = PCA(n_components=len(variables))
    pcaCompleto.fit(X_stand)
    figVarianzaPCA= px.bar(x=np.arange(1,len(variables)+1), 
                            y = np.round(np.cumsum(pcaCompleto.explained_variance_ratio_)*100,2),
                            labels={"x":"Componente", "y":"Varianza Explicada"},
                            text_auto=True,
                            height=500)
    figVarianzaPCA.update_layout(paper_bgcolor="#0f2537", font_color="white")
    return(figVarianzaPCA)

@app.callback(Output('MatrixPCA', 'figure'),
              [Input('Dropdown_PCAVar', 'value'),Input('Dropdown_PCACom', 'value')])
def matrix_PCA(variables,componentes):
    X=df[variables].replace({"No_diabetes":0,"Diabetes":1})
    scaler = StandardScaler()
    X_stand= scaler.fit_transform(X)
    pca = PCA(n_components=componentes)
    numeros= pd.Series(np.arange(1,componentes+1).astype("str"))
    nombres= pd.Series(["Componente "]*componentes)
    columnas= list(nombres+numeros)
    x_PCA = pca.fit_transform(X_stand)
    x_PCA = pd.DataFrame(x_PCA, columns=columnas)
    x_PCA["diabetes"] = df.diabetes
    figPCA = px.scatter_matrix(x_PCA, color="diabetes", dimensions=columnas ,
                            height=800)
    figPCA.update_layout(paper_bgcolor="#0f2537", font_color="white")
    return(figPCA)

@app.callback(Output('ScorePlot', 'figure'),
              [Input('Dropdown_PCAVar', 'value'),Input('Dropdown_PCACom', 'value'),
              Input('dd_componente1','value'), Input('dd_componente2','value')])
def scorePlot_PCA(variables,componentes, componente1, componente2):
    X=df[variables].replace({"No_diabetes":0,"Diabetes":1})
    scaler = StandardScaler()
    X_stand= scaler.fit_transform(X)
    pca = PCA(n_components=componentes)
    numeros= pd.Series(np.arange(1,componentes+1).astype("str"))
    nombres= pd.Series(["Componente "]*componentes)
    columnas= list(nombres+numeros)
    x_PCA = pca.fit_transform(X_stand)
    x_PCA = pd.DataFrame(x_PCA, columns=columnas)
    df_junto= pd.concat([df, x_PCA], axis=1)
    df_junto.reset_index(inplace=True)
    variables.append("index")
    print(df_junto.columns)
    print(variables)
    figComp = px.scatter(df_junto, x=componente1, y=componente2, hover_data=variables, color="diabetes")
    figComp.update_layout(paper_bgcolor="#0f2537", font_color="white")
    return(figComp)


@app.callback(Output('LoadingPlot', 'figure'),
              [Input('Dropdown_PCAVar', 'value'),Input('Dropdown_PCACom', 'value'),
              Input('dd_componente1','value'), Input('dd_componente2','value')])
def loading_PCA(variables,componentes, componente1, componente2):
    X=df[variables].replace({"No_diabetes":0,"Diabetes":1})
    scaler = StandardScaler()
    X_stand= scaler.fit_transform(X)
    pca = PCA(n_components=componentes)
    pca.fit(X_stand)
    loading= pd.DataFrame(pca.components_, columns= X.columns).T.reset_index()
    numeros= pd.Series(np.arange(1,componentes+1).astype("str"))
    nombres= pd.Series(["Componente "]*componentes)
    columnas= list(nombres+numeros)
    columnas.insert(0, "Variable")
    loading.columns= columnas
    colores=px.colors.qualitative.Light24
    data=[]
    for i in range(len(loading)):
        data.append(go.Scatter(x=[0, loading.iloc[i][componente1]], y=[0,loading.iloc[i][componente2]],
                                name=loading.iloc[i,0], marker_color=colores[i]))
    
    fig= go.Figure(data=data)
    fig.update_layout(paper_bgcolor="#0f2537", font_color="white",
                        xaxis_title=componente1, yaxis_title=componente2)
    return(fig)


@app.callback(Output('Biplot', 'figure'),
              [Input('Dropdown_PCAVar', 'value'),Input('Dropdown_PCACom', 'value'),
              Input('dd_componente1','value'), Input('dd_componente2','value')])
def biplot_PCA(variables,componentes, componente1, componente2):
    X=df[variables].replace({"No_diabetes":0,"Diabetes":1})
    scaler = StandardScaler()
    X_stand= scaler.fit_transform(X)
    pca = PCA(n_components=componentes)
    pca.fit(X_stand)
    numeros= pd.Series(np.arange(1,componentes+1).astype("str"))
    nombres= pd.Series(["Componente "]*componentes)
    columnas= list(nombres+numeros)
    x_PCA = pca.transform(X_stand)
    x_PCA = pd.DataFrame(x_PCA, columns=columnas)
    x_PCA["diabetes"] = df.diabetes
    loading= pd.DataFrame(pca.components_, columns= X.columns).T.reset_index()
    columnas.insert(0, "Variable")
    loading.columns= columnas
    colores=px.colors.qualitative.Light24
    data=[]
    for i in range(len(loading)):
        data.append(go.Scatter(x=[0, loading.iloc[i][componente1]], y=[0,loading.iloc[i][componente2]],
                                name=loading.iloc[i,0], marker_color=colores[i]))
    data.append(go.Scatter(x=x_PCA[componente1], y=x_PCA[componente2], 
                       mode= "markers", name="Datos"))
    fig= go.Figure(data=data)
    fig.update_layout(paper_bgcolor="#0f2537", font_color="white", 
                    xaxis_title=componente1, yaxis_title=componente2)
    return(fig)

@app.callback(Output('MahalanobisPlot', 'figure'),
              [Input('Dropdown_PCAVar', 'value'),Input('Dropdown_PCACom', 'value'),
              Input('dd_componente1','value'), Input('dd_componente2','value')])
def mahalanobis_PCA(variables,componentes, componente1, componente2):
    X=df[variables].replace({"No_diabetes":0,"Diabetes":1})
    scaler = StandardScaler()
    X_stand= scaler.fit_transform(X)
    pca = PCA(n_components=componentes)
    pca.fit(X_stand)
    numeros= pd.Series(np.arange(1,componentes+1).astype("str"))
    nombres= pd.Series(["Componente "]*componentes)
    columnas= list(nombres+numeros)
    x_PCA = pca.transform(X_stand)
    x_PCA = pd.DataFrame(x_PCA, columns=columnas)
    df_mahala= x_PCA[[componente1, componente2]]
    y_mu = df_mahala - np.mean(df_mahala)
    cov = np.cov(df_mahala.values.T)
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(y_mu, inv_covmat)
    mahal = np.dot(left, y_mu.T)
    df_mahala["Distancia Mahalanobis"]= mahal.diagonal()
    from scipy.stats import chi2
    data=[]
    lineaChi= chi2.ppf(0.90, df_mahala.shape[1]-1)
    data.append(go.Scatter(x=[0, df_mahala.shape[0]], y=[lineaChi, lineaChi],
                            name="Límite"))
    
    data.append(go.Scatter(y=df_mahala['Distancia Mahalanobis'], 
                        mode= "markers", name="Datos"))
    fig= go.Figure(data=data)
    fig.update_layout(paper_bgcolor="#0f2537", font_color="white")
    return(fig)

@app.callback(Output('TablaFactor', 'children'),
              [Input('Dropdown_PCAVar', 'value'),Input('Dropdown_PCACom', 'value')])
def tabla_factores(variables,componentes):
    X = df[variables]
    scaler = StandardScaler()
    X_stand= scaler.fit_transform(X)
    fa = FactorAnalyzer(componentes, rotation="varimax", method="principal")
    fa.fit(X_stand)
    numeros= pd.Series(np.arange(1,componentes+1).astype("str"))
    nombres= pd.Series(["Componente "]*componentes)
    columnas= list(nombres+numeros)
    factorTable=pd.DataFrame(fa.loadings_, index=X.columns, columns=columnas)
    factorTable["Comunality"]= fa.get_communalities()
    factorTable=factorTable.round(4)
    return dbc.Table.from_dataframe(factorTable,striped=True, bordered=True, 
                                         hover=True,index=True, dark=True)





@app.callback(Output('waterfall', 'figure'),
              [Input('Dropdown_shapN', 'value'), Input('Dropdown_MLVars', 'value')])
def shapValues(n_dato, variables):
    XModelo= df[variables]
    y= df["diabetes"].replace({"No_diabetes":0,"Diabetes":1})
    X_train, X_test, y_train, y_test = train_test_split(XModelo, y, test_size=0.2, random_state=10)
    modelo= LinearDiscriminantAnalysis()
    modelo.fit(X_train, y_train)
    explainer = shap.Explainer(modelo.predict, XModelo)
    shap_values = explainer(XModelo)
    df_shap= pd.DataFrame({"Datos":shap_values[n_dato].data, "Shap":shap_values[n_dato].values})
    df_datos= pd.DataFrame(XModelo.iloc[n_dato])
    df_datos.reset_index(inplace=True)
    df_datos.columns= ["Variable","Datos"]
    df_datos.set_index("Datos", inplace=True)
    df_datos= df_datos.join(df_shap.set_index("Datos"), on="Datos")
    df_datos["ShapAbs"]= df_datos["Shap"].abs()
    df_datos.sort_values(by="ShapAbs", inplace=True)
    df_datos.reset_index(inplace=True)
    df_datos["Compuesto"]=df_datos["Variable"]+"= "+df_datos["Datos"].astype("str")
    figWaterfall = go.Figure(go.Waterfall(
            orientation = "h", 
            y = df_datos["Compuesto"],
            x = df_datos["Shap"],
            connector = {"mode":"between", "line":{"width":1, "color":"rgb(0, 0, 0)", "dash":"solid"}},
            base=shap_values[n_dato].base_values,
            text=df_datos["Shap"].round(3)
        ))
    figWaterfall.update_layout(paper_bgcolor="#0f2537", font_color="white")
    return(figWaterfall)

@app.callback(Output('ConfusionTrain', 'figure'),
              [Input('Dropdown_MLVars', 'value')])
def confusion_train(variables):
    XModelo= df[variables]
    y= df["diabetes"]
    X_train, X_test, y_train, y_test = train_test_split(XModelo, y, test_size=0.3, random_state=10)
    modelo= LinearDiscriminantAnalysis()
    modelo.fit(X_train, y_train)
    figConfusionTrain = px.imshow(confusion_matrix(y_train, modelo.predict(X_train)),
                x= modelo.classes_, y= modelo.classes_, text_auto=True,
                labels= {"x":"Valor Real", "y":"Valor Predicción", "color":"Cantidad"})
    figConfusionTrain.update_layout(paper_bgcolor="#0f2537", font_color="white")
    return(figConfusionTrain)

@app.callback(Output('ConfusionTest', 'figure'),
              [Input('Dropdown_MLVars', 'value')])
def confusion_train(variables):
    XModelo= df[variables]
    y= df["diabetes"]
    X_train, X_test, y_train, y_test = train_test_split(XModelo, y, test_size=0.3, random_state=10)
    modelo= LinearDiscriminantAnalysis()
    modelo.fit(X_train, y_train)
    figConfusionTest = px.imshow(confusion_matrix(y_test, modelo.predict(X_test)),
                x= modelo.classes_, y= modelo.classes_, text_auto=True,
                labels= {"x":"Valor Real", "y":"Valor Predicción", "color":"Cantidad"})
    figConfusionTest.update_layout(paper_bgcolor="#0f2537", font_color="white")
    return(figConfusionTest)

# ------ Aurotización --------

#USERNAME_PASSWORD_PAIRS= [['Equipo5','diabetespa55']]
#auth= dash_auth.BasicAuth(app,USERNAME_PASSWORD_PAIRS)
server= app.server

    

# -------------------- Correr aplicación ------------------------
if __name__ == '__main__':
    app.run_server()
