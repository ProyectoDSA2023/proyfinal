import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import datetime as dt
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output

# Leer el archivo CSV
csv_file = "./SECOP-ContratosIIResultado.csv"
df = pd.read_csv(csv_file)    

# Inicializar Dash
app = dash.Dash(__name__)

# Crear pie charts
def generate_pie_chart():
    fig = px.pie(df, names=df.columns[1], title=' ')
    return fig

def generate_pie_chart2():
    fig2 = px.pie(df, names=df.columns[3], title=' ')
    return fig2
    
def generate_pie_chart3():
    fig3 = px.pie(df, names=df.columns[4], title=' ')
    return fig3

# Definir disposicion del tablero
app.layout = html.Div([
    html.H1("Dashboard SECOP-Contratos"),
    
    # 1ra seccion: Histograma
    html.Div([
        html.H2("Histograma SECOP-Contratos"),
        
        # Texto desplegable para seleccionar columna
        dcc.Dropdown(
            id='histogram-column-dropdown',
            options=[
                {'label': column, 'value': column}
                for column in df.columns
            ],
            value=df.columns[-1],  # Por defecto la última columna
            style={'width': '50%'}
        ),
        
        # Dibujar Histograma
        dcc.Graph(id='histogram-plot')
    ]),
    
    # 2da seccion: Tabla
    html.Div([
        html.H2("Tabla SECOP-Contratos"),
        
        # Texto desplegable para filtrar por cluster
        dcc.Dropdown(
            id='filter-dropdown',
            options=[
                {'label': value, 'value': value}
                for value in df[df.columns[-1]].unique()
            ],
            multi=True,  # Permitir filtro por multiples valores
            placeholder="Seleccione cluster",
            style={'width': '50%'}
        ),
        
        # Dibujar Tabla
        dash_table.DataTable(
            id='filtered-values-table',
            columns=[{'name': col, 'id': col} for col in df.columns],
            data=[],
            page_size=20  # Num de registros por pagina
        )
    ]),
    
    # 3ra seccion: PieChart1 OrdenContratacion
    html.Div([
        html.H2("Parámetros relevantes para el análsis de SECOP-Contratos"),
        html.H2("Orden de Contratación"),
    
        dcc.Graph(
            id='pie-chart',
            figure=generate_pie_chart()
        )
    ]),

    # 4ta seccion: PieChart2 RamaEstatal
    html.Div([
        html.H2("Rama Estatal"),
    
        dcc.Graph(
            id='pie-chart2',
            figure=generate_pie_chart2()
        )
    ]),

    # 5ta seccion: PieChart3 Estado Contrato
    html.Div([
        html.H2("Estado Contrato"),
    
        dcc.Graph(
            id='pie-chart3',
            figure=generate_pie_chart3()
        )
    ]),    

    # Boton para leer csv
    html.Button("Reload CSV", id='reload-button', n_clicks=0, style={'margin-top': '20px'}),
    #html.A(html.Button('Refresh Data'),href='/'),
           
])

# Actualizar los valores desplegados en el histograma basados en la seleccion de la caja de texto desplegable.
@app.callback(
    Output('histogram-plot', 'figure'),
    [Input('histogram-column-dropdown', 'value')]
)
def update_histogram(selected_column):
    fig = px.histogram(df, x=selected_column, title=f'Histograma para {selected_column}')
    return fig

# Actualizar los valores desplegados en la tabla  basados en la seleccion de la caja de texto desplegable.
@app.callback(
    Output('filtered-values-table', 'data'),
    [Input('filter-dropdown', 'value')]
)
def update_filtered_values_table(selected_values):
    if not selected_values:
        return df.to_dict('records')
    
    filtered_df = df[df[df.columns[-1]].isin(selected_values)]  
    return filtered_df.to_dict('records')


# Actualizar archivo csv
def reload_csv(n_clicks):
    global df
    df = pd.read_csv(csv_file)
    
    # Update pie chart figures
    histog_fig = update_histogram(selected_column)
    table_fig = update_filtered_values_table(selected_values)
    pie_fig = generate_pie_chart()
    pie_fig2 = generate_pie_chart2()
    pie_fig3 = generate_pie_chart3()
        
    return histog_fig, table_fig, pie_fig, pie_fig2, pie_fig3
  
# Ejecutar la app
if __name__ == '__main__':
    app.run_server(host="0.0.0.0", debug=False)



