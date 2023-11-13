 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:22:20 2023

@author: Equipo DSA
"""

# Importe el conjunto de datos
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import silhouette_score 

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('SECOP-ContratosII.csv' , sep=',')

dfD = df.loc[:, ~df.columns.isin(['Nombre Entidad','Nit Entidad','Ciudad','Localización','Anno BPIN','ID Contrato','Fecha Inicio Liquidacion','Fecha de Firma','Reversion','Proceso de Compra','Referencia del Contrato','Codigo de Categoria Principal','Descripcion del Proceso','TipoDocProveedor','Codigo Proveedor','Documento Proveedor','Proveedor Adjudicado','Nombre Representante Legal','Nacionalidad Representante Legal','Tipo de Identificación Representante Legal','Identificación Representante Legal','Género Representante Legal','URLProceso','Puntos del Acuerdo','Fecha Fin Liquidacion','Pilares del Acuerdo','Condiciones de Entrega','Estado BPIN','Código BPIN','Codigo Entidad','Obligación Ambiental','Obligaciones Postconsumo','Justificacion Modalidad de Contratacion','Saldo CDP','EsPostConflicto','Presupuesto General de la Nacion – PGN','Sistema General de Participaciones','Sistema General de Regalías','Recursos Propios (Alcaldías, Gobernaciones y Resguardos Indígenas)','Recursos de Credito','Recursos Propios','Ultima Actualizacion','Valor Amortizado','Origen de los Recursos','Entidad Centralizada', 'Objeto del Contrato', "Fecha de Inicio del Contrato","Fecha de Fin del Contrato","Fecha de Inicio de Ejecucion","Fecha de Fin de Ejecucion", "Liquidación","Es Grupo", "Valor Pendiente de Amortizacion"])]

# Filtramos la informacion por los contratos que no esten en aprobacion.
index_names = dfD[ (dfD['Estado Contrato'] == "En aprobación" ) ].index
dfD.drop(index_names, inplace = True)

# Filtramos la informacion por los contratos que no esten en enviado a proveedor.
index_names = dfD[ (dfD['Estado Contrato'] == "enviado Proveedor" ) ].index
dfD.drop(index_names, inplace = True)

# Filtramos la informacion por los contratos que no esten en borrador.
index_names = dfD[ (dfD['Estado Contrato'] == "Borrador" ) ].index
dfD.drop(index_names, inplace = True)

# Creamos una copia del set de todos los departamentos.
dfCor = dfD.copy(deep=True)

dfCor["Departamento"].replace({"Amazonas":1,"Antioquia":2,"Arauca":3,"Atlántico":4,"Bolívar":5,"Boyacá":6,"Caldas":7,"Caquetá":8,"Casanare":9,"Cauca":10,"Cesar":11,"Chocó":12,"Córdoba":13,"Cundinamarca":14,"Distrito Capital de Bogotá":15,"Guainía":16,"Guaviare":17,"Huila":18,"La Guajira":19,"Magdalena":20,"Meta":21,"Nariño":22,"Norte de Santander":23,"Putumayo":24,"Quindío":25,"Risaralda":26,"San Andrés, Providencia y Santa Catalina":27,"Santander":28,"Sucre":29,"Tolima":30,"Valle del Cauca":31,"Vaupés":32,"Vichada":33,"No Definido":0}, inplace = True)
dfCor["Orden"].replace({"Nacional":1,"Territorial":2,"Corporación Autónoma":3,"No Definido":0}, inplace = True)
dfCor["Sector"].replace({"agricultura":1,"Ambiente y Desarrollo Sostenible":2,"Ciencia Tecnología":3,"Cultura":4,"defensa":5,"deportes":6,"Educación Nacional":7,"Hacienda y Crédito Público":8,"Inclusión Social y Reconciliación":9,"Industria":10,"Información Estadística":11,"Inteligencia Estratégica y Contrainteligencia":12,"interior":13,"Ley de Justicia":14,"Minas y Energía":15,"No aplica/No pertenece":0,"Planeación":16,"Presidencia de la República":17,"Relaciones Exteriores":18,"Salud y Protección Social":19,"Servicio Público":20,"Tecnologías de la Información y las Comunicaciones":21,"Trabajo":22,"Transporte":23,"Vivienda, Ciudad y Territorio":24,"No Definido":25}, inplace = True)
dfCor["Rama"].replace({"Ejecutivo":1,"Judicial":2,"Corporación Autónoma":3,"Legislativo":0, "No Definido":4}, inplace = True)
dfCor["Estado Contrato"].replace({"Activo":1,"Borrador":2, "Cerrado":3, "En aprobación":4,"En ejecución":5,"Modificado":6,"Prorrogado":7,"Suspendido":8,"cedido":9,"enviado Proveedor":10, "terminado":11}, inplace = True)
#dfCor["Estado Contrato"].replace({"Activo":0,"cedido":0,"Cerrado":1,"En ejecución":0,"Modificado":0,"Prorrogado":0,"Suspendido":0,"terminado":1, "Borrador":0, "En aprobación":0, "enviado Proveedor":0}, inplace = True)
dfCor["Tipo de Contrato"].replace({"Acuerdo de cooperación":1,"Arrendamiento de inmuebles":2,"Arrendamiento de muebles":3,"Asociación Público Privada":4,"Comisión":5,"Comodato":6,"Compraventa":7,"Concesión":8,"Consultoría":9,"DecreeLaw092/2017":10,"Emprestito":11,"Interventoría":12,"Negocio fiduciario":13,"Obra":14,"Otro":15,"No Especificado":0,"Prestación de servicios":16,"Seguros":17,"Servicios financieros":18,"Suministros":19, "Venta inmuebles":20, "Venta muebles":21, "Acuerdo Marco de Precios":22,"No Definido":23}, inplace = True)
dfCor["Modalidad de Contratacion"].replace({"CCE-19-Concurso_Meritos_Con_Lista_Corta_1Sobre":1,"CCE-20-Concurso_Meritos_Sin_Lista_Corta_1Sobre":2,"Concurso de méritos abierto":3,"Concurso de méritos con precalificación":4,"Contratación Directa (con ofertas)":5,"Contratación directa":6,"Contratación régimen especial":7,"Contratación régimen especial (con ofertas)":8,"Enajenación de bienes con sobre cerrado":9,"Enajenación de bienes con subasta":10,"Licitación Pública Acuerdo Marco de Precios":11,"Licitación pública":12,"Licitación pública Obra Publica":13,"Mínima cuantía":14,"No Definido":15,"Seleccion Abreviada Menor Cuantia Sin Manifestacion Interes":16,"Selección Abreviada de Menor Cuantía":17,"Selección abreviada subasta inversa":18, "No Definido":19}, inplace = True)
#dfCor["Es Grupo"].replace({"Si":1,"No":0}, inplace = True)
dfCor["Es Pyme"].replace({"Si":1,"No":0}, inplace = True)
dfCor["Habilita Pago Adelantado"].replace({"Si":1,"No":0, "No Definido":2}, inplace = True)
#dfCor["Liquidación"].replace({"Si":1,"No":0,"No Definido":3}, inplace = True)
dfCor["Destino Gasto"].replace({"Inversión":1,"Funcionamiento":2,"No Definido":0}, inplace = True)


# Cambios el tipo de dato
dfCor["Departamento"] = dfCor["Departamento"].astype(int)
dfCor["Orden"] = dfCor["Orden"].astype(int)
dfCor["Sector"] = dfCor["Sector"].astype(int)
dfCor["Rama"] = dfCor["Rama"].astype(int)
dfCor["Estado Contrato"] = dfCor["Estado Contrato"].astype(int)
dfCor["Tipo de Contrato"] = dfCor["Tipo de Contrato"].astype(int)
dfCor["Modalidad de Contratacion"] = dfCor["Modalidad de Contratacion"].astype(int)
dfCor["Es Pyme"] = dfCor["Es Pyme"].astype(int)
dfCor["Habilita Pago Adelantado"] = dfCor["Habilita Pago Adelantado"].astype(int)
dfCor["Valor del Contrato"] = dfCor["Valor del Contrato"].astype(float)
dfCor["Valor de pago adelantado"] = dfCor["Valor de pago adelantado"].astype(float)
dfCor["Valor Facturado"] = dfCor["Valor Facturado"].astype(float)
dfCor["Valor Pendiente de Pago"] = dfCor["Valor Pendiente de Pago"].astype(float)
dfCor["Valor Pagado"] = dfCor["Valor Pagado"].astype(float)
dfCor["Destino Gasto"] = dfCor["Destino Gasto"].astype(float)

# Iniciamos proceso de cluster.
dfCluster = dfCor.loc[:, ~dfCor.columns.isin(["Orden","Rama","Sector","Es Pyme","Valor del Contrato","Valor Pagado","Valor Pendiente de Ejecucion","Tipo de Contrato","Valor de pago adelantado","Valor Facturado","Valor Pendiente de Pago","Saldo Vigencia"])]

#Importe MLFlow para registrar los experimentos, el regresor de bosques aleatorios y la métrica de error cuadrático medio
import mlflow
import mlflow.sklearn
from sklearn import metrics


# defina el servidor para llevar el registro de modelos y artefactos
# mlflow.set_tracking_uri('http://localhost:5000')
# registre el experimento
experiment = mlflow.set_experiment("sklearn-contratos")

# Aquí se ejecuta MLflow sin especificar un nombre o id del experimento. MLflow los crea un experimento para este cuaderno por defecto y guarda las características del experimento y las métricas definidas. 
# Para ver el resultado de las corridas haga click en Experimentos en el menú izquierdo. 
with mlflow.start_run(experiment_id=experiment.experiment_id):
    # defina los parámetros del modelo
    n_clusters=3
    max_iter=100

    # Cree el modelo con los parámetros definidos y entrénelo
    
    modelo=KMeans(n_clusters=n_clusters,max_iter=max_iter)
    modelo.fit(dfCluster)
    # Realice predicciones de prueba
    modelo.predict(dfCluster)
  
    # Registre los parámetros
    mlflow.log_param("n_clusters", n_clusters)
    mlflow.log_param("max_iter", max_iter)
  
    # Registre el modelo
    mlflow.sklearn.log_model(modelo, "cluster-model")
  
    # Cree y registre la métrica de interés
    #mse = mean_squared_error(y_test, predictions)
    #mlflow.log_metric("mse", mse)
    #print(mse)
    print(modelo.inertia_)
    
