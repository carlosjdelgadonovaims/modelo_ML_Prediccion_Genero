# -*- coding: utf-8 -*-
"""
Archivo Python adaptado desde el notebook de databricks dispuesto desde el servicio de Azure-Data Sandbox
Created on Mon Feb 15 22:14:16 2021
@author: Carlos Delgado
"""

## Configrurando el storage account key


storage_account_name = "Storage Account"
storage_account_key = "Storage Account Key"
container = "Storage Account Source Container"
container_raw = "Storage Account Source Container Raw"


dbutils.fs.ls("abfss://raw@stupramonitoreomercado.dfs.core.windows.net/OTROS/SNR/JSON/")
dbutils.fs.ls("abfss://sandbox@stupramonitoreomercado.dfs.core.windows.net/OTROS/SNR/nombres_entrenamiento_espanol_filtrados/")

# Algoritmo adaptado basado en https://nlpforhackers.io/introduction-machine-learning/ y adaptado del repositorio https://github.com/Jcharis/Python-Machine-Learning/tree/master/Gender%20Classification%20With%20%20Machine%20Learning

#importando librerias iniciales de administracion de datos
import pandas as pd
import numpy as np

#importando libreria y paquedes de ML desde Scikit-learn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


from pyspark.sql.functions import *
from pyspark.sql.types import *

df_original = spark.read.csv("abfss://sandbox@stupramonitoreomercado.dfs.core.windows.net/OTROS/SNR/nombres_entrenamiento_espanol_filtrados/nombres_entrenamiento_filtrados_test.csv",header=True)
display(df_original)


#almacenando los nombres por genero a pandas dataframes
df_pd_nombres_mf_general = df_original.select("*").toPandas()
df_pd_nombres_mf_general.head()



# Limpieza de datos
# Verificando consistencia de columnnas 
print(df_pd_nombres_mf_general.columns)

# Verificando tipos de datos
print(df_pd_nombres_mf_general.dtypes)

# Verificando valores nulos
print(df_pd_nombres_mf_general.isnull().isnull().sum())

# Numero de nombres femeninos
print("Numero de nombres femeninos: %s" %(len(df_pd_nombres_mf_general[df_pd_nombres_mf_general.sex == 'F'])))
# Numero de nombres masculinos
print("Numero de nombres masculinos: %s" %(len(df_pd_nombres_mf_general[df_pd_nombres_mf_general.sex == 'M'])))

df_names = df_pd_nombres_mf_general

# Remplazando con ceros y unos.
df_names.sex.replace({'F':0,'M':1},inplace=True)
df_names.sex.unique()
df_names.dtypes

Xfeatures = df_pd_nombres_mf_general['name']


#Extraccion de las características del df vectorizando
cv = CountVectorizer()
X = cv.fit_transform(Xfeatures.values.astype('U')) #Con el fin de no generar problemas en nombres con determinados carcateres 
cv.get_feature_names()


# Conformando el diccionario con la extraccion de las primeras y ultimas letras de cada uno de los nombres
def features(name):
    name = name.lower()
    return {
        'first-letter': name[0], # Primera letra
        'first2-letters': name[0:2], # Primeras 2 letras
        'first3-letters': name[0:3], # Primeras 3 letras
        'last-letter': name[-1], # Ultima letra
        'last2-letters': name[-2:], # Ultimas dos letras
        'last3-letters': name[-3:], # Ultimas tres letras
    }
    
    
    
# Vectorize the features function
features = np.vectorize(features)
#Ejemplo
print(features(["Anna", "Camilo", "Antonio","Margarita","Judith","Samuel"]))


#Extrayendo las características para el conjunto de datos vectorizado
df_X = features(df_names['name'].values.astype('U'))
df_y = df_names['sex']

#Ejemplo
arreglo = features(["Mike", "Julia"])
dv = DictVectorizer()
dv.fit(corpus)
transformed = dv.transform(arreglo)
print(transformed)


dv.get_feature_names()

# Partiendo porcentaje de entrenamiento y testeo
dfX_train, dfX_test, dfy_train, dfy_test = train_test_split(df_X, df_y, test_size=0.2, random_state=42)
dfX_train
dv = DictVectorizer()
dv.fit_transform(dfX_train)


#Definicion del clasificador Decision Trees
dclf = DecisionTreeClassifier()
my_xfeatures = dv.transform(dfX_train)
dclf.fit(my_xfeatures, dfy_train)

#Creacion de la funcion para mayor facilidad
def prediccionGenero(a):
    test_name1 = [a]
    transform_dv =dv.transform(features(test_name1))
    vector = transform_dv.toarray()
    
    if dclf.predict(vector) == 0:
        #print("Female")
        return "Female"
    else:
        #print("Male")
        return("Male")
        

#leyendo el archivo desde el json del storage y alamcenandolo como pandas dataframe
dbutils.fs.ls("abfss://sandbox@stupramonitoreomercado.dfs.core.windows.net/OTROS/SNR/")
#df_from_json = spark.read.json("abfss://sandbox@stupramonitoreomercado.dfs.core.windows.net/OTROS/SNR/nombres_apellido_nombre.json")
df_from_json = spark.read.json("abfss://sandbox@stupramonitoreomercado.dfs.core.windows.net/OTROS/SNR/nombres_comunes_masculino_femenino.json")

display(df_from_json)

df_to_predict = df_from_json.select("*").toPandas()
df_to_predict.head()     


#Prediccion de los nuevos nombres que se le presentan al modelo
final_gender = []

#for item in df_to_predict.NOMBRES:
for item in df_to_predict.PRIMER_NOMBRE:    
    #print("Nombre: %s ---- Genero: %s" %(item,clf.predict((item, ))))
    if pd.isnull(item) == True or item == '':
        final_gender.append("")
    else:
        final_gender.append(prediccionGenero(item))

#Campo donde se almacena la predicción de los nombres      
df_to_predict['PREDICCION'] = final_gender
display(df_to_predict)


#Guardando los reusltados de la predicción en CSV hacia el data lake
#df_to_predict.to_csv('/tmp/Prediccion_nombres_apellido_nombre.csv', index=False)
df_to_predict.to_csv('/tmp/Prediccion_nombres_para_clasificar.csv', index=False)
dbutils.fs.cp('file:/tmp/Prediccion_nombres_para_clasificar.csv', '/mnt/auxiliar/OTROS/KAGGLE/nombres_espanol_entrenamiento_test1/Prediccion_final_nombres_nombres_comunes_para_clasificar.csv')
