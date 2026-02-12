import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preparar_datos(df, target_col):
    """
    Toma un DataFrame, separa la variable objetivo, imputa valores nulos en las 
    características (X) y las escala para tener media 0 y desviación estándar 1.
    """
    
    # 1. Separar características (X) y variable objetivo (y)
    # Eliminamos la columna objetivo para obtener solo las features
    X = df.drop(columns=[target_col])
    
    # Convertimos la columna objetivo a un array de numpy
    y = df[target_col].to_numpy()
    
    # 2. Imputar valores faltantes en X
    # Instanciamos el imputador con la estrategia 'mean' (promedio)
    imputer = SimpleImputer(strategy='mean')
    
    # fit_transform aprende el promedio de cada columna y reemplaza los NaNs
    # Devuelve un array de numpy, por lo que perdemos los nombres de columnas de pandas
    X_imputed = imputer.fit_transform(X)
    
    # 3. Escalar las características
    # Instanciamos el escalador
    scaler = StandardScaler()
    
    # fit_transform calcula la media y la desviación estándar de X_imputed
    # y luego transforma los datos: z = (x - u) / s
    X_scaled = scaler.fit_transform(X_imputed)
    
    # 4. Devolver los dos arrays de numpy
    return X_scaled, y
