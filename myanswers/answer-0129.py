import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def clasificar_sensores_con_ventanas(df, target_col, ventana=3):
    # 1. Calcular la desviación estándar móvil en columnas numéricas
    # Conservamos solo las numéricas para el rolling, luego reincorporamos el target
    df_numeric = df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
    df_rolling = df_numeric.rolling(window=ventana).std()
    
    # Añadimos la columna objetivo de vuelta al dataframe transformado
    df_rolling[target_col] = df[target_col]
    
    # 2. Sustituir infinitos por NaN y eliminar filas con faltantes
    # (El rolling inicial siempre genera NaNs en las primeras filas)
    df_rolling = df_rolling.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Separar características (X) y etiqueta (y)
    X = df_rolling.drop(columns=[target_col])
    y = df_rolling[target_col]
    
    # División de datos para validación
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Entrenar el modelo LinearSVC
    modelo = LinearSVC(random_state=42, max_iter=10000)
    modelo.fit(X_train, y_train)
    
    # 4. Calcular Accuracy
    predicciones = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, predicciones)
    
    return (modelo, accuracy)
