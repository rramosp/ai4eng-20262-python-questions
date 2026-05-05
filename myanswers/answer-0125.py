import pandas as pd

def limpiar_dataset_ventas(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Eliminar filas duplicadas
    df = df.copy()
    df = df.drop_duplicates()
    
    # 2. Normalizar columnas de tipo string (object)
    # Convertimos a minúsculas y eliminamos espacios al inicio/final
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.lower().str.strip()
    
    # 3. Convertir la columna 'fecha' a datetime
    if 'fecha' in df.columns:
        print (df['fecha'])
        df['fecha'] = pd.to_datetime(df['fecha'])
    
    # 4. Imputar valores nulos en columnas numéricas con la mediana
    cols_numericas = df.select_dtypes(include=['number']).columns
    df[cols_numericas] = df[cols_numericas].fillna(df[cols_numericas].median())
    
    return df
