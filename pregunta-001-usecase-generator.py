import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import random

def generar_caso_de_uso_preparar_datos():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función preparar_datos.
    """
    
    # 1. Configuración aleatoria de dimensiones
    n_rows = random.randint(3,6 )       # Entre 5 y 15 filas
    n_features = random.randint(2, 3)    # Entre 2 y 5 columnas de características
    
    # 2. Generar datos aleatorios
    # Creamos una matriz de floats aleatorios
    data = np.random.randn(n_rows, n_features)
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    
    df = pd.DataFrame(data, columns=feature_cols)
    
    # Introducimos algunos NaNs aleatorios (aprox 10% de los datos)
    mask = np.random.choice([True, False], size=df.shape, p=[0.1, 0.9])
    df[mask] = np.nan
    
    # Añadimos la columna target (sin NaNs, generalmente)
    target_col = 'target_variable'
    df[target_col] = np.random.randint(0, 2, size=n_rows) # Binario 0 o 1
    
    # ---------------------------------------------------------
    # 3. Construir el objeto INPUT
    # ---------------------------------------------------------
    input_data = {
        'df': df.copy(), # Pasamos una copia para no alterar el original durante el cálculo del output
        'target_col': target_col
    }
    
    # ---------------------------------------------------------
    # 4. Calcular el OUTPUT esperado (Ground Truth)
    #    Aquí replicamos la lógica que debería tener la función preparar_datos
    # ---------------------------------------------------------
    
    # A. Separar X e y
    X_expected = df.drop(columns=[target_col])
    y_expected = df[target_col].to_numpy()
    
    # B. Imputar (Mean)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_expected)
    
    # C. Escalar (StandardScaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    output_data = (X_scaled, y_expected)
    
    return input_data, output_data

# --- Ejemplo de uso ---
if __name__ == "__main__":
    # Generamos un caso
    entrada, salida_esperada = generar_caso_de_uso_preparar_datos()
    
    print("=== INPUT (Diccionario) ===")
    print(f"Target Column: {entrada['target_col']}")
    print("DataFrame (primeras 5 filas con posibles NaNs):")
    print(entrada['df'].head())
    
    print("\n=== OUTPUT ESPERADO (Tupla de arrays) ===")
    X_res, y_res = salida_esperada
    print(f"Shape de X procesada: {X_res.shape}")
    print(f"Shape de y: {y_res.shape}")
    print("Ejemplo de primera fila escalada:", X_res[0])
