from flask import Flask, request, jsonify
import xgboost as xgb
import pandas as pd
import numpy as np

app = Flask(__name__)

# Cargar el modelo XGBoost guardado
model = xgb.Booster()
model.load_model('xgboost_v3.json')

# Función de preprocesamiento para convertir los datos de entrada
def preprocess_input(data):
    df = pd.DataFrame(data)

    # Aplicar las transformaciones necesarias para las características de fecha
    df['day_of_week'] = pd.to_datetime(df['ds']).dt.dayofweek
    df['day_of_month'] = pd.to_datetime(df['ds']).dt.day
    df['month'] = pd.to_datetime(df['ds']).dt.month
    df['year'] = pd.to_datetime(df['ds']).dt.year
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['quarter'] = pd.to_datetime(df['ds']).dt.quarter
    df['week_of_year'] = pd.to_datetime(df['ds']).dt.isocalendar().week
    df['day_of_year'] = pd.to_datetime(df['ds']).dt.dayofyear

    # Asegurar que la columna 'is_holiday' esté presente
    if 'is_holiday' not in df.columns:
        df['is_holiday'] = 0  # Si no se encuentra la columna, la creamos con valores predeterminados

    # Para predicciones, no tenemos el valor 'y', así que no es necesario
    # Eliminar la columna 'ds' ya que no es necesaria para el modelo
    df = df.drop(columns=['ds'], errors='ignore')

    # Reordenar las columnas según lo esperado por el modelo
    expected_columns = ['day_of_week', 'day_of_month', 'month', 'year', 'is_weekend',
                        'quarter', 'week_of_year', 'day_of_year', 'is_holiday', 
                        'rolling_mean_2', 'rolling_mean_7', 'rolling_mean_30']

    # Para predicciones no tenemos medias móviles, así que podemos usar valores predeterminados
    for col in ['rolling_mean_2', 'rolling_mean_7', 'rolling_mean_30']:
        if col not in df.columns:
            df[col] = 0  # O valores predeterminados que elijas para predicciones

    # Reordenar las columnas en el DataFrame
    df = df[expected_columns]

    return df


@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos de la solicitud
    data = request.get_json(force=True)
    print("Datos recibidos:", data)

    # Preprocesar los datos
    df_input = preprocess_input(data)
    
    # Verificar el orden de las columnas
    print("Columnas del DataFrame de entrada:", df_input.columns)

    # Convertir los datos a DMatrix, que es la estructura de datos que usa XGBoost
    dmatrix_input = xgb.DMatrix(df_input)

    # Realizar las predicciones
    predictions = model.predict(dmatrix_input)

    # Obtener la importancia de las características
    importance_dict = model.get_score(importance_type='weight')  # Puedes cambiar 'weight' por 'gain', 'cover', etc.
    
    # Formatear las predicciones y la importancia en un formato JSON amigable
    result = {
        "predictions": pd.DataFrame({'Predicted': predictions}).to_dict(orient='records'),
        "feature_importance": importance_dict
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)



#Paso 2: Uso de la API para realizar predicciones
#import requests

#url = 'http://127.0.0.1:5000/predict'
#data = [
#    {"ds": "2024-01-01"},
#    {"ds": "2024-01-02"}
#]

#response = requests.post(url, json=data)  # Asegúrate de usar 'json=data' para enviar los datos correctamente
#print(response.json())

#Aclaración para testear en postman: no utilizar data =, sólo
#[
#    {"ds": "2024-01-01"},
#    {"ds": "2024-01-02"}
#]

#Paso 3: Ejecución del servidor Flask
#python app_xgboost.py