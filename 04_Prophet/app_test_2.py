from flask import Flask, request, jsonify
from prophet import Prophet
import pandas as pd

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Extraer parámetros del modelo del JSON
    model_params = data.get('model_params', {})
    
    # Crear el modelo Prophet con los parámetros proporcionados
    m = Prophet(
        growth=model_params.get('growth', 'linear'),
        changepoints=model_params.get('changepoints', None),
        n_changepoints=model_params.get('n_changepoints', 25),
        changepoint_range=model_params.get('changepoint_range', 0.8),
        yearly_seasonality=model_params.get('yearly_seasonality', 'auto'),
        weekly_seasonality=model_params.get('weekly_seasonality', 'auto'),
        daily_seasonality=model_params.get('daily_seasonality', True),
        holidays=model_params.get('holidays', None),
        seasonality_mode=model_params.get('seasonality_mode', 'additive'),
        seasonality_prior_scale=model_params.get('seasonality_prior_scale', 10.0),
        holidays_prior_scale=model_params.get('holidays_prior_scale', 10.0),
        changepoint_prior_scale=model_params.get('changepoint_prior_scale', 0.05),
        mcmc_samples=model_params.get('mcmc_samples', 0),
        interval_width=model_params.get('interval_width', 0.95),
        uncertainty_samples=model_params.get('uncertainty_samples', 1000),
        stan_backend=model_params.get('stan_backend', None)
    )
    
    # Agregar días festivos si se especifica
    if 'country_holidays' in model_params:
        m.add_country_holidays(country_name=model_params['country_holidays'])
    
    # Obtener los datos de entrenamiento y las fechas para la predicción
    training_data = pd.DataFrame(data.get('training_data', []))
    prediction_dates = pd.DataFrame(data.get('prediction_dates', []))
    
    # Ajustar el modelo
    model = m.fit(training_data)
    
    # Crear el DataFrame futuro
    future = model.make_future_dataframe(periods=len(prediction_dates), freq='D')
    
    # Hacer la predicción
    forecast = model.predict(future)
    
    # Formatear el resultado
    result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(len(prediction_dates)).to_dict(orient='records')
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

# Para integrar la API en la aplicación:

#import requests

#url = 'http://127.0.0.1:5000/predict'
#{
#  "model_params": {
#    "growth": "linear",
#    "changepoints": null,
#    "n_changepoints": 25,
#    "changepoint_range": 0.8,
#    "yearly_seasonality": "auto",
#    "weekly_seasonality": "auto",
#    "daily_seasonality": true,
#    "holidays": null,
#    "seasonality_mode": "additive",
#    "seasonality_prior_scale": 10.0,
#    "holidays_prior_scale": 10.0,
#    "changepoint_prior_scale": 0.05,
#    "mcmc_samples": 0,
#    "interval_width": 0.95,
#    "uncertainty_samples": 1000,
#    "stan_backend": null,
#    "country_holidays": "UY"
#  },
#  "prediction_dates": [
#    {"ds": "2023-10-01"},
#    {"ds": "2023-10-02"}
#  ]
#}

#response = requests.post(url, json=data)
#predictions = response.json()
#print(predictions)

# Para ejecutar:
# python app_test_2.py