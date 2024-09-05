from flask import Flask, request, jsonify
from prophet.serialize import model_from_json
import pandas as pd

app = Flask(__name__)

# Cargar el modelo
with open('prophet_model_completo_v1.json', 'r') as fin:
    model = model_from_json(fin.read())

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame(data)
    future = model.make_future_dataframe(periods=len(df), freq='D')
    forecast = model.predict(future)
    result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(len(df)).to_dict(orient='records')
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

# Para integrar la API en la aplicación:

#import requests

#url = 'http://127.0.0.1:5000/predict'
#data = [
#    {"ds": "2023-10-01"},
#    {"ds": "2023-10-02"},
    # Agrega más fechas según sea necesario
#]

#response = requests.post(url, json=data)
#predictions = response.json()
#print(predictions)

# Para ejecutar:
# python app.py