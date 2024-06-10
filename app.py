from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('formulario.html')

# Cargar el modelo
model = joblib.load('random_forest_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        street_robbery = float(request.form['street_robbery'])
        agg_assault = float(request.form['agg_assault'])

        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[street_robbery, agg_assault]], columns=['Street_robbery', 'Agg_assault'])
        app.logger.debug(f'DataFrame creado: {data_df}')

        # Realizar predicciones
        prediction_proba = model.predict_proba(data_df)[0]
        app.logger.debug(f'Probabilidades de predicción: {prediction_proba}')

        # Asignar categorías basadas en probabilidades
        if prediction_proba[1] > 0.75:
            resultado = "ALTA probabilidad de robo"
        elif prediction_proba[1] > 0.50:
            resultado = "MEDIA probabilidad de robo"
        elif prediction_proba[1] > 0.25:
            resultado = "BAJA probabilidad de robo"
        else:
            resultado = "POCA probabilidad de robo"

        # Devolver las predicciones como respuesta JSON
        return jsonify({'prediction': resultado})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
