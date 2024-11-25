from flask import Flask, request, jsonify
import pickle
import numpy as np

# Rutas de los modelos
CROP_MODEL = 'models/crop_model.pickle'
DROUGHT_MODEL = 'models/drought_model.pkl'
FLOOD_MODEL = 'models/flood_model.pkl'
GRADIENT_BOOST_MODEL = 'models/gradient_boost_model.pkl'

app = Flask(__name__)

@app.route('/predecirCrop', methods=['POST'])
def predecir_crop():
    return procesar_prediccion(CROP_MODEL, "Crop")

@app.route('/predecirDrought', methods=['POST'])
def predecir_drought():
    return procesar_prediccion(DROUGHT_MODEL, "Drought")

@app.route('/predecirFlood', methods=['POST'])
def predecir_flood():
    return procesar_prediccion(FLOOD_MODEL, "Flood")

@app.route('/predecirFire', methods=['POST'])
def predecir_fire():
    return procesar_prediccion(GRADIENT_BOOST_MODEL, "Fire")

def procesar_prediccion(model_path, model_name):
    """Función genérica para cargar un modelo y procesar predicciones."""
    try:
        datos = request.json  # Leer datos JSON
        if 'input' not in datos:
            raise ValueError("El campo 'input' no está en los datos enviados.")
        
        # Convertir los datos a NumPy y asegurarse de que sean 2D
        input_data = np.array(datos['input'], dtype=float).reshape(1, -1)

        # Cargar el modelo
        loaded_model = pickle.load(open(model_path, 'rb'))
        
        # Verificar si es clasificación con probabilidades
        if hasattr(loaded_model, 'predict_proba'):
            prediction_probabilities = loaded_model.predict_proba(input_data)
            classes = loaded_model.classes_
            # Convertir resultados a tipos nativos de Python
            result = {str(class_name): float(prediction_probabilities[0][i]) for i, class_name in enumerate(classes)}
        else:
            prediction = loaded_model.predict(input_data)
            # Convertir predicción a lista de tipos nativos
            result = {"prediction": [int(p) if isinstance(p, np.integer) else float(p) for p in prediction]}
        
        return jsonify(result)

    except FileNotFoundError:
        return jsonify({
            "error": f"No se encontró el archivo del modelo '{model_name}'. Asegúrate de que esté en el directorio correcto."
        }), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Flask necesita estas configuraciones para trabajar con Vercel
from flask import Response as BaseResponse
from werkzeug.wrappers.response import Response
BaseResponse.autocorrect_location_header = False
app.response_class = Response
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Usa el puerto proporcionado por Railway o 5000 como predeterminado
    app.run(host="0.0.0.0", port=port)
