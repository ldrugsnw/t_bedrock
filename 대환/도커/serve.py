from flask import Flask, request, jsonify
from inference import model_fn, predict_fn

app = Flask(__name__)
model = None

def load_model():
    global model
    model_dir = '/opt/ml/model'
    model = model_fn(model_dir)

@app.route('/ping', methods=['GET'])
def ping():
    health = model is not None  # Check if the model is loaded
    status = 200 if health else 404
    return jsonify(status=status)

@app.route('/invocations', methods=['POST'])
def invocations():
    data = request.get_json()
    result = predict_fn(data, model)
    return jsonify(result)

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=8080)
