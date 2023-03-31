from src.utils.utils import get_image, predict_image
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/predict", methods = ["GET"])
def predict():
    if request.is_json:
        canvas = request.args.get("canvasData")
        image = get_image(canvas)
        models_predictions = predict_image(image)

        return jsonify(models_predictions)

@app.route("/")
def index():        
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug = True)
