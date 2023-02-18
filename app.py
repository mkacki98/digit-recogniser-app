from src.utils import get_image, predict_image
from torch.nn.functional import normalize
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/predict", methods = ["GET"])
def predict():
    if request.is_json:
        canvas = request.args.get("canvasData")
        image = get_image(canvas)
        predictions, predicted_digit = predict_image(image)

        return jsonify({"probs": predictions.tolist(), 'pred_digit': predicted_digit.item()})

@app.route("/")
def index():        
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5151)
