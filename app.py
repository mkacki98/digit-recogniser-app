from flask import Flask, render_template, request, jsonify
import base64
from torch.nn.functional import normalize
import cv2
import numpy as np
import torch
app = Flask(__name__)

def get_image(canvas):
    decoded_canvas = base64.b64decode(canvas.split(',')[1].encode())
    canvas_as_np = np.frombuffer(decoded_canvas, dtype=np.uint8)
    img = cv2.imdecode(canvas_as_np, flags=1)

    resized_img = cv2.resize(img,(28,28))
    cv2.imwrite('image_resized.jpg', resized_img)
    
    return resized_img

def predict_image(image):
    model = torch.load(f"models/mnist_classifier_base")

    image = image[:,:,0].flatten() # (784,)
    image = (torch.from_numpy(image) / 255.0).reshape(1, 784)

    predictions = model(normalize(image))
    return predictions, torch.argmax(predictions)

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
