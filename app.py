import base64
import cv2
import numpy as np
import torch

from torch.nn.functional import normalize
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

def get_image(canvas):
    """ Decode the canvas and pass it as OpenCV image. """

    decoded_canvas = base64.b64decode(canvas.split(',')[1].encode())
    canvas_as_np = np.frombuffer(decoded_canvas, dtype=np.uint8)
    
    img = cv2.imdecode(canvas_as_np, flags=1)
    resized_img = cv2.resize(img,(28,28))
    
    return resized_img

def predict_image(image):
    """ Load a model and use it to predict the image. """

    model = torch.load(f"models/2cl-1fc_bs-64_lr-0.001_epoch-10")
    model.eval()

    image = image[:,:,0] # (28,28)
    image = (torch.from_numpy(image)/1.0)  
    image = torch.unsqueeze(torch.unsqueeze(image, 0), 0) # (1,1,28,28)

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
