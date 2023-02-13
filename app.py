from flask import Flask, render_template, request
import base64
import json
import cv2
import numpy as np

app = Flask(__name__)


@app.route("/")
def index():
    if request.is_json:

        canvas = request.args.get("canvasData")
        decoded_canvas = base64.b64decode(canvas.split(',')[1].encode())
        canvas_as_np = np.frombuffer(decoded_canvas, dtype=np.uint8)

        # Load image with OpenCV, save the image
        img = cv2.imdecode(canvas_as_np, flags=1)
        cv2.imwrite('image.jpg', img)
        

    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5151)
