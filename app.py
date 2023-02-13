from flask import Flask, render_template, request
import base64
import json

app = Flask(__name__)


@app.route("/")
def index():
    if request.is_json:
        canvas = request.args.get("canvasData")
        canvas_decoded = base64.b64decode(canvas + b"==")
        img_file = open("image.jpeg", "wb")
        img_file.write(canvas_decoded)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5151)
