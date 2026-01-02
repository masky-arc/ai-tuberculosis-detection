from flask import Flask, request, render_template
from src.image_validator import is_valid_chest_xray
from src.predict import predict_tb
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        path = os.path.join("uploads", file.filename)
        file.save(path)

        if not is_valid_chest_xray(path):
            return "Invalid X-ray image"

        result = predict_tb(path)
        return result

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
