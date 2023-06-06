from flask import Flask, request, jsonify, render_template, Response
from image_prediction import ImageClassification
from utils import encodeImageIntoBase64, decodeImage
from flask_cors import CORS, cross_origin
import base64

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST", "GET"])
@cross_origin()
def predictRoute():
    try:
        image = request.json.get("image")
        decodeImage(image, clApp.filename)
        image_classification = ImageClassification("data/inputImage.jpg")
        result = image_classification.infer_and_display_image()
        print(result)

        return result

    except ValueError as val:
        print(val)
        return Response("Value not found inside json data", status=400)

    except Exception as e:
        print(e)
        return Response("Invalid input", status=400)


if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host="0.0.0.0", debug=True)
