import os

from flask import Flask, jsonify, request
from flask_cors import CORS

from image_extraction.image_processor import extract_image_data
from image_extraction.vision_processor import analyze_image


app = Flask(__name__)
CORS(app)

IMAGE_UPLOAD_FOLDER = "image_extraction/uploads"
os.makedirs(IMAGE_UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return jsonify({
        "status": "success",
        "message": "Multimodal Extraction API is running"
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy"
    })


@app.route("/api/extract/image", methods=["POST"])
def extract_image():
    print("\n========== IMAGE REQUEST ==========")
    print("Content-Type:", request.content_type)
    print("Files received:", list(request.files.keys()))
    print("Form fields:", list(request.form.keys()))
    print("===================================\n")

    try:
        if "image" not in request.files:
            return jsonify({
                "status": "error",
                "message": "No image file provided",
                "debug": {
                    "content_type": request.content_type,
                    "files_received": list(request.files.keys()),
                    "form_fields": list(request.form.keys())
                }
            }), 400

        image = request.files["image"]

        if image.filename == "":
            return jsonify({
                "status": "error",
                "message": "No image selected"
            }), 400

        image_path = os.path.join(
            IMAGE_UPLOAD_FOLDER,
            image.filename
        )

        image.save(image_path)

        image_data = extract_image_data(image_path)
        visual_analysis = analyze_image(image_path)

        image_data["visual_analysis"] = visual_analysis

        return jsonify({
            "status": "success",
            "data": image_data
        })

    except Exception as e:
        print("ERROR:", repr(e))

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )