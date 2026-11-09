
from backend.wav2vec2_inference import predict_wav2vec2
from werkzeug.exceptions import RequestEntityTooLarge
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory

from backend.inference import predict_emotion


app = Flask(__name__)

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


@app.get("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/style.css")
def stylesheet():
    return send_from_directory(FRONTEND_DIR, "style.css")


@app.get("/script.js")
def javascript():
    return send_from_directory(FRONTEND_DIR, "script.js")

# Maximum request size: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


@app.get("/api/health")
def health_check():
    """Check whether the API is running."""
    return jsonify({
        "status": "ok",
        "message": "Speech Emotion Recognition API is running"
    }), 200



@app.post("/api/predict")
def predict():
    """
    Receive a WAV recording and predict emotion
    using the selected model.
    """
    if "audio" not in request.files:
        return jsonify({
            "error": "No audio file provided."
        }), 400

    audio_file = request.files["audio"]

    if not audio_file.filename:
        return jsonify({
            "error": "No filename provided."
        }), 400

    if not audio_file.filename.lower().endswith(".wav"):
        return jsonify({
            "error": "Only WAV files are supported."
        }), 400

    audio_bytes = audio_file.read()

    try:
        # Read selected model from the frontend.
        # Default to Wav2Vec2.
        model_type = request.form.get("model", "wav2vec2")

        if model_type == "cnn":
            result = predict_emotion(audio_bytes)

        elif model_type == "wav2vec2":
            result = predict_wav2vec2(audio_bytes)

        else:
            return jsonify({
                "error": "Unsupported model."
            }), 400

        # Include model name in the response.
        result["model"] = model_type

        return jsonify(result), 200

    except ValueError as exc:
        return jsonify({
            "error": str(exc)
        }), 400

    except Exception:
        app.logger.exception("Prediction failed")

        return jsonify({
            "error": "Prediction failed on the server."
        }), 500



@app.errorhandler(RequestEntityTooLarge)
def file_too_large(error):
    return jsonify({
        "error": "Audio file exceeds the 10 MB limit."
    }), 413


if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5002
    )
    
