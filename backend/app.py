
from flask import Flask, jsonify, request
from werkzeug.exceptions import RequestEntityTooLarge

from backend.inference import predict_emotion


app = Flask(__name__)

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
    Receive a WAV recording and predict emotion.
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
        result = predict_emotion(audio_bytes)

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
        port=5000
    )
