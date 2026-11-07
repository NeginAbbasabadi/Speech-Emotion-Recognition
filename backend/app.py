
from flask import Flask, jsonify

app =  Flask(__name__)


@app.get("/api/health")
def health_check():
    """
    Health check endpoint.

    Returns a JSON response confirming that
    the Flask backend is running.
    """
    return jsonify({
        "status": "ok",
        "message": "Speech Emotion Recognition API is running"
    }), 200


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
