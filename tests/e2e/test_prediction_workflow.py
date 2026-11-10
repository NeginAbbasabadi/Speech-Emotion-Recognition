
import io
import json
import math
import struct
import wave

from playwright.sync_api import Page, Route, expect


BASE_URL = "http://127.0.0.1:5002"


# --------------------------------------------------
# Test audio
# --------------------------------------------------

def create_test_wav():
    """Generate a valid 1-second WAV file at 16 kHz."""
    sample_rate = 16000
    samples = sample_rate

    audio = io.BytesIO()

    with wave.open(audio, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)

        for i in range(samples):
            value = int(
                0.3 * 32767 *
                math.sin(2 * math.pi * 440 * i / sample_rate)
            )
            wav.writeframesraw(struct.pack("<h", value))

    return audio.getvalue()


def upload_test_audio(page: Page):
    page.locator("#audioFile").set_input_files({
        "name": "test_recording.wav",
        "mimeType": "audio/wav",
        "buffer": create_test_wav(),
    })


# --------------------------------------------------
# Example API results
# --------------------------------------------------

CNN_RESULT = {
    "emotion": "angry",
    "confidence": 0.65,
    "probabilities": {
        "neutral": 0.10,
        "happy": 0.15,
        "sad": 0.10,
        "angry": 0.65,
    },
    "model": "cnn",
}

WAV2VEC2_RESULT = {
    "emotion": "happy",
    "confidence": 0.85,
    "probabilities": {
        "neutral": 0.05,
        "happy": 0.85,
        "sad": 0.05,
        "angry": 0.05,
    },
    "model": "wav2vec2",
}



def mock_prediction_api(route: Route):
    """Return a simulated prediction based on model selection."""

    # Multipart requests contain binary audio data.
    # Read the body as bytes rather than UTF-8 text.
    body = route.request.post_data_buffer or b""

    if b'name="model"\r\n\r\ncnn\r\n' in body:
        result = CNN_RESULT

    elif b'name="model"\r\n\r\nwav2vec2\r\n' in body:
        result = WAV2VEC2_RESULT

    else:
        raise AssertionError(
            "The request did not contain a supported model."
        )

    route.fulfill(
        status=200,
        content_type="application/json",
        body=json.dumps(result),
    )



# --------------------------------------------------
# Test 1: Upload and analyze
# --------------------------------------------------

def test_upload_and_analyze_audio(page: Page):
    page.route("**/api/predict", mock_prediction_api)

    page.goto(BASE_URL)

    upload_test_audio(page)

    expect(page.locator("#analyzeBtn")).to_be_enabled()

    page.locator("#analyzeBtn").click()

    expect(page.locator("#resultSection")).to_be_visible()

    expect(page.locator("#emotionLabel")).to_have_text(
        "Happy"
    )

    expect(page.locator("#confidenceLabel")).to_have_text(
        "85.0%"
    )

    expect(page.locator("#resultModelName")).to_contain_text(
        "Wav2Vec2"
    )

    expect(page.locator("#probabilities")).to_contain_text(
        "Happy"
    )


# --------------------------------------------------
# Test 2: Compare both models
# --------------------------------------------------

def test_compare_models(page: Page):
    page.route("**/api/predict", mock_prediction_api)

    page.goto(BASE_URL)

    upload_test_audio(page)

    expect(page.locator("#compareBtn")).to_be_enabled()

    page.locator("#compareBtn").click()

    expect(page.locator("#comparisonSection")).to_be_visible()

    cards = page.locator(".comparison-model")

    expect(cards).to_have_count(2)

    expect(cards.nth(0)).to_contain_text("CNN")
    expect(cards.nth(0)).to_contain_text("Angry")
    expect(cards.nth(0)).to_contain_text("65.0%")

    expect(cards.nth(1)).to_contain_text("Wav2Vec2")
    expect(cards.nth(1)).to_contain_text("Happy")
    expect(cards.nth(1)).to_contain_text("85.0%")


# --------------------------------------------------
# Test 3: API error handling
# --------------------------------------------------

def test_prediction_error_message(page: Page):
    def failing_api(route: Route):
        route.fulfill(
            status=500,
            content_type="application/json",
            body=json.dumps({
                "error": "Prediction failed on the server."
            }),
        )

    page.route("**/api/predict", failing_api)

    page.goto(BASE_URL)

    upload_test_audio(page)

    page.locator("#analyzeBtn").click()

    expect(page.locator("#status")).to_contain_text(
        "Prediction failed on the server."
    )

    expect(page.locator("#resultSection")).to_be_hidden()

    # The user should be able to try again.
    expect(page.locator("#analyzeBtn")).to_be_enabled()
