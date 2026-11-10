
import json
import re

import pytest
from playwright.sync_api import Page, expect


BASE_URL = "http://127.0.0.1:5002"


# --------------------------------------------------
# Browser fixture with simulated microphone
# --------------------------------------------------

@pytest.fixture
def microphone_page(playwright):
    browser = playwright.chromium.launch(
        headless=True,
        args=[
            "--use-fake-device-for-media-stream",
            "--use-fake-ui-for-media-stream",
        ],
    )

    context = browser.new_context(
        permissions=["microphone"],
    )

    page = context.new_page()

    yield page

    context.close()
    browser.close()


# --------------------------------------------------
# Test 1: Recording controls and timer
# --------------------------------------------------

def test_microphone_recording(microphone_page: Page):
    page = microphone_page
    page.goto(BASE_URL)

    start = page.locator("#startRecordingBtn")
    stop = page.locator("#stopRecordingBtn")
    indicator = page.locator("#recordingIndicator")
    timer = page.locator("#recordingTimer")

    expect(start).to_be_enabled()
    expect(stop).to_be_disabled()

    # Begin capturing synthetic microphone audio.
    start.click()

    expect(indicator).to_have_class(
        re.compile(r"\bactive\b")
    )

    expect(page.locator("#recordingLabel")).to_have_text(
        "RECORDING"
    )

    expect(start).to_be_disabled()
    expect(stop).to_be_enabled()

    # Wait for the recording timer to advance.
    expect(timer).not_to_have_text(
        "00:00",
        timeout=5000,
    )

    # Stop and allow WAV conversion to finish.
    stop.click()

    expect(stop).to_be_disabled()

    expect(
        page.locator("#analyzeBtn")
    ).to_be_enabled(timeout=15000)

    expect(indicator).not_to_have_class(
        re.compile(r"\bactive\b")
    )

    expect(
        page.locator("#recordingStatus")
    ).to_contain_text("Recording ready")

    # The recorded audio should have a browser preview.
    expect(
        page.locator("#audioPreviewSection")
    ).to_be_visible()

    expect(
        page.locator("#audioPreview")
    ).to_have_attribute(
        "src",
        re.compile(r"^blob:")
    )


# --------------------------------------------------
# Test 2: Recorded WAV is submitted for prediction
# --------------------------------------------------

def test_recorded_audio_prediction(microphone_page: Page):
    page = microphone_page

    captured_requests = []

    def mock_api(route):
        request_body = (
            route.request.post_data_buffer or b""
        )

        captured_requests.append(request_body)

        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({
                "emotion": "happy",
                "confidence": 0.85,
                "probabilities": {
                    "neutral": 0.05,
                    "happy": 0.85,
                    "sad": 0.05,
                    "angry": 0.05,
                },
                "model": "wav2vec2",
            }),
        )

    page.route("**/api/predict", mock_api)
    page.goto(BASE_URL)

    # Record synthetic audio.
    page.locator("#startRecordingBtn").click()

    expect(
        page.locator("#stopRecordingBtn")
    ).to_be_enabled()

    # Capture at least one second.
    expect(
        page.locator("#recordingTimer")
    ).not_to_have_text("00:00", timeout=5000)

    page.locator("#stopRecordingBtn").click()

    expect(
        page.locator("#analyzeBtn")
    ).to_be_enabled(timeout=15000)

    # Choose Wav2Vec2 and submit the recording.
    page.locator("#modelSelect").select_option(
        "wav2vec2"
    )

    page.locator("#analyzeBtn").click()

    expect(
        page.locator("#resultSection")
    ).to_be_visible()

    expect(
        page.locator("#emotionLabel")
    ).to_have_text("Happy")

    expect(
        page.locator("#confidenceLabel")
    ).to_have_text("85.0%")

    # Check the actual HTTP request was sent.
    assert len(captured_requests) == 1

    body = captured_requests[0]

    # Verify WAV data exists in the multipart payload.
    assert b"RIFF" in body
    assert b"WAVE" in body
    assert b"fmt " in body
    assert b"data" in body

    # Verify the file and model were transmitted.
    assert b"voice_recording.wav" in body
    assert b'name="model"\r\n\r\nwav2vec2\r\n' in body
