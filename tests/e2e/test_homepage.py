
from playwright.sync_api import Page, expect

BASE_URL = "http://127.0.0.1:5002"


def test_homepage_loads(page: Page):
    page.goto(BASE_URL)

    expect(page).to_have_title(
        "VoxSense AI | Speech Emotion Recognition"
    )

    expect(page.locator("#startRecordingBtn")).to_be_visible()
    expect(page.locator("#stopRecordingBtn")).to_be_visible()

    expect(page.locator("#modelSelect")).to_be_visible()
    expect(page.locator("#analyzeBtn")).to_be_disabled()
    expect(page.locator("#compareBtn")).to_be_disabled()


def test_model_selection(page: Page):
    page.goto(BASE_URL)

    selector = page.locator("#modelSelect")

    expect(selector).to_have_value("wav2vec2")

    selector.select_option("cnn")

    expect(selector).to_have_value("cnn")
