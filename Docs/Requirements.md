# Speech Emotion Recognition — Software Requirements

## 1. Project Overview

The goal is to develop a full-stack web application that recognizes emotional expressions in recorded speech using a Convolutional Neural Network (CNN).

The user will record or upload a short audio clip, and the application will predict one of four emotion categories: happy, sad, angry, or neutral.

The application will use a Flask backend, a PyTorch CNN model, and a browser-based frontend.

## 2. Functional Requirements

**FR-01 — Audio Recording**

The application shall allow users to record short speech clips using their microphone.

**FR-02 — Audio Upload**

The application shall allow users to upload supported audio files instead of recording directly.

**FR-03 — Audio Preprocessing**

The backend shall convert supported audio inputs into a standardized format and extract the same audio features used during model training.

**FR-04 — Emotion Classification**

The CNN shall classify each processed speech recording into one of four emotion categories: happy, sad, angry, or neutral.

**FR-05 — Prediction Results**

The application shall display the predicted emotion and an associated model confidence score.

**FR-06 — Error Handling**

The application shall display meaningful error messages when an audio file is invalid, recording permission is denied, or prediction fails.

## 3. Non-Functional Requirements

**NFR-01 — Usability**

The interface shall be responsive, accessible, and easy to use on desktop and mobile browsers.

**NFR-02 — Performance**

The initial target is to return a prediction within five seconds after the server receives a valid short audio clip on the development machine. Actual performance will be measured during testing.

**NFR-03 — Maintainability**

The codebase shall separate audio processing, model inference, backend routing, and frontend components.

**NFR-04 — Reliability**

The application shall handle invalid, empty, or unsupported audio inputs without crashing.

**NFR-05 — Documentation**

The project shall contain setup instructions, API documentation, architecture diagrams, and code comments where appropriate.

**NFR-06 — Privacy**

Audio recordings shall be processed without permanent server-side storage by default.

## 4. Machine Learning Requirements

- Dataset: RAVDESS, initially using the speech recordings for the four selected emotions.
- Audio features: log-Mel spectrograms or MFCC feature maps, with one approach selected before model training.
- Model architecture: Convolutional Neural Network implemented in PyTorch.
- Training environment: Local development machine with Apple Silicon GPU acceleration.
- Evaluation: Accuracy, macro F1-score, and confusion matrix on recordings from held-out speakers.
- Robustness: Evaluate the effect of background-noise augmentation.

The model recognizes vocal patterns associated with acted emotional expressions; its predictions do not establish the speaker's actual emotional or psychological state.

## 5. MVP Scope

The first working version shall support recording or uploading a short speech clip, sending it to the Flask backend, preprocessing it, generating a CNN prediction, and displaying the result.

The following features are outside the initial scope:

- Continuous live speech recognition
- User accounts and authentication
- Database storage of recordings
- Support for additional emotion categories
- Emotion history and analytics dashboards
- Multilingual emotion recognition guarantees

## 6. Acceptance Criteria

The MVP is considered functionally complete when:

1. A user can successfully record or upload a supported audio clip.
2. The frontend sends the recording to the Flask API.
3. The backend preprocesses the audio and successfully executes the trained CNN.
4. The API returns a valid JSON prediction with an emotion label and confidence score.
5. The frontend displays the result.
6. Invalid audio inputs generate appropriate error responses.
7. The completed application passes the defined integration tests.


flowchart TD
    A["Browser: Record or upload audio"]
    B["Flask API: Validate input"]
    C["Audio preprocessing"]
    D["CNN model inference"]
    E["JSON prediction response"]
    F["Browser: Display emotion"]

    A -->|HTTP POST| B
    B --> C
    C --> D
    D --> E
    E --> F


Model performance will be reported separately using held-out test data. A final accuracy target will be established after baseline evaluation.
