"""Real-time facial emotion recognition using the bundled FER model."""

import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parent
MODEL_JSON = ROOT / "models" / "fer.json"
MODEL_WEIGHTS = ROOT / "models" / "fer.h5"
EMOTIONS = (
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
)


def load_emotion_model() -> tf.keras.Model:
    """Load the architecture and weights shipped with this project."""
    if not MODEL_JSON.is_file() or not MODEL_WEIGHTS.is_file():
        raise FileNotFoundError(
            "Model files were not found. Expected models/fer.json and models/fer.h5."
        )

    # The bundled architecture was saved by Keras 2. Keras 3 interprets its
    # legacy batch_input_shape incorrectly, so normalize that field first.
    architecture = json.loads(MODEL_JSON.read_text(encoding="utf-8"))
    first_layer = architecture["config"]["layers"][0]["config"]
    batch_input_shape = first_layer.pop("batch_input_shape", None)
    if batch_input_shape is not None:
        first_layer["input_shape"] = batch_input_shape[1:]

    model = tf.keras.models.model_from_json(
        json.dumps(architecture),
        custom_objects={"Sequential": tf.keras.Sequential},
    )
    model.load_weights(MODEL_WEIGHTS)
    return model


def main() -> None:
    model = load_emotion_model()
    face_detector = cv2.CascadeClassifier(
        str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    )
    if face_detector.empty():
        raise RuntimeError("OpenCV could not load the face detector.")

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError(
            "Could not open the webcam. Close other camera apps and try again."
        )

    input_shape = model.input_shape
    if len(input_shape) != 4 or input_shape[1:3] != (48, 48):
        camera.release()
        raise RuntimeError(f"Expected a 48x48 model input, got {input_shape}.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a frame from the webcam.")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5, minSize=(48, 48)
            )

            for x, y, width, height in faces:
                face = gray[y : y + height, x : x + width]
                face = cv2.resize(face, (48, 48)).astype("float32") / 255.0
                prediction = model.predict(face[None, ..., None], verbose=0)[0]
                emotion = EMOTIONS[int(np.argmax(prediction))]
                confidence = float(np.max(prediction)) * 100

                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{emotion} ({confidence:.1f}%)",
                    (x, max(y - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Facial Emotion Recognition - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()