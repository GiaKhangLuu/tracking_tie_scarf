# Tracking Tying Scarf

This project is a Streamlit + WebRTC pose-tracking app for monitoring the red scarf tying gesture ("deo khan quang do").

It uses MediaPipe Pose landmarks, compares detected arm/hand features against a template sequence, and gives real-time feedback:

- Correct pose
- Wrong pose
- Finish state when the final gesture is held

## Features

- Real-time webcam processing in browser via `streamlit-webrtc`
- Landmark extraction with MediaPipe Pose
- Template-based pose error scoring with NumPy
- State machine feedback: `IDLE`, `WARMUP`, `CORRECT`, `WRONG`, `FINISH`
- Visual feedback overlays and particles/icons (`smile`, `sad`, `finish`)
- Vietnamese UI labels and status text

## Requirements

From `requirements.txt`:

- `streamlit`
- `streamlit-webrtc`
- `opencv-python-headless==4.10.0.84`
- `opencv-python==4.12.0.88`
- `twilio`
- `mediapipe==0.10.14`
- `numpy==2.2.6`

## Setup

1. Create and activate a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Configure Streamlit secrets for Twilio.

Create `.streamlit/secrets.toml`:

```toml
TWILIO_ACCOUNT_SID = "your_twilio_account_sid"
TWILIO_AUTH_TOKEN = "your_twilio_auth_token"
```

The app reads these values at startup:

- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`

## Run

```bash
streamlit run app.py
```

Then open the local Streamlit URL in your browser, allow camera access, and click **Bat dau** (Start).

## How It Works

1. Webcam frames are captured through WebRTC.
2. MediaPipe Pose detects upper-body landmarks.
3. A feature vector (elbow and wrist angle/relative-position signals) is computed.
4. The current frame feature is matched against `template_seq.npy`.
5. A state machine updates feedback:
	 - `IDLE`: Not enough required landmarks detected.
	 - `WARMUP`: Landmarks are being tracked consistently.
	 - `CORRECT`: Pose error below threshold for enough consecutive frames.
	 - `WRONG`: Pose error above threshold for enough consecutive frames.
	 - `FINISH`: Final straight-arm condition held for enough frames.

Important constants in `app.py`:

- `WRONG_ANGLE_THRESHOLD = 20`
- `K_CONSECUTIVE_FRAMES = 50`
- `FINISH_CONSECUTIVE_FRAMES = 30`
- `MIN_VIS = 0.6`

## Demo

<video src="./asset/tie_scarf.mp4" controls width="720"></video>

Source video: https://www.youtube.com/watch?v=30SAzvnVCow

