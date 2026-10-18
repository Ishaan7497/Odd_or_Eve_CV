# Odd or Eve Computer Vision

A computer vision version of "Odd or Eve" (Hand Cricket). Play against an adaptive AI by showing hand gestures to your webcam.

## Features
- Real-time hand tracking with MediaPipe HandLandmarker
- Gesture input for numbers 1–10
- Three match formats: Classic (1–6), Extended (1–10) and Crazy
- Three AI difficulty levels: Easy, Medium and Hard
- Coin toss, innings break, target chasing and super over
- Sound effects, crowd audio, particles and celebrations
- Quit with a gesture or a key

## Installation

```bash
git clone https://github.com/Ishaan7497/Odd_or_Eve_CV.git
cd Odd_or_Eve_CV
pip install -r requirements.txt
```

The `hand_landmarker.task` model comes with the repo. To use a different copy, set the `HAND_LANDMARKER_PATH` environment variable.

## Usage

```bash
python OoE_CV.py
```

### Gestures
| Number | Gesture |
|--------|---------|
| 1–4 | That many fingers up, thumb folded |
| 5 | Open hand |
| 6 | Thumb only |
| 7 | Thumb + index |
| 8 | Thumb + index + middle |
| 9 | Thumb + four fingers except pinky |
| 10 | Both hands cupped / closed |

Hold a gesture steady until the ring fills to confirm it.

### Match formats
- **Classic**: numbers 1–6. Same number as the bowler means you're out.
- **Extended**: numbers 1–10.
- **Crazy**: numbers next to each other (like 3 and 4, or 1 and 10) mean you're out. The same number scores its square.

### Quitting
- Press **ESC** or **Q**, or
- Hold a closed fist with **both hands** for 3 seconds.

### Optional audio
- Put `bgm.mp3`, `bgm.ogg` or `bgm.wav` next to the script for background music.
- Any sound effect can be replaced by dropping in a file with the same name (for example `crowd_cheer.mp3` or `stumps_hit.wav`).

## Requirements
- Python 3.9+
- A webcam
- OpenCV, MediaPipe, NumPy, pygame (see `requirements.txt`)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
