"""
HAND CRICKET - Gesture-based cricket vs an adaptive AI.

Show fingers to the webcam to play your shot / bowl your delivery.
Gestures: 1-5 = number of fingers up,  6-9 = thumb + closed fingers, 10 = cupped hand.

Controls inside the game:
  ESC  - quit
"""

import os
import sys
import cv2
import time
import random
import subprocess
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from collections import Counter

try:
    import pygame
    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
    SOUND_ON = True
    print("[sound] pygame mixer initialized -- synthesized audio enabled")
except Exception as e:
    SOUND_ON = False
    print(f"[sound] pygame mixer FAILED to init: {e}")
    print("[sound] To enable audio, install pygame:  pip install pygame")
    print("[sound] (On Linux you may also need:  sudo apt install libsdl2-mixer-2.0-0)")


def _make(pattern, sr=22050):
    segs = []
    for freq, dur, vol in pattern:
        n = max(1, int(sr * dur))
        t = np.linspace(0, dur, n, False)
        wave = np.sin(2 * np.pi * freq * t)
        atk = min(int(0.008 * sr), n // 4)
        rel = min(int(0.030 * sr), n // 3)
        if atk > 0:
            wave[:atk] *= np.linspace(0, 1, atk)
        if rel > 0:
            wave[-rel:] *= np.linspace(1, 0, rel)
        segs.append(wave * vol)
    data = np.clip(np.concatenate(segs), -1, 1)
    mono = (data * 32767).astype(np.int16)
    stereo = np.column_stack([mono, mono])
    return pygame.sndarray.make_sound(stereo)


def _chord(freqs, dur, vol, sr=22050):
    n = max(1, int(sr * dur))
    t = np.linspace(0, dur, n, False)
    wave = np.zeros(n)
    for f in freqs:
        wave += np.sin(2 * np.pi * f * t)
    wave /= len(freqs)
    atk = min(int(0.008 * sr), n // 4)
    rel = min(int(0.040 * sr), n // 3)
    if atk > 0:
        wave[:atk] *= np.linspace(0, 1, atk)
    if rel > 0:
        wave[-rel:] *= np.linspace(1, 0, rel)
    return wave * vol


def _make_chord_seq(chord_list, sr=22050):
    segs = [_chord(f, d, v, sr) for f, d, v in chord_list]
    data = np.clip(np.concatenate(segs), -1, 1)
    mono = (data * 32767).astype(np.int16)
    stereo = np.column_stack([mono, mono])
    return pygame.sndarray.make_sound(stereo)


if SOUND_ON:
    SND = {
        'click':     _make([(880, 0.04, 0.30), (1320, 0.05, 0.25)]),
        'hover':     _make([(660, 0.03, 0.18)]),
        'run':       _make([(620, 0.06, 0.35), (820, 0.10, 0.40)]),
        'four':      _make_chord_seq([
                        ([523, 659], 0.10, 0.45),
                        ([659, 784], 0.10, 0.45),
                        ([784, 988], 0.20, 0.55),
                     ]),
        'six':       _make_chord_seq([
                        ([523, 659, 784], 0.10, 0.50),
                        ([587, 740, 880], 0.10, 0.50),
                        ([659, 831, 988], 0.10, 0.50),
                        ([784, 988, 1175], 0.32, 0.65),
                     ]),
        'wicket':    _make([(440, 0.05, 0.45), (330, 0.10, 0.50),
                            (220, 0.12, 0.55), (165, 0.20, 0.55)]),
        'out':       _make([(330, 0.18, 0.55), (247, 0.20, 0.55),
                            (165, 0.30, 0.55), (110, 0.40, 0.50)]),
        'coin':      _make([(900, 0.04, 0.30), (1200, 0.04, 0.30),
                            (900, 0.04, 0.25), (1200, 0.04, 0.30),
                            (900, 0.04, 0.20), (1300, 0.05, 0.30)]),
        'toss_win':  _make_chord_seq([
                        ([784, 988], 0.10, 0.50),
                        ([1047, 1319], 0.20, 0.55),
                     ]),
        'win':       _make_chord_seq([
                        ([392, 494], 0.10, 0.50),
                        ([523, 659], 0.10, 0.50),
                        ([659, 784], 0.10, 0.50),
                        ([784, 988, 1175], 0.45, 0.65),
                     ]),
        'lose':      _make_chord_seq([
                        ([330, 415], 0.18, 0.50),
                        ([277, 349], 0.18, 0.50),
                        ([220, 277], 0.20, 0.50),
                        ([165, 196], 0.45, 0.45),
                     ]),
        'tie':       _make([(523, 0.12, 0.45), (494, 0.12, 0.45), (523, 0.30, 0.50)]),
        'tick':      _make([(1500, 0.025, 0.20)]),
        # Cutscene sounds
        'crowd_cheer': _make_chord_seq([
                        ([220, 277, 330, 415], 0.30, 0.30),
                        ([247, 311, 370, 466], 0.30, 0.35),
                        ([262, 330, 392, 494], 0.40, 0.40),
                        ([220, 277, 330, 392, 494], 0.80, 0.45),
                     ]),
        'crowd_aww':   _make_chord_seq([
                        ([523, 466], 0.25, 0.35),
                        ([466, 415], 0.30, 0.35),
                        ([415, 370], 0.40, 0.35),
                        ([370, 311], 0.60, 0.35),
                     ]),
        # "OUT!" call: vowel-formant-ish, suggests a shouted syllable.
        # Two-syllable shape: "OW" -> "T" (sharp tail).
        'out_call':    _make_chord_seq([
                        # "OW" — low formant cluster, gliding up slightly (shouted feel)
                        ([260, 520, 1040], 0.05, 0.55),
                        ([280, 560, 1120], 0.18, 0.65),
                        ([300, 600, 1200], 0.18, 0.65),
                        # "T" — sharp high-freq snap
                        ([1800, 2400], 0.05, 0.55),
                        ([1200, 1600], 0.04, 0.30),
                     ]),
        'stumps_hit':  _make([(80, 0.04, 0.7), (60, 0.06, 0.7), (40, 0.10, 0.6),
                              (200, 0.04, 0.5), (150, 0.06, 0.4)]),
        'umpire':      _make([(2000, 0.18, 0.4), (1800, 0.10, 0.35)]),
        'whoosh':      _make([(220, 0.04, 0.25), (440, 0.06, 0.30),
                              (660, 0.08, 0.30), (880, 0.06, 0.25)]),
    }

    def play(name):
        s = SND.get(name)
        if s is None:
            return
        try:
            s.play()
        except Exception as e:
            # Report once per name
            if not hasattr(play, '_failed'):
                play._failed = set()
            if name not in play._failed:
                play._failed.add(name)
                print(f"[sound] play({name!r}) failed: {e}")

    # Optional: override any sound by dropping a matching audio file next to the script.
    # Recognized: out_call.{mp3,wav,ogg}, crowd_cheer.{mp3,wav,ogg}, crowd_aww.{mp3,wav,ogg},
    # stumps_hit.{mp3,wav,ogg}, etc. The synthesized version is kept as fallback.
    _script_dir = (os.path.dirname(os.path.abspath(__file__))
                   if '__file__' in globals() else os.getcwd())
    for _name in list(SND.keys()):
        for _ext in ('mp3', 'wav', 'ogg'):
            _override_path = os.path.join(_script_dir, f"{_name}.{_ext}")
            if os.path.isfile(_override_path):
                try:
                    SND[_name] = pygame.mixer.Sound(_override_path)
                    print(f"[sound] loaded override: {_name}.{_ext}")
                    break
                except Exception as _e:
                    print(f"[sound] failed to load override {_name}.{_ext}: {_e}")

    # Confirm audio actually plays by triggering a quick startup chime
    try:
        SND['click'].play()
        print("[sound] startup chime played -- if you didn't hear it, "
              "check your system volume / output device")
    except Exception as e:
        print(f"[sound] startup chime failed: {e}")
else:
    FALLBACK = {
        'click':    '/System/Library/Sounds/Pop.aiff',
        'hover':    '/System/Library/Sounds/Tink.aiff',
        'run':      '/System/Library/Sounds/Glass.aiff',
        'four':     '/System/Library/Sounds/Glass.aiff',
        'six':      '/System/Library/Sounds/Glass.aiff',
        'wicket':   '/System/Library/Sounds/Basso.aiff',
        'out':      '/System/Library/Sounds/Basso.aiff',
        'coin':     '/System/Library/Sounds/Pop.aiff',
        'toss_win': '/System/Library/Sounds/Glass.aiff',
        'win':      '/System/Library/Sounds/Glass.aiff',
        'lose':     '/System/Library/Sounds/Basso.aiff',
        'tie':      '/System/Library/Sounds/Pop.aiff',
        'tick':     '/System/Library/Sounds/Tink.aiff',
        'crowd_cheer': '/System/Library/Sounds/Glass.aiff',
        'crowd_aww':   '/System/Library/Sounds/Basso.aiff',
        'stumps_hit':  '/System/Library/Sounds/Basso.aiff',
        'umpire':      '/System/Library/Sounds/Tink.aiff',
        'whoosh':      '/System/Library/Sounds/Pop.aiff',
    }

    def play(name):
        path = FALLBACK.get(name)
        if not path or not os.path.exists(path):
            return
        try:
            subprocess.Popen(['afplay', path],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
#  BACKGROUND MUSIC
#  Looks for bgm.mp3 / bgm.wav / bgm.ogg next to this script and loops it.
#  Pauses during celebrations and louder moments so SFX cut through.
# ─────────────────────────────────────────────────────────────────────────────

BGM_PATH = None
for _candidate in ('bgm.mp3', 'bgm.ogg', 'bgm.wav'):
    _p = os.path.join(
        os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals()
        else os.getcwd(),
        _candidate,
    )
    if os.path.isfile(_p):
        BGM_PATH = _p
        break


def bgm_start(volume=0.20):
    if not SOUND_ON or BGM_PATH is None:
        return
    try:
        pygame.mixer.music.load(BGM_PATH)
        pygame.mixer.music.set_volume(volume)
        pygame.mixer.music.play(loops=-1)
        print(f"[sound] background music loaded: {os.path.basename(BGM_PATH)}")
    except Exception as e:
        print(f"[sound] background music FAILED: {e}")


def bgm_volume(v):
    if SOUND_ON:
        try:
            pygame.mixer.music.set_volume(v)
        except Exception:
            pass


def bgm_stop():
    if SOUND_ON:
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass


if BGM_PATH is None:
    print("[sound] (no bgm.mp3/ogg/wav found next to the script; "
          "drop one in to enable background music)")


DEEP        = (18, 22, 30)
PANEL       = (32, 38, 52)
PANEL_HI    = (52, 60, 78)
LINE        = (75, 85, 105)
TEXT        = (240, 244, 248)
TEXT_DIM    = (175, 185, 200)
TEXT_MUTE   = (110, 120, 135)
PRIMARY     = (95, 220, 120)
GOLD        = (35, 200, 250)
RED         = (60, 75, 235)
BLUE        = (240, 175, 90)
PURPLE      = (220, 100, 240)


def rounded_rect(frame, pt1, pt2, color, radius=12, thickness=-1):
    x1, y1 = pt1
    x2, y2 = pt2
    radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if thickness == -1:
        cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(frame, (x1 + radius, y1 + radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x2 - radius, y1 + radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x1 + radius, y2 - radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x2 - radius, y2 - radius), radius, color, -1, cv2.LINE_AA)


def panel_bg(frame, y1, y2, alpha=0.78):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y1), (w, y2), DEEP, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.line(frame, (0, y1), (w, y1), LINE, 1, cv2.LINE_AA)
    cv2.line(frame, (0, y2), (w, y2), LINE, 1, cv2.LINE_AA)


def vignette(frame, alpha=0.55):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (5, 8, 14), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def text(frame, s, pos, scale=0.75, color=TEXT, bold=False, shadow=True, center=False):
    th = 3 if bold else 2
    if center:
        (tw, _), _ = cv2.getTextSize(str(s), cv2.FONT_HERSHEY_DUPLEX, scale, th)
        pos = (pos[0] - tw // 2, pos[1])
    if shadow:
        cv2.putText(frame, str(s), (pos[0] + 1, pos[1] + 2),
                    cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(frame, str(s), pos, cv2.FONT_HERSHEY_DUPLEX, scale, color, th, cv2.LINE_AA)


def glow_text(frame, s, pos, scale, color, bold=True, layers=4, center=False):
    th = 4 if bold else 2
    if center:
        (tw, _), _ = cv2.getTextSize(str(s), cv2.FONT_HERSHEY_DUPLEX, scale, th)
        pos = (pos[0] - tw // 2, pos[1])
    for i in range(layers, 0, -1):
        cv2.putText(frame, str(s), pos, cv2.FONT_HERSHEY_DUPLEX,
                    scale, color, th + i * 2, cv2.LINE_AA)
    cv2.putText(frame, str(s), pos, cv2.FONT_HERSHEY_DUPLEX,
                scale, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(frame, str(s), pos, cv2.FONT_HERSHEY_DUPLEX,
                scale, color, th, cv2.LINE_AA)


def draw_pill(frame, pos, label, color, scale=0.65, padding=(14, 8)):
    x, y = pos
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, scale, 2)
    x2 = x + tw + padding[0] * 2
    y1 = y - th - padding[1]
    y2 = y + padding[1]
    rounded_rect(frame, (x, y1), (x2, y2), color, radius=(y2 - y1) // 2, thickness=-1)
    cv2.putText(frame, label, (x + padding[0], y),
                cv2.FONT_HERSHEY_DUPLEX, scale, (10, 10, 10), 2, cv2.LINE_AA)
    return x2


def progress_bar(frame, prog, w, h, height=6):
    cv2.rectangle(frame, (0, h - height), (w, h), (10, 12, 18), -1)
    if prog > 0:
        col = PRIMARY if prog < 1.0 else GOLD
        cv2.rectangle(frame, (0, h - height), (int(w * prog), h), col, -1)



# --- original finger counter (temporary) ---

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

mp_draw = mp.solutions.drawing_utils


tip_ids = [4, 8, 12, 16, 20]

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

   
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    results = hands.process(rgb)

    total_fingers = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = hand_landmarks.landmark
            fingers = []

            if landmarks[4].x > landmarks[3].x:
                fingers.append(1)
            else:
                fingers.append(0)

            for i in range(1,5):
                if landmarks[tip_ids[i]].y < landmarks[tip_ids[i]-2].y:
                    fingers.append(1)
                else:
                    fingers.append(0)

            total_fingers = fingers.count(1)

    cv2.putText(
        frame,
        f"Fingers: {total_fingers}",
        (50,100),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0,255,0),
        3
    )

    cv2.imshow("Hand Cricket Input", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()