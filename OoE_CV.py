"""
HAND CRICKET - Gesture-based cricket vs an adaptive AI.

Show fingers to the webcam to play your shot / bowl your delivery.
Gestures: 1-5 = number of fingers up,  6-9 = thumb + closed fingers, 10 = cupped hand.

Controls inside the game:
  ESC / Q            - quit
  Both fists for 3s  - quit
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


class Particles:
    def __init__(self, cx, cy, count=65, gravity=0.45,
                 speed_range=(4, 15), colors=None, life_range=(35, 90)):
        if colors is None:
            colors = [(120, 255, 140), (90, 220, 255), (35, 200, 250),
                      (200, 120, 255), (95, 95, 255), (255, 255, 255)]
        self.p = []
        for _ in range(count):
            ang = random.uniform(0, 2 * np.pi)
            spd = random.uniform(*speed_range)
            lf = random.randint(*life_range)
            self.p.append({
                'x': float(cx), 'y': float(cy),
                'vx': spd * np.cos(ang),
                'vy': spd * np.sin(ang) - random.uniform(3, 9),
                'r': random.randint(4, 11),
                'c': random.choice(colors),
                'life': lf, 'ml': lf, 'g': gravity,
            })

    def update_draw(self, frame):
        alive = []
        for p in self.p:
            p['x'] += p['vx']; p['y'] += p['vy']
            p['vy'] += p['g']; p['vx'] *= 0.99
            p['life'] -= 1
            if p['life'] > 0:
                a = p['life'] / p['ml']
                r = max(1, int(p['r'] * a))
                cv2.circle(frame, (int(p['x']), int(p['y'])), r, p['c'], -1, cv2.LINE_AA)
                alive.append(p)
        self.p = alive
        return bool(self.p)


class Confetti:
    def __init__(self, w, h, count=120):
        cols = [(95, 220, 120), (35, 200, 250), (240, 175, 90),
                (220, 100, 240), (60, 75, 235), (255, 255, 255)]
        self.p = []
        for _ in range(count):
            self.p.append({
                'x': random.uniform(0, w),
                'y': random.uniform(-h, 0),
                'vx': random.uniform(-1.0, 1.0),
                'vy': random.uniform(2.5, 5.5),
                'rot': random.uniform(0, 2 * np.pi),
                'spin': random.uniform(-0.25, 0.25),
                'sz': random.randint(5, 11),
                'c': random.choice(cols),
            })
        self.h = h

    def update_draw(self, frame):
        h = self.h
        for p in self.p:
            p['x'] += p['vx']; p['y'] += p['vy']; p['rot'] += p['spin']
            if p['y'] > h + 20:
                p['y'] = -10
                p['x'] = random.uniform(0, frame.shape[1])
            cx, cy = int(p['x']), int(p['y'])
            s = p['sz']
            c, s_ = np.cos(p['rot']), np.sin(p['rot'])
            pts = np.array([[-s, -s // 2], [s, -s // 2], [s, s // 2], [-s, s // 2]],
                           dtype=np.float32)
            R = np.array([[c, -s_], [s_, c]], dtype=np.float32)
            pts = pts @ R.T
            pts[:, 0] += cx; pts[:, 1] += cy
            cv2.fillPoly(frame, [pts.astype(np.int32)], p['c'], cv2.LINE_AA)


class FX:
    def __init__(self):
        self.q = []
        self.confetti = None

    def trigger(self, kind, **kw):
        self.q.append({'k': kind, 't': 0, **kw})

    def confetti_on(self, w, h):
        self.confetti = Confetti(w, h, count=140)

    def clear(self):
        self.q = []
        self.confetti = None

    def draw(self, frame):
        h, w = frame.shape[:2]
        alive = []
        if self.confetti is not None:
            self.confetti.update_draw(frame)
        for a in self.q:
            t = a['t']; k = a['k']

            if k == 'OUT':
                DUR = 95
                if t >= DUR: continue
                a_blend = 0.32 * (1 - t / DUR)
                ov = frame.copy()
                cv2.rectangle(ov, (0, 0), (w, h), (35, 35, 215), -1)
                cv2.addWeighted(ov, a_blend, frame, 1 - a_blend, 0, frame)
                if t < 22:
                    sh = int(10 * (1 - t / 22))
                    M = np.float32([[1, 0, random.randint(-sh, sh)],
                                    [0, 1, random.randint(-sh, sh)]])
                    frame[:] = cv2.warpAffine(frame, M, (w, h))
                # NOTE: "OUT!" text removed here — the CELEBRATION screen renders it.
                alive.append({**a, 't': t + 1})

            elif k == 'BIGRUN':
                DUR = 115
                if t >= DUR: continue
                ps = a.get('_ps')
                if ps is None:
                    ps = Particles(w // 2, h // 2, count=42, speed_range=(5, 17))
                    a['_ps'] = ps
                if t < 18:
                    a_blend = 0.22 * (1 - t / 18)
                    ov = frame.copy()
                    cv2.rectangle(ov, (0, 0), (w, h), (35, 200, 250), -1)
                    cv2.addWeighted(ov, a_blend, frame, 1 - a_blend, 0, frame)
                ps.update_draw(frame)
                lbl = a.get('lbl', 'BIG HIT!')
                glow_text(frame, lbl, (w // 2, h // 3),
                          1.9, GOLD, bold=True, layers=3, center=True)
                alive.append({**a, 't': t + 1})

            elif k == 'RUN_POP':
                DUR = 50
                if t >= DUR: continue
                runs = a.get('runs', 1)
                p = t / DUR
                y = int(h * 0.45 - p * 70)
                alpha = 1 - p
                col = a.get('col', PRIMARY)
                tmp = frame.copy()
                glow_text(tmp, f"+{runs}", (w // 2, y),
                          1.7, col, bold=True, layers=3, center=True)
                cv2.addWeighted(tmp, alpha, frame, 1 - alpha, 0, frame)
                alive.append({**a, 't': t + 1})

            elif k == 'FLASH':
                DUR = 30
                if t >= DUR: continue
                ov = frame.copy()
                cv2.rectangle(ov, (0, 0), (w, h), a.get('col', (95, 220, 120)), -1)
                a_blend = 0.13 * (1 - t / DUR)
                cv2.addWeighted(ov, a_blend, frame, 1 - a_blend, 0, frame)
                alive.append({**a, 't': t + 1})

            elif k == 'WIN':
                DUR = 220
                if t >= DUR: continue
                ps = a.get('_ps')
                if ps is None:
                    ps = Particles(w // 2, int(h * 0.35), count=55, speed_range=(5, 18))
                    a['_ps'] = ps
                ps.update_draw(frame)
                if t in (40, 80, 130, 170):
                    self.q.append({'k': '_BURST', 't': 0,
                                   'cx': random.randint(int(w * 0.2), int(w * 0.8)),
                                   'cy': random.randint(int(h * 0.2), int(h * 0.5))})
                alive.append({**a, 't': t + 1})

            elif k == '_BURST':
                DUR = 35
                if t >= DUR: continue
                ps = a.get('_ps')
                if ps is None:
                    ps = Particles(a['cx'], a['cy'], count=20, speed_range=(3, 9))
                    a['_ps'] = ps
                ps.update_draw(frame)
                alive.append({**a, 't': t + 1})

        self.q = alive


fx = FX()


HAND_CONN = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
             (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
             (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

G_MAP = {
    (0, 1, 0, 0, 0): 1, (0, 1, 1, 0, 0): 2, (0, 1, 1, 1, 0): 3,
    (0, 1, 1, 1, 1): 4, (1, 1, 1, 1, 1): 5, (1, 0, 0, 0, 0): 6,
    (1, 1, 0, 0, 0): 7, (1, 1, 1, 0, 0): 8, (1, 1, 1, 1, 0): 9,
}


def _d(a, b):
    return float(np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2))


def fstates(lms):
    hs = _d(lms[0], lms[9])
    if hs < 0.001:
        return (0, 0, 0, 0, 0)
    t = 1 if _d(lms[4], lms[5]) > hs * 0.45 else 0
    return tuple([t] + [1 if _d(lms[i], lms[0]) > _d(lms[i - 2], lms[0])
                        else 0 for i in [8, 12, 16, 20]])


def is_cup(hl):
    return len(hl) >= 2 and all(sum(fstates(l)) <= 1 for l in hl)


def raw_gesture(mpr, valid):
    if not mpr.hand_landmarks:
        return None
    if 10 in valid and is_cup(mpr.hand_landmarks):
        return 10
    if len(mpr.hand_landmarks) == 1:
        v = G_MAP.get(fstates(mpr.hand_landmarks[0]))
        if v is not None and v in valid:
            return v
    return None


def draw_hands(frame, mpr, w, h):
    if not mpr.hand_landmarks:
        return
    for lms in mpr.hand_landmarks:
        for a, b in HAND_CONN:
            cv2.line(frame,
                     (int(lms[a].x * w), int(lms[a].y * h)),
                     (int(lms[b].x * w), int(lms[b].y * h)),
                     (0, 200, 255), 2, cv2.LINE_AA)
        for lm in lms:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)),
                       4, (255, 255, 255), -1, cv2.LINE_AA)


class GestureConfirm:
    def __init__(self, hold=1.30):
        self.hold = hold
        self.cur = None
        self.t0 = None
        self._last_tick = 0

    def update(self, raw):
        now = time.time()
        if raw != self.cur:
            self.cur = raw
            self.t0 = now if raw is not None else None
            self._last_tick = 0
            return None, 0.0
        if self.cur is None:
            return None, 0.0
        e = now - self.t0
        p = min(e / self.hold, 1.0)
        if e >= self.hold:
            v = self.cur
            self.reset()
            return v, 1.0
        return None, p

    def reset(self):
        self.cur = None
        self.t0 = None
        self._last_tick = 0


class Brain:
    def __init__(self, diff):
        self.diff = diff
        self.hist = []

    def pick(self, rng, is_bowling):
        rng = list(rng)
        if self.diff == 'EASY':
            # EASY: still random, but with slight high-number bias (1.2x weight on higher numbers)
            weights = [1.0 + 0.15 * (n - rng[0]) / max(1, rng[-1] - rng[0])
                       for n in rng]
            return random.choices(rng, weights=weights)[0]
        
        predict = None
        if len(self.hist) >= 3:
            predict = Counter(self.hist[-6:]).most_common(1)[0][0]
            if predict not in rng:
                predict = None
        
        if self.diff == 'MEDIUM':
            if is_bowling and predict and random.random() < 0.28:
                return predict
            if not is_bowling and predict and random.random() < 0.22:
                others = [n for n in rng if n != predict]
                if others:
                    return random.choice(others)
            # Weighted toward middle + high numbers (moderate aggression)
            mid = (rng[0] + rng[-1]) / 2
            w = [1.5 / (abs(n - mid) + 0.6) for n in rng]
            # Boost high numbers by 30%
            w = [w[i] * (1.0 + 0.30 * (rng[i] - rng[0]) / max(1, rng[-1] - rng[0]))
                 for i in range(len(rng))]
            return random.choices(rng, weights=w)[0]
        else:  # HARD
            if is_bowling and predict and random.random() < 0.52:
                return predict
            if not is_bowling and predict and random.random() < 0.42:
                others = [n for n in rng if n != predict]
                if others:
                    return random.choice(others)
            # HARD: strong bias toward high numbers (greedy, aggressive)
            weights = [1.0 + 0.50 * (n - rng[0]) / max(1, rng[-1] - rng[0])
                       for n in rng]
            return random.choices(rng, weights=weights)[0]

    def record(self, v):
        self.hist.append(v)
        self.hist = self.hist[-12:]


COIN_DUR = 45  # ~1.5 seconds at 30 fps


def draw_coin(frame, t, user_n, comp_n, w, h):
    cx, cy = w // 2, h // 2 - 20
    R = 92
    LAND = COIN_DUR - 15  # land 15 frames (~0.5s) before the end

    vignette(frame, alpha=0.62)
    glow_text(frame, "COIN  TOSS", (w // 2, 80),
              1.3, GOLD, bold=True, layers=3, center=True)
    cv2.ellipse(frame, (cx + 6, cy + 12), (R, R // 5), 0, 0, 360,
                (10, 12, 16), -1, cv2.LINE_AA)

    if t < LAND:
        prog = t / LAND
        period = max(2, int(12 * (1 - prog * 0.85)))
        ph = (t % period) / period
        xr = max(6, int(R * abs(2 * ph - 1)))
        front = ph < 0.5
        cv2.ellipse(frame, (cx, cy), (xr, R), 0, 0, 360,
                    (8, 145, 210) if front else (4, 110, 180), -1, cv2.LINE_AA)
        cv2.ellipse(frame, (cx, cy), (max(2, int(xr * 0.85)), int(R * 0.92)),
                    0, 0, 360,
                    (35, 200, 250) if front else (25, 165, 220), -1, cv2.LINE_AA)
        cv2.ellipse(frame, (cx, cy), (xr, R), 0, 0, 360,
                    (60, 230, 255), 3, cv2.LINE_AA)
        if xr > 30:
            # Show a "?" while flipping, not the actual numbers — keeps suspense
            glow_text(frame, "?", (cx, cy + 18),
                      2.4, (255, 255, 255), bold=True, layers=2, center=True)
        text(frame, "Flipping...", (w // 2, cy + R + 78),
             0.85, TEXT_DIM, center=True)
    else:
        # Landed: reveal both numbers
        age = t - LAND
        rr = R + age * 2
        al = max(0.0, 1 - age / 15)
        cv2.ellipse(frame, (cx, cy), (rr, rr // 2), 0, 0, 360,
                    (35, 200, 250), max(1, int(4 * al)), cv2.LINE_AA)
        cv2.ellipse(frame, (cx, cy), (R, R), 0, 0, 360,
                    (8, 145, 210), -1, cv2.LINE_AA)
        cv2.ellipse(frame, (cx, cy), (int(R * 0.86), int(R * 0.86)), 0, 0, 360,
                    (35, 200, 250), -1, cv2.LINE_AA)
        cv2.ellipse(frame, (cx, cy), (R, R), 0, 0, 360,
                    (90, 230, 255), 4, cv2.LINE_AA)
        for num, xo, lbl in [(user_n, -55, "YOU"), (comp_n, 55, "CPU")]:
            glow_text(frame, str(num), (cx + xo, cy + 14),
                      1.9, (255, 255, 255), bold=True, layers=2, center=True)
            text(frame, lbl, (cx + xo, cy + R + 38), 0.7, TEXT_DIM, center=True)
        text(frame, f"You: {user_n}     CPU: {comp_n}",
             (w // 2, cy + R + 78), 0.85, TEXT_DIM, center=True)


def topbar(frame, g, w, dlbl, dkey):
    BAR_H = 86
    panel_bg(frame, 0, BAR_H, alpha=0.86)
    glow_text(frame, "HAND CRICKET", (18, 34), 0.78, PRIMARY, bold=True, layers=2)
    mode_s = g.get('mode') or ''
    if mode_s:
        text(frame, mode_s, (18, 64), 0.62, TEXT_DIM)
    dcol = {'EASY': PRIMARY, 'MEDIUM': GOLD, 'HARD': RED}.get(dkey, TEXT)
    pill_x = w // 2 - 70
    draw_pill(frame, (pill_x, 38), f"CPU  {dlbl.strip()}", dcol, scale=0.55)
    s1 = '---' if g['innings_scores'][0] is None else str(g['innings_scores'][0])
    s2 = '---' if g['innings_scores'][1] is None else str(g['innings_scores'][1])
    box_w = 260
    bx1 = w - box_w - 14
    rounded_rect(frame, (bx1, 12), (w - 14, 74), PANEL, 10, -1)
    rounded_rect(frame, (bx1, 12), (w - 14, 74), LINE, 10, 1)
    text(frame, "1st", (bx1 + 18, 36), 0.55, TEXT_MUTE)
    text(frame, "2nd", (bx1 + 18, 64), 0.55, TEXT_MUTE)
    text(frame, s1, (bx1 + 70, 38), 0.78, TEXT, bold=True)
    text(frame, s2, (bx1 + 70, 66), 0.78, TEXT, bold=True)
    if g['innings'] == 1 and g['innings_scores'][0] is not None:
        target = g['innings_scores'][0] + 1
        text(frame, f"TARGET {target}", (bx1 + 140, 38), 0.58, GOLD, bold=True)
        need = target - g['current_runs']
        if need > 0:
            text(frame, f"NEED {need}", (bx1 + 140, 66), 0.58, TEXT_DIM)
        else:
            text(frame, "WON!", (bx1 + 140, 66), 0.58, PRIMARY, bold=True)


def gesture_indicator(frame, raw, prog, w):
    cx, cy, R = w - 50, 120, 28
    rounded_rect(frame, (cx - R - 6, cy - R - 6),
                 (cx + R + 6, cy + R + 6), PANEL, 12, -1)
    cv2.circle(frame, (cx, cy), R, DEEP, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), R, LINE, 1, cv2.LINE_AA)
    if prog > 0:
        cv2.ellipse(frame, (cx, cy), (R - 2, R - 2),
                    -90, 0, int(360 * prog),
                    PRIMARY if prog < 1.0 else GOLD, 4, cv2.LINE_AA)
    lbl = str(raw) if raw is not None else '.'
    text(frame, lbl, (cx, cy + 8), 1.05, GOLD, bold=True, center=True)
    text(frame, "HOLD", (cx, cy + R + 22), 0.42, TEXT_MUTE, center=True)


def resolve(bat, bowl, mode):
    if mode == 'CRAZY':
        d = abs(bat - bowl)
        if d == 1 or d == 9:
            return 0, True, f"OUT! Adjacent ({bat} vs {bowl})"
        if bat == bowl:
            r = bat ** 2
            return r, False, f"SQUARED! {bat}^2={r}"
        return bat, False, f"+{bat}"
    if bat == bowl:
        return 0, True, f"OUT! Both chose {bat}"
    return bat, False, f"+{bat}"


def new_game():
    return {
        'state': 'MODE_SELECT',
        'mode': None, 'run_range': None,
        'toss_choice': None, 'toss_user_num': None, 'toss_comp_num': None,
        'toss_total': None, 'toss_result_str': None,
        'user_won_toss': False, 'toss_comp_dec': None,
        'user_batting': None, 'batting_first': None,
        'current_runs': 0, 'innings': 0, 'innings_scores': [None, None],
        'is_super_over': False, 'super_balls': 6,
        'ball_msg': '', 'ball_out': False, 'ball_runs': 0,
        'you_showed': None, 'comp_showed': None,
        'ball_timer': None, 'coin_timer': 0,
        'result_sound_played': False,
        # celebration triggers
        'fifty_shown': False, 'hundred_shown': False,
        'pending_celebration': None,
        'match_celebration_shown': False,
        'celebration_start': None,
        'celebration_acknowledged': False,
    }


def end_of_ball_update(g):
    """Decide where the match goes after a ball has been resolved.

    Returns a dict of updates for the game state (it does not modify `g`):
    back to PLAYING, an INNINGS_BREAK (including the start of a super over),
    or the final GAME_RESULT.

    Rules:
      * Normal innings end on the first wicket.
      * A super over is 6 balls per side; wickets cost no runs but do not
        end the innings.
      * A chase (2nd innings) ends as soon as the target is passed.
      * A tie after two normal innings starts a super over (user bats first).
    """
    first = g['innings_scores'][0]
    chasing = g['innings'] == 1 and first is not None
    target_passed = chasing and g['current_runs'] > first

    upd = {}
    if g['is_super_over']:
        balls_left = g['super_balls'] - 1
        upd['super_balls'] = balls_left
        over = balls_left <= 0 or target_passed
    else:
        over = g['ball_out'] or target_passed

    if not over:
        upd['state'] = 'PLAYING'
        return upd

    if g['innings'] == 0:
        upd.update({
            'innings_scores': [g['current_runs'], None],
            'innings': 1,
            'user_batting': not g['user_batting'],
            'current_runs': 0,
            'state': 'INNINGS_BREAK',
        })
        if g['is_super_over']:
            upd['super_balls'] = 6
        return upd

    scores = [first, g['current_runs']]
    if scores[0] == scores[1] and not g['is_super_over']:
        upd.update({
            'innings_scores': [None, None],
            'is_super_over': True, 'innings': 0,
            'current_runs': 0, 'super_balls': 6,
            'user_batting': True, 'batting_first': 'user',
            'state': 'INNINGS_BREAK',
        })
        return upd

    upd.update({'innings_scores': scores, 'state': 'GAME_RESULT'})
    return upd


def check_milestone(g):
    """Return 'HUNDRED' or 'FIFTY' the first time the score reaches it, else None.

    Checks 100 first so a single big ball (e.g. 10^2 in CRAZY mode) that jumps
    past both milestones celebrates the hundred rather than the fifty."""
    runs = g['current_runs']
    if not g['hundred_shown'] and runs >= 100:
        g['hundred_shown'] = g['fifty_shown'] = True
        return 'HUNDRED'
    if not g['fifty_shown'] and runs >= 50:
        g['fifty_shown'] = True
        return 'FIFTY'
    return None


def screen_mode_select(frame, conf, w, h, t):
    vignette(frame, alpha=0.7)
    pulse = 1.0 + 0.04 * np.sin(t * 0.08)
    glow_text(frame, "HAND CRICKET", (w // 2, 100),
              1.9 * pulse, PRIMARY, bold=True, layers=4, center=True)
    text(frame, "Pick your match format",
         (w // 2, 140), 0.78, TEXT_DIM, center=True)

    cards = [
        ("CLASSIC",  "1 finger",  "Numbers 1 to 6",                       PRIMARY),
        ("EXTENDED", "2 fingers", "Numbers 1 to 10",                      GOLD),
        ("CRAZY",    "3 fingers", "Adjacent=OUT, same=squared",           PURPLE),
    ]
    card_h = 84
    spacing = 14
    total_h = len(cards) * card_h + (len(cards) - 1) * spacing
    y0 = h // 2 - total_h // 2 + 30

    for i, (title_, ges, sub, col) in enumerate(cards):
        y = y0 + i * (card_h + spacing)
        x1, x2 = 50, w - 50
        rounded_rect(frame, (x1, y), (x2, y + card_h), PANEL, 14, -1)
        rounded_rect(frame, (x1, y), (x2, y + card_h), LINE, 14, 1)
        rounded_rect(frame, (x1, y), (x1 + 6, y + card_h), col, 3, -1)
        bx = x1 + 28
        cv2.circle(frame, (bx + 18, y + card_h // 2), 22, col, -1, cv2.LINE_AA)
        text(frame, str(i + 1), (bx + 18, y + card_h // 2 + 9),
             0.95, (10, 10, 10), bold=True, center=True)
        text(frame, title_, (bx + 60, y + 36), 0.85, col, bold=True)
        text(frame, sub,    (bx + 60, y + 64), 0.62, TEXT_DIM)
        (tw, _), _ = cv2.getTextSize(ges, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)
        text(frame, ges, (x2 - tw - 28, y + card_h // 2 + 8), 0.65, TEXT)

    text(frame, "Hold the gesture until the ring fills",
         (w // 2, h - 28), 0.7, TEXT_MUTE, center=True)


def screen_difficulty(frame, conf, w, h, t):
    vignette(frame, alpha=0.7)
    glow_text(frame, "DIFFICULTY", (w // 2, 100),
              1.6, GOLD, bold=True, layers=3, center=True)
    text(frame, "How smart should the AI be?",
         (w // 2, 140), 0.74, TEXT_DIM, center=True)

    info = [
        ("EASY",   "1 finger",  "Random: pure fun",                       PRIMARY, 1),
        ("MEDIUM", "2 fingers", "AI notices your patterns",               GOLD,    2),
        ("HARD",   "3 fingers", "AI tracks and counters you",             RED,     3),
    ]
    card_h = 84
    spacing = 14
    total_h = len(info) * card_h + (len(info) - 1) * spacing
    y0 = h // 2 - total_h // 2 + 30

    for i, (name, ges, desc, col, stars) in enumerate(info):
        y = y0 + i * (card_h + spacing)
        x1, x2 = 50, w - 50
        rounded_rect(frame, (x1, y), (x2, y + card_h), PANEL, 14, -1)
        rounded_rect(frame, (x1, y), (x2, y + card_h), LINE, 14, 1)
        rounded_rect(frame, (x1, y), (x1 + 6, y + card_h), col, 3, -1)
        text(frame, name, (x1 + 30, y + 36), 0.88, col, bold=True)
        text(frame, desc, (x1 + 30, y + 64), 0.6, TEXT_DIM)
        sx = x1 + 250
        for s in range(3):
            scol = col if s < stars else (60, 65, 80)
            cx_, cy_ = sx + s * 28, y + 38
            pts = []
            for k in range(10):
                ang = -np.pi / 2 + k * np.pi / 5
                rr = 11 if k % 2 == 0 else 5
                pts.append((cx_ + int(rr * np.cos(ang)),
                            cy_ + int(rr * np.sin(ang))))
            cv2.fillPoly(frame, [np.array(pts, dtype=np.int32)], scol, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(ges, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)
        text(frame, ges, (x2 - tw - 28, y + card_h // 2 + 8), 0.65, TEXT)

    text(frame, "Hold the gesture until the ring fills",
         (w // 2, h - 28), 0.7, TEXT_MUTE, center=True)


def screen_toss_choice(frame, w, h):
    panel_bg(frame, h - 230, h - 14, alpha=0.85)
    glow_text(frame, "TOSS CALL IT", (40, h - 192), 0.92, GOLD, bold=True, layers=2)
    text(frame, "Show 1 finger=ODD, 2 fingers=EVEN",
         (40, h - 142), 0.72, TEXT_DIM)
    rounded_rect(frame, (40, h - 110), (220, h - 50), PANEL_HI, 12, -1)
    rounded_rect(frame, (40, h - 110), (220, h - 50), PRIMARY, 12, 1)
    text(frame, "1  =  ODD", (60, h - 70), 0.85, TEXT, bold=True)
    rounded_rect(frame, (240, h - 110), (420, h - 50), PANEL_HI, 12, -1)
    rounded_rect(frame, (240, h - 110), (420, h - 50), GOLD, 12, 1)
    text(frame, "2  =  EVEN", (260, h - 70), 0.85, TEXT, bold=True)


def screen_toss_number(frame, g, w, h):
    panel_bg(frame, h - 200, h - 14, alpha=0.85)
    text(frame, f"YOUR CALL: {g['toss_choice']}",
         (40, h - 165), 0.85, PRIMARY, bold=True)
    text(frame, "Show 1 or 2 fingers for your toss number",
         (40, h - 115), 0.72, TEXT_DIM)
    text(frame, "(CPU picks 1 or 2 too. Sum decides.)",
         (40, h - 78), 0.6, TEXT_MUTE)


def screen_toss_result(frame, g, conf, w, h):
    panel_bg(frame, h - 280, h - 14, alpha=0.88)
    text(frame,
         f"You: {g['toss_user_num']}  CPU: {g['toss_comp_num']}  =  {g['toss_total']}  ({g['toss_result_str']})",
         (40, h - 248), 0.78, TEXT)
    if g['user_won_toss']:
        glow_text(frame, "YOU WON THE TOSS!",
                  (40, h - 198), 1.0, PRIMARY, bold=True, layers=2)
        text(frame, "1 finger=BAT     2 fingers=BOWL",
             (40, h - 145), 0.82, GOLD, bold=True)
        text(frame, "Hold to confirm",
             (40, h - 92), 0.65, TEXT_MUTE)
    else:
        text(frame, "You lost the toss.",
             (40, h - 198), 0.9, RED)
        text(frame, f"CPU chose to {g['toss_comp_dec']}!",
             (40, h - 145), 0.85, GOLD, bold=True)
        text(frame, "Show any gesture to begin",
             (40, h - 92), 0.7, TEXT_MUTE)


def screen_play(frame, g, dlbl, dkey, w, h):
    rng = g['run_range']
    role = "BATTING" if g['user_batting'] else "BOWLING"
    rcol = PRIMARY if g['user_batting'] else BLUE
    panel_bg(frame, h - 240, h - 14, alpha=0.86)
    pill_w = 240
    rounded_rect(frame, (32, h - 230), (32 + pill_w, h - 188), DEEP, 10, -1)
    rounded_rect(frame, (32, h - 230), (32 + pill_w, h - 188), rcol, 10, 1)
    cv2.circle(frame, (52, h - 209), 8, rcol, -1, cv2.LINE_AA)
    text(frame, f"YOU: {role}", (70, h - 200), 0.78, rcol, bold=True)
    rounded_rect(frame, (w - 250, h - 230), (w - 32, h - 188), DEEP, 10, -1)
    rounded_rect(frame, (w - 250, h - 230), (w - 32, h - 188), GOLD, 10, 1)
    text(frame, "RUNS", (w - 232, h - 200), 0.55, TEXT_DIM)
    text(frame, str(g['current_runs']), (w - 60, h - 200),
         1.05, GOLD, bold=True)
    if g['is_super_over']:
        text(frame, f"SUPER OVER  Balls left: {g['super_balls']}",
             (40, h - 158), 0.74, GOLD, bold=True)
    elif g['innings'] == 1 and g['innings_scores'][0] is not None:
        target = g['innings_scores'][0] + 1
        need = target - g['current_runs']
        text(frame, f"TARGET {target}   NEED {need}",
             (40, h - 158), 0.74, GOLD, bold=True)
    text(frame, f"Show {rng[0]} to {rng[-1]}", (40, h - 115), 0.74, TEXT)
    if g['mode'] == 'CRAZY':
        text(frame, "Adjacent=OUT  Same number=runs squared",
             (40, h - 75), 0.6, PURPLE)
    else:
        text(frame, "Same number = OUT", (40, h - 75), 0.6, TEXT_MUTE)
    if dkey == 'HARD':
        text(frame, "AI is reading you...", (40, h - 38), 0.55, (140, 130, 230))


def screen_ball_result(frame, g, w, h, pause_seconds):
    panel_bg(frame, h - 250, h - 14, alpha=0.88)
    col = RED if g['ball_out'] else PRIMARY
    glow_text(frame, g['ball_msg'], (40, h - 220), 0.92, col, bold=True, layers=2)
    u = "You" if g['user_batting'] else "CPU"
    c = "CPU" if g['user_batting'] else "You"
    text(frame, f"{u}: {g['you_showed']}     {c}: {g['comp_showed']}",
         (40, h - 168), 0.78, TEXT_DIM)
    text(frame, f"Innings total: {g['current_runs']}",
         (40, h - 122), 0.82, GOLD, bold=True)
    if g['innings'] == 1 and g['innings_scores'][0] is not None:
        text(frame, f"Target {g['innings_scores'][0] + 1}",
             (40, h - 80), 0.74, TEXT)
    start = g.get('ball_timer') or time.time()
    elapsed = time.time() - start
    remaining = max(0.0, pause_seconds - elapsed)
    bar_w = 220
    bx = w - bar_w - 30
    by = h - 50
    cv2.rectangle(frame, (bx, by), (bx + bar_w, by + 8), PANEL, -1)
    p = 1 - remaining / pause_seconds if pause_seconds > 0 else 1
    cv2.rectangle(frame, (bx, by), (bx + int(bar_w * p), by + 8), PRIMARY, -1)
    text(frame, f"Next ball in {remaining:.1f}s",
         (bx, by - 6), 0.55, TEXT_MUTE)


def screen_innings_break(frame, g, w, h):
    panel_bg(frame, h - 240, h - 14, alpha=0.88)
    if g['is_super_over'] and g['innings'] == 0:
        glow_text(frame, "TIE! SUPER OVER!",
                  (40, h - 210), 1.05, GOLD, bold=True, layers=2)
        text(frame, "6 balls each. Outs cost no runs.",
             (40, h - 158), 0.62, TEXT_DIM)
        text(frame, "You bat first.", (40, h - 110), 0.78, PRIMARY, bold=True)
    elif g['is_super_over'] and g['innings'] == 1:
        s = g['innings_scores'][0]
        text(frame, f"Super Over: 1st score {s}",
             (40, h - 210), 0.85, GOLD, bold=True)
        role = "BATTING" if g['user_batting'] else "BOWLING"
        rcol = PRIMARY if g['user_batting'] else BLUE
        text(frame, f"You are now {role}",
             (40, h - 158), 0.85, rcol, bold=True)
        text(frame, f"Target {s + 1}",
             (40, h - 110), 0.75, GOLD, bold=True)
    else:
        s = g['innings_scores'][0]
        who = "Your" if g['batting_first'] == 'user' else "CPU's"
        glow_text(frame, "INNINGS OVER",
                  (40, h - 210), 0.95, GOLD, bold=True, layers=2)
        text(frame, f"{who} score: {s}",
             (40, h - 162), 0.78, TEXT)
        role = "BATTING" if g['user_batting'] else "BOWLING"
        rcol = PRIMARY if g['user_batting'] else BLUE
        text(frame, f"You are now {role}: Target {s + 1}",
             (40, h - 110), 0.78, rcol, bold=True)
    text(frame, "Show any gesture to continue",
         (40, h - 56), 0.7, TEXT_MUTE)


def screen_game_result(frame, g, dlbl, w, h):
    vignette(frame, alpha=0.78)
    s1, s2 = g['innings_scores']
    user_s, cpu_s = (s1, s2) if g['batting_first'] == 'user' else (s2, s1)
    if user_s is None or cpu_s is None:
        return
    if user_s > cpu_s:
        glow_text(frame, "YOU WIN!",
                  (w // 2, h // 2 - 40), 2.4, PRIMARY,
                  bold=True, layers=4, center=True)
        margin = user_s - cpu_s
        text(frame, f"By {margin} run{'s' if margin != 1 else ''}",
             (w // 2, h // 2 + 5), 0.85, PRIMARY, center=True)
    elif cpu_s > user_s:
        glow_text(frame, "YOU LOSE",
                  (w // 2, h // 2 - 40), 2.2, RED,
                  bold=True, layers=4, center=True)
        margin = cpu_s - user_s
        text(frame, f"CPU won by {margin} run{'s' if margin != 1 else ''}",
             (w // 2, h // 2 + 5), 0.85, RED, center=True)
    else:
        glow_text(frame, "STILL A TIE",
                  (w // 2, h // 2 - 40), 1.9, GOLD,
                  bold=True, layers=4, center=True)
    pre = "Super Over: " if g['is_super_over'] else ""
    text(frame, f"{pre}You {user_s}  CPU {cpu_s}",
         (w // 2, h // 2 + 50), 0.95, TEXT, bold=True, center=True)
    text(frame, f"Difficulty: {dlbl.strip()}  Mode: {g['mode']}",
         (w // 2, h // 2 + 90), 0.7, TEXT_DIM, center=True)
    panel_bg(frame, h - 90, h - 14, alpha=0.85)
    rounded_rect(frame, (w // 2 - 260, h - 70), (w // 2 - 20, h - 26),
                 PANEL_HI, 12, -1)
    rounded_rect(frame, (w // 2 - 260, h - 70), (w // 2 - 20, h - 26),
                 PRIMARY, 12, 1)
    text(frame, "1  PLAY AGAIN",
         (w // 2 - 240, h - 40), 0.78, TEXT, bold=True)
    rounded_rect(frame, (w // 2 + 20, h - 70), (w // 2 + 260, h - 26),
                 PANEL_HI, 12, -1)
    rounded_rect(frame, (w // 2 + 20, h - 70), (w // 2 + 260, h - 26),
                 RED, 12, 1)
    text(frame, "2  QUIT", (w // 2 + 100, h - 40), 0.78, TEXT, bold=True)


# ─────────────────────────────────────────────────────────────────────────────
#  CELEBRATION SCREEN
# ─────────────────────────────────────────────────────────────────────────────

CELEBRATION_CONFIG = {
    # User got out (batting): stumps hit + OUT! call + crowd aww
    'OUT_BATTING': {
        'label': 'OUT!',
        'color': (60, 60, 255),
        # Sequence: [(sound_name, delay_in_seconds), ...]
        'auto_sounds': [
            ('stumps_hit', 0.00),   # immediate impact
            ('out_call',   0.15),   # "OUT!" call right after
            ('crowd_aww',  0.55),   # crowd reacts
        ],
        'subtext': None,
        'prompt': None,
        'duration': 2.8,
    },
    # User took a wicket (bowling): stumps + OUT! call + cheer (raise hand for extra cheer)
    'WICKET_BOWLING': {
        'label': 'WICKET!',
        'color': (95, 220, 120),
        'auto_sounds': [
            ('stumps_hit',  0.00),
            ('out_call',    0.15),
            ('crowd_cheer', 0.50),
        ],
        'subtext': None,
        'prompt': 'RAISE YOUR HAND TO CELEBRATE',
        'cheer_sound': 'crowd_cheer',
        'duration': 3.5,
    },
    'FIFTY': {
        'label': 'FIFTY!',
        'color': (35, 200, 250),
        'auto_sounds': None,
        'subtext': '50 RUNS',
        'prompt': 'RAISE YOUR HAND TO CELEBRATE',
        'cheer_sound': 'crowd_cheer',
        'duration': 3.5,
    },
    'HUNDRED': {
        'label': 'HUNDRED!',
        'color': (95, 220, 120),
        'auto_sounds': None,
        'subtext': '100 RUNS: CENTURY!',
        'prompt': 'RAISE YOUR HAND TO CELEBRATE',
        'cheer_sound': 'crowd_cheer',
        'duration': 4.0,
    },
    'MATCH_WIN': {
        'label': 'YOU WIN!',
        'color': (95, 220, 120),
        'auto_sounds': None,
        'subtext': 'Victory!',
        'prompt': 'RAISE YOUR HAND TO CELEBRATE',
        'cheer_sound': 'crowd_cheer',
        'duration': 4.5,
    },
    'MATCH_LOSE': {
        'label': 'DEFEAT',
        'color': (60, 75, 235),
        'auto_sounds': [('crowd_aww', 0.0)],
        'subtext': 'Better luck next time',
        'prompt': None,
        'duration': 3.0,
    },
}


def is_hand_raised(mpr):
    """Detect: any hand landmark with the wrist well above the bottom of the frame
    AND the hand mostly facing up (index tip above wrist)."""
    if not mpr.hand_landmarks:
        return False
    for lms in mpr.hand_landmarks:
        # index fingertip (8) above wrist (0) by at least 15% of frame height
        if lms[8].y < lms[0].y - 0.15:
            return True
    return False


def screen_celebration(frame, g, conf_kind, w, h, mpr, frame_idx):
    """Renders the celebration / out screen for the given kind.
    Returns True when the celebration is finished and game can progress.
    """
    cfg = CELEBRATION_CONFIG[conf_kind]
    vignette(frame, alpha=0.65)

    # Pulsing big label
    pulse = 1.0 + 0.06 * np.sin(frame_idx * 0.18)
    glow_text(frame, cfg['label'], (w // 2, int(h * 0.40)),
              2.2 * pulse, cfg['color'], bold=True, layers=4, center=True)

    if cfg.get('subtext'):
        text(frame, cfg['subtext'], (w // 2, int(h * 0.50)),
             0.9, TEXT, center=True)

    elapsed = time.time() - (g.get('celebration_start') or time.time())

    # Auto-sounds: play each sound at its scheduled delay
    sounds_played = g.setdefault('_celebration_sounds_played', set())
    auto_sounds = cfg.get('auto_sounds')
    if auto_sounds:
        for idx, (snd, delay) in enumerate(auto_sounds):
            if idx not in sounds_played and elapsed >= delay:
                print(f"[debug] celebration sound: {snd} (elapsed={elapsed:.2f}s, kind={conf_kind})")
                play(snd)
                sounds_played.add(idx)

    if cfg.get('prompt'):
        if not g.get('celebration_acknowledged'):
            pulse_txt = 0.6 + 0.4 * abs(np.sin(frame_idx * 0.12))
            col = tuple(int(c * pulse_txt + 240 * (1 - pulse_txt))
                        for c in cfg['color'])
            glow_text(frame, cfg['prompt'], (w // 2, int(h * 0.62)),
                      1.0, col, bold=True, layers=2, center=True)
            text(frame, "(or wait to skip)", (w // 2, int(h * 0.70)),
                 0.6, TEXT_MUTE, center=True)
            if is_hand_raised(mpr):
                if cfg.get('cheer_sound'):
                    print(f"[debug] hand-raise cheer: {cfg['cheer_sound']} (kind={conf_kind})")
                    play(cfg['cheer_sound'])
                g['celebration_acknowledged'] = True
                g['celebration_start'] = time.time() - cfg['duration'] * 0.55
                fx.trigger('FLASH', col=cfg['color'])
                fx.confetti_on(w, h)

    return elapsed >= cfg['duration']


# ─────────────────────────────────────────────────────────────────────────────
#  QUIT-BY-GESTURE
#  Hold a closed fist with BOTH hands for 3 seconds anywhere in the game to
#  quit. Shows a confirmation ring around the gesture indicator.
# ─────────────────────────────────────────────────────────────────────────────

QUIT_HOLD = 3.0  # Must hold for 3 seconds — well past any normal turn duration


class QuitDetector:
    """Quit gesture: both hands visible AND both showing closed fist for 3+ seconds.

    To prevent accidental quits during gameplay (where closed fists might
    register between turns), this only counts time when BOTH hands are
    continuously fists. Any interruption — even one frame of a different
    gesture — resets the timer to zero."""

    def __init__(self):
        self.start_time = None
        self.consecutive_fist_frames = 0

    def update(self, mpr):
        # Need at least two hands, both closed fists
        two_fists = False
        if mpr.hand_landmarks and len(mpr.hand_landmarks) >= 2:
            fists = sum(1 for lms in mpr.hand_landmarks
                        if fstates(lms) == (0, 0, 0, 0, 0))
            two_fists = fists >= 2

        if not two_fists:
            # Any non-fist frame: reset completely
            self.start_time = None
            self.consecutive_fist_frames = 0
            return False, 0.0

        # Require at least 5 consecutive frames of fists before starting the timer.
        # This filters out single-frame misdetections during normal play.
        self.consecutive_fist_frames += 1
        if self.consecutive_fist_frames < 5:
            return True, 0.0

        if self.start_time is None:
            self.start_time = time.time()
        elapsed = time.time() - self.start_time
        return True, min(1.0, elapsed / QUIT_HOLD)


def draw_quit_indicator(frame, prog, w, h):
    """Small indicator on the lower-right showing how close the quit gesture is."""
    cx, cy = w - 50, h - 70
    R = 22
    rounded_rect(frame, (cx - R - 6, cy - R - 6),
                 (cx + R + 6, cy + R + 6), (30, 12, 18), 10, -1)
    cv2.circle(frame, (cx, cy), R, (15, 8, 10), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), R, (60, 60, 80), 1, cv2.LINE_AA)
    if prog > 0:
        cv2.ellipse(frame, (cx, cy), (R - 3, R - 3),
                    -90, 0, int(360 * prog),
                    (60, 75, 235), 4, cv2.LINE_AA)
    text(frame, "QUIT", (cx, cy + 6), 0.42, (220, 220, 230),
         bold=True, center=True)
    text(frame, "2 fists", (cx, cy + R + 16), 0.38, TEXT_MUTE, center=True)




# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL PATH
# ═══════════════════════════════════════════════════════════════════════════════

def find_model():
    candidates = [
        os.environ.get("HAND_LANDMARKER_PATH", ""),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task"),
        os.path.abspath("hand_landmarker.task"),
        os.path.expanduser("~/hand_landmarker.task"),
        os.path.expanduser("~/Desktop/hand_landmarker.task"),
        os.path.expanduser("~/Desktop/OOEFOLDER/hand_landmarker.task"),
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    print("ERROR: hand_landmarker.task not found.")
    print("Place hand_landmarker.task next to this script,")
    print("or set HAND_LANDMARKER_PATH env var.")
    sys.exit(1)


def main():
    model_path = find_model()
    base = mp_python.BaseOptions(model_asset_path=model_path)
    det = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(base_options=base, num_hands=2))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cv2.namedWindow("Hand Cricket", cv2.WINDOW_NORMAL)

    bgm_start(volume=0.20)

    gc = GestureConfirm()
    quit_det = QuitDetector()
    g = new_game()
    brain = Brain('EASY')
    dlbl, dkey = ' EASY', 'EASY'
    frame_idx = 0
    PAUSE_SECONDS = 1.2

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        frame_idx += 1

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = (hsv[..., 1].astype(np.int32) * 80 // 100).clip(0, 255).astype(np.uint8)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mpi = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        mpr = det.detect(mpi)

        draw_hands(frame, mpr, w, h)

        state = g['state']
        if state == 'MODE_SELECT':
            valid = range(1, 4)
        elif state == 'DIFFICULTY_SELECT':
            valid = range(1, 4)
        elif state in ('TOSS_CHOICE', 'TOSS_NUMBER', 'BAT_BOWL'):
            valid = range(1, 3)
        elif state in ('COIN_ANIM', 'TOSS_RESULT', 'INNINGS_BREAK',
                       'GAME_RESULT', 'CELEBRATION'):
            valid = range(1, 11)
        else:
            valid = g['run_range'] or range(1, 7)

        raw = raw_gesture(mpr, valid)
        conf, prog = gc.update(raw)

        # Quit gesture (closed fist held for 1.5s) — works in any state
        quit_active, quit_prog = quit_det.update(mpr)
        if quit_prog >= 1.0:
            break

        if state not in ('MODE_SELECT', 'DIFFICULTY_SELECT'):
            topbar(frame, g, w, dlbl, dkey)
        gesture_indicator(frame, raw, prog, w)
        progress_bar(frame, prog, w, h)
        draw_quit_indicator(frame, quit_prog, w, h)

        if state == 'MODE_SELECT':
            screen_mode_select(frame, conf, w, h, frame_idx)
            if conf in (1, 2, 3):
                g['mode'] = ['1 to 6', '1 to 10', 'CRAZY'][conf - 1]
                g['run_range'] = [range(1, 7), range(1, 11), range(1, 11)][conf - 1]
                g['state'] = 'DIFFICULTY_SELECT'
                play('click'); gc.reset()

        elif state == 'DIFFICULTY_SELECT':
            screen_difficulty(frame, conf, w, h, frame_idx)
            if conf in (1, 2, 3):
                dmap = {1: (' EASY', 'EASY'), 2: ('MEDIUM', 'MEDIUM'), 3: ('  HARD', 'HARD')}
                dlbl, dkey = dmap[conf]
                brain = Brain(dkey)
                g['state'] = 'TOSS_CHOICE'
                play('click'); gc.reset()

        elif state == 'TOSS_CHOICE':
            screen_toss_choice(frame, w, h)
            if conf in (1, 2):
                g['toss_choice'] = "ODD" if conf == 1 else "EVEN"
                g['state'] = 'TOSS_NUMBER'
                play('click'); gc.reset()

        elif state == 'TOSS_NUMBER':
            screen_toss_number(frame, g, w, h)
            if conf in (1, 2):
                cn = random.randint(1, 2)
                tot = conf + cn
                rs = "EVEN" if tot % 2 == 0 else "ODD"
                won = rs == g['toss_choice']
                g.update({
                    'toss_user_num': conf, 'toss_comp_num': cn,
                    'toss_total': tot, 'toss_result_str': rs,
                    'user_won_toss': won, 'coin_timer': 0,
                })
                if not won:
                    dec = random.choice(['BAT', 'BOWL'])
                    g.update({
                        'toss_comp_dec': dec,
                        'user_batting': (dec == 'BOWL'),
                        'batting_first': 'user' if dec == 'BOWL' else 'computer',
                    })
                play('coin')
                g['state'] = 'COIN_ANIM'
                gc.reset()

        elif state == 'COIN_ANIM':
            g['coin_timer'] += 1
            draw_coin(frame, g['coin_timer'],
                      g['toss_user_num'], g['toss_comp_num'], w, h)
            if g['coin_timer'] >= COIN_DUR:
                g['state'] = 'TOSS_RESULT'
                if g['user_won_toss']:
                    play('toss_win')

        elif state == 'TOSS_RESULT':
            screen_toss_result(frame, g, conf, w, h)
            if g['user_won_toss']:
                if conf in (1, 2):
                    g['user_batting'] = (conf == 1)
                    g['batting_first'] = 'user' if conf == 1 else 'computer'
                    g['current_runs'] = 0
                    g['state'] = 'PLAYING'
                    play('click'); gc.reset()
            else:
                if conf is not None:
                    g['current_runs'] = 0
                    g['state'] = 'PLAYING'
                    play('click'); gc.reset()

        elif state == 'PLAYING':
            screen_play(frame, g, dlbl, dkey, w, h)
            if conf is not None:
                # The CPU must choose before it learns this ball's gesture,
                # otherwise the "prediction" would be reading the player's hand.
                comp = brain.pick(g['run_range'], not g['user_batting'])
                brain.record(conf)
                bv = conf if g['user_batting'] else comp
                bw = comp if g['user_batting'] else conf
                runs, out, msg = resolve(bv, bw, g['mode'])
                if not out:
                    g['current_runs'] += runs
                g.update({
                    'you_showed': conf, 'comp_showed': comp,
                    'ball_msg': msg, 'ball_out': out, 'ball_runs': runs,
                    'ball_timer': time.time(), 'state': 'BALL_RESULT',
                })

                is_six  = (g['mode'] == '1 to 6'  and bv == 6)
                is_ten  = (g['mode'] == '1 to 10' and bv == 10)
                is_sq   = (g['mode'] == 'CRAZY'   and not out and bv == bw)
                is_four = (g['mode'] == '1 to 6'  and bv == 4) or \
                          (g['mode'] == '1 to 10' and bv in (7, 8))

                if out:
                    if g['user_batting']:
                        # User got out — red flash + camera shake
                        fx.trigger('OUT')
                        g['pending_celebration'] = 'OUT_BATTING'
                    else:
                        # User took a wicket — green flash (positive)
                        fx.trigger('FLASH', col=(95, 220, 120))
                        g['pending_celebration'] = 'WICKET_BOWLING'
                elif is_six or is_ten or is_sq:
                    lbl = "SIX!" if is_six else ("TEN!" if is_ten else f"{bv}^2={runs}!")
                    if g['user_batting']:
                        fx.trigger('BIGRUN', lbl=lbl)
                        fx.trigger('RUN_POP', runs=runs, col=GOLD)
                        play('six')
                    else:
                        fx.trigger('FLASH', col=(60, 60, 200))
                        fx.trigger('RUN_POP', runs=runs, col=(80, 100, 240))
                        play('out')
                elif is_four:
                    if g['user_batting']:
                        fx.trigger('FLASH', col=(160, 230, 90))
                        fx.trigger('RUN_POP', runs=runs, col=PRIMARY)
                        play('four')
                    else:
                        fx.trigger('FLASH', col=(80, 80, 200))
                        fx.trigger('RUN_POP', runs=runs, col=(80, 100, 240))
                        play('run')
                else:
                    if g['user_batting']:
                        fx.trigger('FLASH', col=(95, 220, 120))
                        fx.trigger('RUN_POP', runs=runs, col=PRIMARY)
                        play('run')
                    else:
                        fx.trigger('FLASH', col=(70, 70, 130))
                        fx.trigger('RUN_POP', runs=runs, col=(120, 130, 220))
                        play('hover')

                # Milestone celebrations (batting only)
                if g['user_batting'] and not out:
                    milestone = check_milestone(g)
                    if milestone:
                        g['pending_celebration'] = milestone
                gc.reset()

        elif state == 'BALL_RESULT':
            screen_ball_result(frame, g, w, h, PAUSE_SECONDS)
            start = g.get('ball_timer') or time.time()
            elapsed = time.time() - start
            if elapsed > PAUSE_SECONDS:
                # dict of game updates for the post-celebration state
                next_state_update = end_of_ball_update(g)

                # If a celebration is pending, run it first; otherwise just transition.
                if g.get('pending_celebration'):
                    g['_post_celebration_update'] = next_state_update
                    g['celebration_start'] = time.time()
                    g['celebration_acknowledged'] = False
                    g.pop('_celebration_sounds_played', None)
                    g['state'] = 'CELEBRATION'
                    bgm_volume(0.06)  # duck music for crowd reactions
                else:
                    g.update(next_state_update)

        elif state == 'CELEBRATION':
            kind = g.get('pending_celebration')
            if kind is None:
                # safety: nothing pending, just resume
                upd = g.pop('_post_celebration_update', None)
                if upd:
                    g.update(upd)
                else:
                    g['state'] = 'PLAYING'
                bgm_volume(0.20)
            else:
                done = screen_celebration(frame, g, kind, w, h, mpr, frame_idx)
                if done:
                    g['pending_celebration'] = None
                    upd = g.pop('_post_celebration_update', None)
                    g['celebration_start'] = None
                    g['celebration_acknowledged'] = False
                    g.pop('_celebration_sounds_played', None)
                    # Stop any confetti/FX from the celebration
                    fx.confetti = None
                    bgm_volume(0.20)  # restore music
                    if upd:
                        g.update(upd)
                    else:
                        # came from GAME_RESULT path — just stay there
                        g['state'] = 'GAME_RESULT'
                    gc.reset()

        elif state == 'INNINGS_BREAK':
            screen_innings_break(frame, g, w, h)
            if conf is not None:
                g['state'] = 'PLAYING'
                play('click'); gc.reset()

        elif state == 'GAME_RESULT':
            screen_game_result(frame, g, dlbl, w, h)
            s1, s2 = g['innings_scores']
            user_s, cpu_s = (s1, s2) if g['batting_first'] == 'user' else (s2, s1)
            if not g.get('result_sound_played', False) and user_s is not None and cpu_s is not None:
                if user_s > cpu_s:
                    fx.trigger('WIN'); fx.confetti_on(w, h); play('win')
                elif cpu_s > user_s:
                    play('lose')
                else:
                    play('tie')
                g['result_sound_played'] = True
            # Match-end celebration (plays once)
            if not g.get('match_celebration_shown', False) and \
                    user_s is not None and cpu_s is not None and user_s != cpu_s:
                g['match_celebration_shown'] = True
                g['pending_celebration'] = 'MATCH_WIN' if user_s > cpu_s else 'MATCH_LOSE'
                g['celebration_start'] = time.time()
                g['celebration_acknowledged'] = False
                g.pop('_celebration_sounds_played', None)
                g['_post_celebration_update'] = {'state': 'GAME_RESULT'}
                g['state'] = 'CELEBRATION'
            if conf == 1:
                g = new_game(); fx.clear(); play('click'); gc.reset()
            elif conf == 2:
                break

        # A gesture held across a screen change must not carry over and
        # instantly confirm something on the next screen.
        if g['state'] != state:
            gc.reset()

        fx.draw(frame)
        cv2.imshow("Hand Cricket", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q') or key == ord('Q'):  # ESC or Q
            break

    cap.release()
    cv2.destroyAllWindows()
    det.close()
    bgm_stop()
    if SOUND_ON:
        pygame.mixer.quit()


if __name__ == "__main__":
    main()