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