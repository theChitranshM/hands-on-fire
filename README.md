# Hands on Fire 🪔

**Diwali, projected onto your hands.** A diya flame sits on your palm. Close your fist and it goes out. Pinch, and sparklers burn from your fingers. Cup your hands and a rocket launches, and fireworks burst on the wall above you. At the end, one small diya stays glowing in your palm.

It all runs live, in one take, with no editing: a MacBook camera tracks your hands, the Mac renders the light, and a cheap projector paints it back onto you.

> Built by a homesick CS student in Vancouver who's from Ajmer, Rajasthan. Diwali back home is a whole street of light. Here it was me, a laptop and a $45 projector, so I wrote the light myself.

<!-- Reel: add the Instagram / YouTube link here -->

---

## What you need

| | |
|---|---|
| **Computer** | A Mac (Apple Silicon recommended; tested on an M1 Pro, macOS 27). macOS only. |
| **Projector** | Any HDMI projector. Tested on a ~$45 Amazon mini projector (Kimwood, shows up as "SGD SX8", true 1920x1080). |
| **Cable** | USB-C → HDMI adapter. |
| **Camera** | The Mac's built-in camera, placed right beside the projector lens. An iPhone via Continuity Camera also works. |
| **Room** | As dark as you can make it, with a plain wall for the fireworks. Wear a dark long-sleeve top. |
| **Software** | [Homebrew](https://brew.sh), then `brew install python@3.11 ffmpeg` |

## Install (5 minutes)

```bash
git clone https://github.com/theChitranshM/hands-on-fire.git
cd hands-on-fire
./setup.sh
```

`setup.sh` creates a virtual environment, installs pinned dependencies, downloads the MediaPipe hand model, generates the sound effects and lists your cameras and displays.

### Try it right now, no hardware needed
```bash
.venv/bin/python run.py --sim
```
Fake hands follow your mouse. Press **SPACE** to step through the whole show, **Q** to quit.

## ⚠️ Camera permission: read this first

macOS only gives camera access to apps that ask for it. **Run the camera tools from the regular Terminal app**, not from inside VS Code, Cursor or an AI coding app. Those often can't show the permission prompt, so they're silently blocked, and they won't appear in the camera settings list.

The easy way: **double-click the files in `launch/`**. Each one opens Terminal and runs one step. The first time, macOS asks *"Terminal would like to access the camera"*. Click **Allow**.

| Double-click | What it does |
|---|---|
| `launch/setup.command` | one-time setup (same as `./setup.sh`) |
| `launch/calibrate.command` | line the camera up with the projector (about 2 min) |
| `launch/latency.command` | measure the lag so effects land where your hand *will* be |
| `launch/brightness.command` | projector test patterns |
| `launch/gestures.command` | see what the tracker sees, with live gesture confidence |
| `launch/show.command` | the show |
| `launch/rehearse.command` | loop the show and log missed gestures |

If macOS says the file "can't be opened because it's from an unidentified developer": right-click it → Open → Open.

## Set up the room (every session)

1. **Projector:** plug in HDMI. In System Settings → Displays set it to **Extend**, not Mirror. In the projector's menu turn **keystone OFF** and any picture "enhancement" off, and turn Game/low-latency mode **on** if it has one. Point it at you and the wall behind or above you.
2. **Camera:** put the MacBook (or iPhone) so the camera sits **right beside the projector lens**, at the same height, facing you. The closer the camera is to the lens, the better the light lines up with your hands.
3. **Calibrate:** `launch/calibrate.command`. Hold a matte white sheet of paper or cardboard where your hands will be. The tool projects dots, finds them and saves the mapping. Then a dot follows your index fingertip. Use the **arrow keys** to nudge it onto the finger, then **ENTER** to save. Re-run any time the camera or projector moves.
4. **Latency:** `launch/latency.command`. It flashes the projector 50 times and times how long the camera takes to see it. Prediction uses this number.
5. **Show:** `launch/show.command`.

## The show

About 15 seconds, all driven by gestures. **Every beat also has a keyboard key**, so a missed gesture never ruins a take.

| # | Beat | Gesture | Key |
|---|---|---|---|
| 1 | Palm diya | open palm | SPACE / 1 |
| 2 | Snuff | close your fist | SPACE / 2 |
| 3 | Relight | open your palm again | SPACE / 3 |
| 4 | Sparklers | pinch (thumb + index) with both hands | SPACE / 4 |
| 5 | Diya catch | cup both hands together | SPACE / 5 |
| 6 | Launch | raise the cupped hands quickly | SPACE / 6 |
| 7 | Firework | automatic after the launch | 7 |
| 8 | Rangoli | palms together in front of your chest, then open them | SPACE / 8 |
| 9 | Finale | arms wide | SPACE / 9 |
| 10 | Quiet end | hold still for about 1 second | SPACE / 0 |

### Controls (either window)

| Key | Action |
|---|---|
| SPACE | next beat (from the end screen: start a new take) |
| B | previous beat |
| R | reset to the start |
| 1–9, 0 | jump to beat 1–9, 10 |
| G | gesture triggers on/off (off = keyboard only) |
| H | show/hide the operator window |
| M | mute |
| D | debug overlay: calibrated area, 9:16 safe zone, fingertip dots |
| C | calibration check: white rings on your fingertips |
| K | log a coffee ☕ |
| ESC | blackout (press again to come back) |
| Q | quit |

### Run options

```bash
.venv/bin/python run.py              # the show
.venv/bin/python run.py --rehearse   # loop the show, auto-restart, summary after every run
.venv/bin/python run.py --sim        # no camera: mouse-driven fake hands
.venv/bin/python run.py --mute       # no sound
.venv/bin/python run.py --hud        # force the operator window on
```

With a projector connected, the show goes fullscreen on it and the operator window (camera feed, current beat, gesture bars, fps, lag) stays on the laptop. Without one, both windows tile side by side on the laptop screen.

## Make it look good on camera

- **Projector closer = brighter.** Budget projectors are dim. A smaller, closer image reads far better on camera than a big, washed-out one. `launch/brightness.command` has test patterns for this.
- **Filming camera** goes behind and slightly to the side of the projector, so it sees the light on your hands without looking into the lens. Lock exposure and focus on your palm **with the flame on**. Shoot 60 fps if you can, so sparks don't smear.
- **Vertical reel?** Press **D**: the teal box is the 9:16 safe zone. Frame your filming camera to it.
- **Palm vs back of hand:** try both with a dark sleeve and use whichever reads brighter.
- Rehearse with `--rehearse` until five runs in a row go clean, then film at least five takes.

## Footage logger (built-in "making of")

Every run is recorded into `captures/`, so the story of the build makes itself:

- `takes/`: every take, with the camera feed and projector output side by side.
- `glitches/`: the previous 10 seconds, auto-saved whenever a gesture is missed, tracking drops or fps dips. Great blooper material.
- `takes.json`, `build_log.md`, `coffee.log`: take history, first-time-it-worked moments, coffee count (press K).
- `story_material.md`: written when you quit. Total takes, time from first commit to first clean take, coffees, milestones, and the 5 best glitch clips. Every number comes from the real logs; nothing is invented.

`captures/` and `calibration/` are git-ignored: your footage and your room's calibration stay on your machine.

## Tuning

Everything lives in **`config.yaml`**, and there are no magic numbers in the code:
- `render.brightness`, `render.scale`: global brightness and effect size.
- `render.flame.*`, `render.sparkler.*`, `render.fireworks.*`, `render.rangoli.*`, `render.bloom.*`: the looks.
- `gestures.*`: thresholds and hold times (watch them live in `launch/gestures.command`).
- `tracker.*`: smoothing and prediction.
- `sequence.*`: timings; `rangoli_enabled: false` cuts beat 8.
- `camera.device`: `"FaceTime"` (built-in), `"iPhone"`, or a number from `tools/list_devices.py`.

`.venv/bin/python tools/effects_preview.py` shows every effect without a camera (`--bench` measures fps).

## Troubleshooting

| Problem | Fix |
|---|---|
| **Camera permission denied / camera not in the settings list** | Run from the Terminal app (double-click `launch/*.command`) and click Allow. Already denied? System Settings → Privacy & Security → Camera → enable Terminal, then quit and reopen Terminal. |
| **No image on the projector** | Check the HDMI adapter and the projector's source. `.venv/bin/python tools/list_devices.py` should list two monitors. |
| **Show opens on the wrong screen** | Displays → **Extend**, not Mirror. Or set `display.projector_monitor` to the index/name from `list_devices.py`. |
| **Show opens in a small window** | No projector was detected (see above). |
| **Operator window missing** | Press H. |
| **Wrong camera** | `tools/list_devices.py`, then set `camera.device`. |
| **Tracking lost / jittery** | Open `launch/gestures.command`. Too dark: add a dim warm lamp outside the shot. Jitter: lower `tracker.one_euro.min_cutoff`. Lag on fast moves: raise `tracker.one_euro.beta`. |
| **Left and right hands swapped** | Flip `tracker.swap_handedness`. |
| **Flame sits next to the palm** | Re-run calibrate, or nudge with the arrows in its verify step. Press C during the show to check. |
| **Lag too high** | Re-run latency. Projector processing off, Game mode on. Move smoothly rather than snappily. |
| **Effects too dim on camera** | Projector closer, raise `render.brightness`, lock the filming camera's exposure on the flame. |
| **A gesture fires too early or too late** | Watch the bars in `launch/gestures.command`, then adjust `gestures.*`. G turns gestures off; SPACE always works. |
| **"Unidentified developer" on a .command file** | Right-click → Open → Open. |

## How it works

```
camera thread ─▶ MediaPipe Hands ─▶ One Euro smoothing ─▶ left/right slots
                                                             │
render loop (60 fps) ◀── predict ahead by measured lag ◀─────┘
   │  camera→projector homography (from calibration)
   │  gestures (hysteresis + hold) ─▶ beat state machine (1–10)
   ▼
GPU (moderngl): flame shader · particles · fireworks · rangoli · bloom
   ▼
projector, fullscreen, pure black everywhere else (black = no light)
```

The illusion is mostly two things: a **pure black** background, so only the effect emits light, and **prediction**. The camera, tracker and projector add roughly 100 ms of lag, so every effect is drawn where your hand is *going to be* when the light lands. `DECISIONS.md` explains every design call.

```
run.py                show entry point
config.yaml           every tunable
launch/               double-click launchers (Terminal)
hof/                  camera, tracker, gestures, sequence, mapping, audio, recorder, hud
hof/render/           engine (windows, layers, bloom), flame, particles + sparkler, fireworks, rangoli
tools/                calibrate, latency_test, brightness_test, gesture_debug, list_devices, effects_preview, gen_sounds
```

## Credits

- Hand tracking: [MediaPipe Hands](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) (Google, Apache 2.0). The model is downloaded by `setup.sh`.
- Rendering: [moderngl](https://github.com/moderngl/moderngl) + [GLFW](https://www.glfw.org/).
- All sounds are synthesized from scratch by `tools/gen_sounds.py`, so there are no samples or licensing issues.

MIT licensed. Make your own festival of lights. Happy Diwali 🪔
