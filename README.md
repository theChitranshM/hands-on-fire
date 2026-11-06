# Hands on Fire 🪔

**Diwali, projected onto your hands.** Join your hands in namaste and a row of diyas lights up across the wall. Open your palm and a clay diya sits in it; circle it for the aarti. Pinch, and a phuljhadi catches from the diya. Point, and a chakri spins on your fingertip. Lower your palm for an anar fountain, lift it to launch a rocket, open your arms for the finale. At the end, sky lanterns rise and one small diya stays glowing in your palm.

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

## Light: keep the room dark

Don't turn the room light on: it washes the projection out on camera. The camera still needs to see your hands, so **the projector lights them itself**. A soft warm pool follows each hand, like diya light on skin. When no hand is found, a dim search light glows over the performance area so a hand entering it gets picked up. On real takes the camera found a hand in 5% of frames in the dark and 33–39% with the light on; the hand light gives the camera that light only where your hands are.

- **UP / DOWN** make the hand light brighter or dimmer live (shown on the operator window). Go brighter if tracking drops; dimmer if it shows too much on the reel.
- **L** turns it off, for example if you light your hands with a small lamp instead.
- If tracking still struggles, put a dim warm lamp low and behind the camera, aimed at your hands and not the wall.

## How to stand and hold things

The light is drawn the way the real objects sit, so move the way you'd hold the real thing. The show is built for **one hand** reaching into the light (only namaste and arms-wide need both). The laptop shows a big cue for the next move.

- **Stand close:** about **1–1.5 m** from the camera and projector. Your hands need to be big in the camera frame. From across the room the tracker keeps losing them, and the projection is dimmer too.
- **Puja:** join your hands in namaste in front of your chest. A row of diyas lights up across the wall one by one (*deepavali* means "row of lamps"), a rangoli blooms at your hands, marigold petals fall, and the conch and bell sound. Part your hands and the rangoli scatters.
- **Diya:** open your palm, tilted toward the projector at chest height like offering a diya. The clay diya rests in the hollow of your palm, its flame always pointing straight up. Move slowly; a real flame would blow out if you swung it.
- **Aarti:** circle the diya slowly. It leaves a golden light trail like a long-exposure photo, and the bell rings on every loop.
- **Snuff / relight:** close your hand slowly over the flame, then open it again.
- **Phuljhadi (sparkler):** pinch thumb and index as if holding the bottom end of the wire. The flame passes from the diya to the sparkler tip. The wire runs up from your fingers along where your hand points, and the burning point creeps toward your fingers. Draw slow arcs. A second hand that pinches lights its own.
- **Chakri:** point your index finger. A ground spinner whirls on your fingertip, its sparks shifting gold → green → red.
- **Anar:** lower your open palm. A flowerpot fountain erupts upward from it.
- **Rocket:** lift your hand up in one steady move. A rocket launches from your palm and bursts on the wall.
- **Finale:** open both arms wide, slowly.
- **Quiet end:** hold the diya still. Sky lanterns (akash kandil) rise from above your hand, everything else fades, one diya stays, and then everything fades to black.

This projector shows the image about 170 ms after the camera sees your hand. The software predicts where your hand is going and steadies the diya when your hand is still, so smooth movement keeps the light locked on.

## The show

| # | Beat | Gesture | Key |
|---|---|---|---|
| 1 | Puja: row of lamps, rangoli, petals | namaste (hands joined) | SPACE / 1 |
| 2 | Diya | open palm | SPACE / 2 |
| 3 | Aarti | circle the diya | SPACE / 3 |
| 4 | Snuff | close your hand | SPACE / 4 |
| 5 | Relight | open your palm again | SPACE / 5 |
| 6 | Phuljhadi | pinch thumb + index (either hand) | SPACE / 6 |
| 7 | Chakri | point your index finger | SPACE / 7 |
| 8 | Anar | lower your open palm | SPACE / 8 |
| 9 | Rocket | lift your hand fast | SPACE / 9 |
| 10 | Firework | automatic after the rocket | 0 |
| 11 | Finale | arms wide | SPACE |
| 12 | Quiet end: sky lanterns, one diya | hold still | SPACE |

If a gesture doesn't come, the beat saves a "missed" glitch clip after 8 s and moves on by itself after `sequence.fallback_s` (14 s), so a take never stalls. Set it to 0 to wait forever.

### Controls (either window)

| Key | Action |
|---|---|
| SPACE | next beat (from the end screen: start a new take) |
| B | previous beat |
| R | reset to the start |
| 1–9, 0 | jump to beat 1–9, 10 |
| G | gesture triggers on/off (off = keyboard only) |
| L | hand light on/off |
| UP / DOWN | hand light brighter / dimmer |
| H | show/hide the operator window (with a big cue for your next move) |
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
| **Tracking lost / jittery** | Press UP a few times to brighten the hand light. Stand 1–1.5 m from the camera. Still lost: a dim warm lamp low behind the camera. Jitter: lower `tracker.one_euro.min_cutoff`. Lag on fast moves: raise `tracker.one_euro.beta`. |
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
