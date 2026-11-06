# Decisions

Judgment calls the brief didn't cover. Simplest option that keeps the illusion convincing.

## Hardware / measured
- **Mac:** Apple M1 Pro, 16 GB, arm64 (checked with `uname -m`). Python 3.11 (`/usr/local/bin/python3.11`, universal build, runs arm64).
- **Projector native resolution (R2):** SGD SX8 (the Kimwood) gets 1920x1080 @ 60 Hz over HDMI, and 1px line gratings stay sharp on the wall (tested 2026-11-05), so the panel really is 1080p. Rendering is native 1920x1080 (`display.max_render_size`). Effects bench on the projector: 231 fps average with vsync off and ~45k particles, so 60 fps has wide headroom.
- **Throw distance that reads on camera (R1):** _pending: fill in after the dark-room test._
- **Tracking light (R3):** _pending: projected light only / warm lamp / IR. Fill in._
- **Measured latency (R4):** _pending: `tools/latency_test.py` writes `calibration/latency.json`._
- **Hand pose (R5):** _pending: palm vs back of hand, with dark sleeve._

## Architecture
- **Two GLFW windows, two GL contexts** (projector + HUD) on the main thread. The HUD is drawn with OpenCV into an image and uploaded as a texture. Mixing GLFW with `cv2.imshow` would put two Cocoa event loops in one process, so it's avoided. The HUD has no vsync and runs at 20 fps so it never halves the projector frame rate.
- **The projector window is true fullscreen**, not borderless. On macOS a borderless window can still get the menu bar on the second display. `AUTO_ICONIFY` is off, so clicking the HUD doesn't minimise the projector.
- **Render at ≤1280x720 into float16 layers, then upscale.** Keeps 60 fps comfortable, and the panel can't resolve more anyway.
- **Two layers.** *Direct* is cleared every frame and holds flames, diyas and rangoli, so the flame never smears behind a moving hand. *Trail* is persistent with decay (`render.trail_decay`) and holds sparks and embers, which is where the motion trails come from cheaply.
- **Hot values bleed to white** in the composite instead of clipping hue. Flame and spark cores read as white-hot on a dim projector.
- **Particles are simulated in numpy and rendered on the GPU** (one instanced draw). macOS caps OpenGL at 4.1, so there are no compute shaders. Vectorised numpy for ~60k particles costs about a millisecond, which is still not CPU blitting.
- **Prediction happens at render time, not in the tracker thread.** Lead = (now − capture time) + measured camera-to-photon latency + `predict_extra_ms` (half a vsync), capped at `predict_max_ms`. Velocity comes from the One Euro filter's derivative, damped by `predict_vel_damping` to avoid overshoot when the hand stops.
- **Handedness:** MediaPipe assumes a mirrored selfie image, so labels are swapped (`tracker.swap_handedness`). Each of two tracking slots keeps an EMA of its label, and two slots can never share a label. This stops the "sparkler jumps to the wrong hand" flicker.
- **Tracking loss:** hold the last pose for `loss_hold_ms` (no extrapolation), then fade `presence` to 0 over `loss_fade_ms`. Every effect multiplies by `presence`.
- **Homography = one plane.** Calibrate on a white board held where the hands will be. Hand-depth parallax is fixed with a live offset nudge (arrow keys in calibrate verify), saved to `calibration/homography.json`.

## Camera
- **Mac built-in FaceTime camera** (your call), not the iPhone. It's 30 fps instead of 60, so tracking runs about 33 ms behind instead of about 16 ms. The measured-latency prediction covers that gap. The laptop has to sit beside the projector lens so camera and projector see the hands from nearly the same point.

## Show control
- **`ESC` toggles blackout** (press again to come back). The brief only says "black out". `Q` quits, which the brief didn't specify.
- **No projector:** the show window and operator window tile side by side on the laptop instead of stacking. `H` hides or shows the operator window.
- **`C` toggles a calibration-check overlay inside the show** (white rings on the fingertips) instead of quitting to the tool. The full re-calibration is `tools/calibrate.py`.
- **Beat 7 is automatic** after `sequence.launch_flight_s`. Beat 10 ends itself after `quiet_hold_s + quiet_fade_s`.
- **Minimum dwell per beat.** A gesture can't fire the next beat until the current one has run at least 0.4 s. Beat 7 → 8 waits `firework_hold_s` and beat 9 → 10 waits `finale_min_s`, so the firework and finale always get to play.
- **Rangoli scatters when the palms come apart** (clap-and-open, or just separating). Detecting a true clap from 30–60 fps landmarks is unreliable; separation is.
- **"Open palm" ignores palm vs back of hand.** Facing detection from 2D landmarks is fragile, and which side you show is an R5 choice, not a gesture.
- **A missed gesture** (no trigger within `rehearse.beat_timeout_s`) saves a glitch clip in both modes. Only `--rehearse` auto-advances, logged as `missed`.
- **`--sim`** runs without a camera: fake hands follow the mouse and you drive the beats with SPACE or the number keys. It lets the visuals be built and tuned away from the dark room.

## Footage logger
- **Glitch clips** are camera on the left and projector output on the right, side by side, with take, beat and time burned in. That's what makes them usable as story B-roll.
- **The "5 best glitch clips"** are ranked by projector brightness × motion energy, preferring a variety of reasons. Brightest-and-busiest is the best proxy for "visually interesting fail".
- **"Time from first commit"** uses `git log`. With no commits yet, `story_material.md` says so and falls back to the first `build_log.md` entry, clearly labelled.

## Audio
- **sounddevice (PortAudio) instead of pygame.mixer.** OpenCV's wheel bundles its own SDL2. pygame loaded a second copy, and macOS warned about duplicate classes ("may cause mysterious crashes"). That risk isn't acceptable mid-take. Sounds are mixed in a small callback: fire-and-forget voices plus fading loops.
