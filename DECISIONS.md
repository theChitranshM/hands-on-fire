# Decisions

Judgment calls the brief didn't cover. Simplest option that keeps the illusion convincing.

## Hardware / measured
- **Mac:** Apple M1 Pro, 16 GB, arm64 (checked with `uname -m`). Python 3.11 (`/usr/local/bin/python3.11`, universal build, runs arm64).
- **Projector native resolution (R2):** SGD SX8 (the Kimwood) gets 1920x1080 @ 60 Hz over HDMI, and 1px line gratings stay sharp on the wall (tested 2026-11-05), so the panel really is 1080p. Rendering is native 1920x1080 (`display.max_render_size`). Effects bench on the projector: 231 fps average with vsync off and ~45k particles, so 60 fps has wide headroom.
- **Throw distance that reads on camera (R1):** _pending: fill in after the dark-room test._
- **Tracking light (R3):** _pending: projected light only / warm lamp / IR. Fill in._
- **Measured latency (R4):** 2026-11-06, FaceTime camera at 29.5 fps into the SGD SX8: median 174 ms, p95 204 ms, min 137 ms (50/50 trials). **Above the 150 ms line.** `predict_max_ms` was raised from 140 to 230 so prediction can cover the full lag instead of being clipped. Re-test after turning off projector processing (00:16): median 172 ms, p95 222 ms. No change, so the SX8's lag is fixed in hardware. **Decision: accept it and choreograph slowly.** Move the hands smoothly with no sudden stops, give each gesture about 1 s, and make the launch a steady lift rather than a flick. Prediction covers the steady part of a movement; only sudden direction changes show the lag.
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

## First real run (2026-11-06): what the footage showed, and the fixes
- **Tracking:** the take footage showed hands only ~50–100 px wide in a dark frame, and 23 "tracking lost" clips in about 3 minutes. Replaying MediaPipe over that footage: baseline found a hand in 18% of frames. Brightening shadows first (gamma 0.55) raised that to 37%, and loosening the confidence thresholds to 0.3 raised it to **45%**. Cropping to the projector area and CLAHE didn't help. Shipped gamma 0.55 + 0.3 thresholds, and the README now says to stand 1–1.5 m from the camera, which is the bigger lever.
- **The footage logger burns the tracking skeleton into the camera half** of take and glitch clips, so a fail is self-explaining (and better B-roll).
- **Realism:**
  - **Diya:** real proportions (~5 cm flame, ~7 cm bowl on a ~9.5 cm palm). It rests in the palm hollow, with the wick lifted by half the bowl height, and stays level with gravity whatever the hand angle.
  - **Sparkler:** gripped at the bottom of a ~22 cm wire that points along the hand, tilted up. The burning point is at the far end and creeps toward the fingers over ~45 s. A short glowing just-burnt section sits under the ball, with a faint grey stick down to the grip. The core is a small white point with thin radiating streaks, not a blob.
  - **Cupped catch:** the same real-size diya settles into the cup.

## Second round (2026-11-06): one-hand show, steadier diya, more Diwali
- **Why the gestures failed:** replaying the classifier over take 3 showed two hands tracked together in under 3% of frames. The performer reaches one arm into the light. Fist, pinch-both, cupped, raise and palms-together never fired; only open palm and arms wide did. So the show is now **one-hand first**:
  - pinch, point, raise, low palm and circle all work with either hand.
  - **Namaste** also counts when two hands closing in merge into one detection, or vanish at their midpoint. Pressed-together palms usually do exactly that to MediaPipe.
- **Fist** is judged on mean finger curl, not "every finger curled", because MediaPipe often leaves one finger half-straight.
- **Fallback:** a beat whose gesture never comes saves a glitch clip at 8 s and moves on by itself at 14 s (`sequence.fallback_s`), logged as `timeout`. A take never stalls in front of the camera.
- **Trembling diya:** the 170 ms prediction was amplifying landmark noise. Three fixes:
  1. Prediction now uses a 3 Hz low-passed velocity.
  2. Prediction is speed-gated: none when the hand is still, full above 1.4 palm sizes per second.
  3. The diya anchor gets its own One Euro stage (min_cutoff 0.5, beta 3 per palm size).
  On take-3 footage, frame-to-frame shake went from 8.0 px to 2.4 px median and the still-hand p90 from 12.7 px to 7.3 px. A dead-zone "hold, then catch up" stabiliser was tried and measured worse (stair-stepping), so it was dropped.
- **New beats:**
  - Puja: deepavali row of lamps, rangoli, marigold and rose petals, shankh and bell.
  - Aarti: a light trail, with a bell on every loop.
  - Phuljhadi: catches flame from the diya.
  - Chakri on the fingertip.
  - Anar from the low palm.
  - Sky lanterns at the quiet end, which now fades the whole picture to black.
- **Placement:** the row of lamps spans the middle 40% of the wall so a 9:16 crop keeps it. Petals are a short, gentle shower, not a blizzard.
- **Take videos are written by wall-clock time:** frames dropped under load are filled with repeats. The first takes played back fast (143 s of video for a 184 s take).
- **The HUD shows a big performer cue** for the next gesture.
