# Launch kit — clustering-tfjs interactive demo

Ready-to-post copy for the launch of the WebGPU-vs-CPU clustering demo. Lead every
channel with the race clip, link the live demo, and keep the honest-methodology
reply on hand for the first technical question.

**Canonical links**

- Live demo: <https://CRJFisher.github.io/clustering-tfjs/>
- GitHub: <https://github.com/CRJFisher/clustering-tfjs>
- npm: <https://www.npmjs.com/package/clustering-tfjs>

**The hook (use verbatim in the H1, og:title, and Show HN title):**
scikit-learn clustering, GPU-accelerated, 100% in your browser — no Python, no install.

---

## Show HN

**Title**

```
Show HN: clustering-tfjs — scikit-learn clustering in your browser, WebGPU vs CPU live race
```

**Body**

```
clustering-tfjs is a TypeScript port of scikit-learn's clustering algorithms
(K-Means, Spectral, Agglomerative, HDBSCAN, SOM) running on TensorFlow.js — no
Python, no install.

The demo does two things in your browser, on your hardware:

1. A live WebGPU-vs-CPU race on the same seeded dataset. The headline workload is
   Spectral RBF affinity construction (O(n²·d)), where the GPU win is real and
   large. Each backend runs in its own Web Worker (TF.js keeps one global backend
   per context, so a fair race needs one worker per backend). Timing brackets the
   full fit including the awaited GPU readback; first-run shader-compile cost is
   shown separately and never folded into the speedup number.

2. The scikit-learn toy-dataset grid (moons / circles / blobs / anisotropic /
   no-structure) recreated live across all five algorithms, with per-algorithm
   sliders, on datasets/params where float32 parity holds.

Honesty notes baked into the page: every lane is float32 on identical input, a
cross-backend equality check is shown, the small-n CPU win is a first-class part
of the crossover slider, and a permanent footer states the numbers come from your
own hardware. Non-WebGPU/mobile visitors get a recorded reference clip plus their
fastest available backend (WebGL → WASM → CPU).

Live demo: https://CRJFisher.github.io/clustering-tfjs/
Source: https://github.com/CRJFisher/clustering-tfjs
```

---

## Reddit

### r/MachineLearning

**Title**

```
[P] clustering-tfjs: scikit-learn clustering (K-Means/Spectral/Agglomerative/HDBSCAN/SOM) in the browser, with a live WebGPU-vs-CPU race
```

**Body**

```
I ported scikit-learn's clustering algorithms to TypeScript on TensorFlow.js and
built an interactive demo that runs entirely in the browser — no Python, no install.

Two things to look at:

- A live WebGPU-vs-CPU race on Spectral RBF affinity construction (O(n²·d)), the
  workload where the GPU win is real. Fair-race protocol: one Web Worker per
  backend, float32 on identical input, timing includes the awaited readback,
  first-run shader compile shown separately. A crossover n-slider shows CPU winning
  at small n and GPU pulling ahead past the marked crossover.
- The familiar toy-dataset grid across all five algorithms, with parity to sklearn
  on curated datasets/params (differences annotated, not hidden).

Methodology is documented in an always-visible expander on the page, and every
number is measured on the visitor's own hardware.

Demo: https://CRJFisher.github.io/clustering-tfjs/
Code: https://github.com/CRJFisher/clustering-tfjs
```

### r/javascript

**Title**

```
clustering-tfjs: 5 clustering algorithms in TypeScript on TF.js, with a WebGPU-vs-CPU race running in two Web Workers
```

**Body**

```
clustering-tfjs runs scikit-learn-compatible clustering (K-Means, Spectral,
Agglomerative, HDBSCAN, SOM) in the browser via TensorFlow.js. New in this release:
a WebGPU backend behind a per-backend ESM loader with navigator.gpu feature
detection and graceful WebGL → WASM → CPU fallback.

The demo races WebGPU against CPU on the same dataset. The interesting JS detail:
TF.js maintains a single global backend per JS context, so the two backends can't
run concurrently on the main thread — each lane gets its own Web Worker importing
tfjs-core plus exactly one backend package. Timing includes the awaited readback so
it's an honest wall-clock number, not a kernel-launch time.

npm install clustering-tfjs

Demo: https://CRJFisher.github.io/clustering-tfjs/
Code: https://github.com/CRJFisher/clustering-tfjs
```

### r/dataisbeautiful

**Title**

```
[OC] WebGPU vs CPU, racing the same clustering job live in the browser
```

**Body**

```
This is the same seeded dataset clustered on two backends side by side — WebGPU and
CPU — each in its own Web Worker, with live wall-clock timers and racing bars. The
workload is Spectral RBF affinity construction. Everything runs in the browser on
your own hardware; drag the n-slider to watch the crossover where the GPU pulls
ahead.

Built with clustering-tfjs (TypeScript + TensorFlow.js).
Interactive version: https://CRJFisher.github.io/clustering-tfjs/
```

(r/dataisbeautiful requires OC to be reproducible and to credit the tool/source —
both satisfied by the live demo link and the shareable permalink encoding the
dataset+params.)

---

## Bluesky

```
scikit-learn clustering, GPU-accelerated, 100% in your browser — no Python, no install.

Watch WebGPU race CPU on the same dataset, live on your hardware, then recreate the
sklearn toy-dataset grid across K-Means, Spectral, Agglomerative, HDBSCAN & SOM.

🔗 https://CRJFisher.github.io/clustering-tfjs/
```

Attach the ≤6s race clip (see "Race clip" below). Thread the methodology reply as
the first self-reply.

---

## Honest-methodology replies

Keep these ready for the first "are these numbers real?" question on any channel.
Each links the proof already on the page.

**"Is the GPU speedup cherry-picked / is this a fair benchmark?"**

```
Fair-race protocol is documented in the Methodology expander on the page: one Web
Worker per backend (TF.js keeps a single global backend per context), float32 on
identical input tensors, and timing brackets the full fit *including* the awaited
GPU readback. First-run shader-compile cost is shown separately and never folded
into the speedup multiplier. The headline workload is Spectral RBF affinity
(O(n²·d)) — I don't claim a generic "GPU is faster" number; K-Means actually only
nets ~1.2× because its Lloyd loop reads back every iteration.
```

**"Your hardware isn't my hardware."**

```
Exactly — that's why there's a permanent footer saying the numbers come from YOUR
hardware. The race runs live in your browser; nothing is pre-recorded except the
reference clip shown to non-WebGPU/mobile visitors. Run it and you'll get your own
multiplier.
```

**"Does it actually match scikit-learn?"**

```
On the curated datasets/params in the grid, yes — and where float32 causes drift
(e.g. HDBSCAN probabilities), it's annotated on the page rather than hidden. The
dataset + params are encoded in the URL, so you can copy the permalink, change the
inputs, and check your own cases.
```

**"Small n — isn't CPU faster there?"**

```
Yes, and that's shown as a first-class part of the crossover slider, not buried.
CPU wins below the marked crossover n; GPU pulls ahead above it. The affinity build
is O(n²·d), so the GPU advantage grows with n.
```

---

## Awesome-list seeding

Open PRs adding the demo to the curated lists where its audience already looks.
Lead each entry with the live-demo link and the one-line hook.

- **Awesome TensorFlow.js** (`tensorflow/tfjs` community lists / `aaronhma/awesome-tensorflow-js`) — under a clustering / ML entry.
- **awesome-webgpu** (`mikbry/awesome-webgpu`) — under demos / applications, framing the WebGPU-vs-CPU race.

Suggested entry text:

```
[clustering-tfjs](https://github.com/CRJFisher/clustering-tfjs) — scikit-learn
clustering (K-Means, Spectral, Agglomerative, HDBSCAN, SOM) in the browser on
TensorFlow.js, with a live WebGPU-vs-CPU race. [Demo](https://CRJFisher.github.io/clustering-tfjs/)
```

---

## Race clip (deferred — record on WebGPU hardware before posting)

AC#3 calls for a real ≤6s looping race clip. This must be captured on actual WebGPU
hardware so it represents a genuine run, not a synthetic animation. Until then the
site falls back to the illustrative schematic `site/public/race-reference.svg`.

**To record and ship the real clip:**

1. Open <https://CRJFisher.github.io/clustering-tfjs/> on a WebGPU-capable browser
   (Chrome/Edge 113+, Safari on macOS/iOS 26).
2. Set the crossover slider to a clear GPU-win n (e.g. 10,000) and run the race.
3. Screen-record the race panels for ≤6s as a loop (QuickTime/macOS screen capture,
   or `ffmpeg`/`gifski` to convert to GIF/MP4 ≤6s).
4. Save as `site/public/race-reference.gif` (or `.mp4`).
5. In `site/index.html`, point the `#race-reference-media` `<img src>` (or switch
   to `<video>`) at the new asset, and update `reveal_reference()` in
   `site/src/race_ui.ts` to load it.
6. Remove the "A recorded clip replaces this at launch" caveat from the figcaption.
7. Reference the same clip from the social posts above (Bluesky/Show HN/Reddit).

The og:image (`site/public/og-image.png`, regenerated from `og-image.svg` via
`site/scripts/render_og_image.sh`) is a designed marketing card and is already in.
