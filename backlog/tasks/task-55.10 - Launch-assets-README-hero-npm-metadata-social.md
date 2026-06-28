---
id: TASK-55.10
title: 'Launch assets: README hero, npm metadata, og:image, race GIF, Show HN kit'
status: In Progress
assignee: []
created_date: '2026-06-26'
updated_date: '2026-06-27'
labels:
  - marketing
  - docs
dependencies:
  - task-55.9
parent_task_id: TASK-55
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

Convert the live demo into a coordinated launch.

- **README hero:** line 1 = the hook ("scikit-learn clustering, GPU-accelerated, 100% in your browser — no Python, no install") → animated race GIF → a big "Open the live demo (no install) →" link → a 30-second copy-paste quickstart. Move the feature checklist below the fold.
- **npm metadata:** update `description` and `keywords` to include `hdbscan`, `som`, `self-organizing-map`, `webgpu`, `gpu-acceleration` (currently the description lists only K-Means/Spectral/Agglomerative).
- **Social assets:** a 1200×630 og:image with baked-in wordmark + URL (self-attributing when reshared), and a ≤6s looping race MP4/GIF referenced by the social cards.
- **Launch kit:** a Show HN title ("Show HN: clustering-tfjs — scikit-learn clustering in your browser, WebGPU vs CPU live race"), Reddit (r/MachineLearning, r/javascript, r/dataisbeautiful) and Bluesky posts, and honest-methodology reply snippets ready for the threads. Seed the demo into Awesome-TensorFlow.js / awesome-webgpu lists.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->

- [x] #1 The README opens with the one-line hook, an animated race GIF, and a live-demo CTA above the install commands
- [x] #2 The npm package description and keywords include `hdbscan`, `som`, `self-organizing-map`, `webgpu`, `gpu-acceleration`
- [ ] #3 A 1200×630 og:image (self-attributing wordmark + URL) and a ≤6s looping race GIF/MP4 exist and are referenced by the social cards — **og:image done; real race clip deferred to a WebGPU-hardware capture (see High-level summary)**
- [x] #4 A launch kit with Show HN / Reddit / Bluesky copy and methodology-reply snippets is ready
- [x] #5 ESLint passes before commit

<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->

## High-level summary

The launch turns the live demo into a coordinated release. The **README** now opens like a landing page: the one-line hook, a linked animated race visual, a prominent "Open the live demo (no install) →" CTA, and a 30-second `npm install` + working `KMeans.fit_predict` quickstart — all above the install matrix, with the feature checklist moved below the fold. The quickstart mirrors the exact public API the live demo generates (`import { Clustering, KMeans }`, `new KMeans(...)`, `await fit_predict`), and the verified output `[0, 0, 1, 1, 0, 2]` is real, not illustrative.

**npm discoverability** is fixed at the source: the package description and keywords now name HDBSCAN, SOM / self-organizing-map, WebGPU, and GPU-acceleration — the capabilities the package already ships but never advertised.

A **self-attributing social card** ships as `site/public/og-image.png` (1200×630), rasterized from a committed SVG source via `site/scripts/render_og_image.sh` so it is reproducible, not a one-off binary. It bakes in the wordmark and the demo URL so it identifies the project wherever it is reshared, and the page's Open Graph + Twitter Card meta (complete set, absolute URLs, alt text on both surfaces) reference it.

The **launch kit** (`backlog/docs/launch-kit.md`) carries ready-to-post copy for Show HN, Reddit (r/MachineLearning, r/javascript, r/dataisbeautiful), and Bluesky, four honest-methodology reply snippets (fair-benchmark protocol, "your hardware", sklearn parity, small-`n` CPU win) that link the proof already on the page, and awesome-list seeding entries. Every snippet scopes the speedup to Spectral RBF affinity and never claims a generic "GPU is faster" number.

### Deferred: the real race clip (AC#3, hardware-gated)

The one outstanding deliverable is the **real ≤6s looping race GIF/MP4**. A genuine clip can only be captured on actual WebGPU hardware recording the live demo; synthesizing one and presenting it as a hardware capture would violate the demo's load-bearing honesty principle. Until it is recorded, the site keeps the illustrative schematic `site/public/race-reference.svg` and **retains** the "A recorded clip replaces this at launch" figcaption caveat. The launch kit documents the exact 7-step swap: record on a WebGPU browser, save as `site/public/race-reference.<gif|mp4>`, point `#race-reference-media` (or a `<video>`) and `reveal_reference()` in `site/src/race_ui.ts` at it, drop the caveat, and reference it from the social posts. This is the only reason the task is In Progress rather than Done.

## What changed

- **README hero** (`README.md`): hook → linked animated race visual → bold live-demo CTA → 30-second quickstart, above the install commands; Features relocated below the fold and added to the ToC. Hero/CTA use plain emphasis (not headings) so they stay out of the document outline. README images use absolute GitHub-Pages URLs because `package.json` `files` excludes `site/`, so relative paths would 404 on npm. The below-fold Node example was aligned to the top-level `new KMeans(...)` form so the page never shows two constructors for the same call. Fixed a pre-existing broken ToC anchor (`#backend-selection` → `#platform-detection--backend-selection`).
- **npm metadata** (`package.json`): rewritten description naming all five algorithms + WebGPU; keywords add `hdbscan`, `som`, `self-organizing-map`, `webgpu`, `gpu-acceleration`.
- **Social card** (`site/public/og-image.svg` → `og-image.png`, `site/scripts/render_og_image.sh`, `site/index.html`): 1200×630 designed card; full OG + Twitter meta with `og:image:alt` and `twitter:image:alt`.
- **Launch kit** (`backlog/docs/launch-kit.md`): channel copy, methodology replies, awesome-list seeding, and the deferred-clip recording/swap procedure.

Reviewed by a five-lens agent panel (correctness/API accuracy, honesty/claims integrity, README IA, AC completeness, metadata/a11y). The quickstart's API and output were confirmed by running the real code; the broken ToC anchor, the constructor inconsistency, an over-precise schematic alt text, and the missing `twitter:image:alt` were caught and fixed. Host-casing lowercasing was considered and declined to stay consistent with the rest of task-55.

<!-- SECTION:NOTES:END -->
