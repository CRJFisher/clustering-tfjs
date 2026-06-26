---
id: TASK-55.10
title: 'Launch assets: README hero, npm metadata, og:image, race GIF, Show HN kit'
status: To Do
assignee: []
created_date: '2026-06-26'
updated_date: '2026-06-26'
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

- [ ] #1 The README opens with the one-line hook, an animated race GIF, and a live-demo CTA above the install commands
- [ ] #2 The npm package description and keywords include `hdbscan`, `som`, `self-organizing-map`, `webgpu`, `gpu-acceleration`
- [ ] #3 A 1200×630 og:image (self-attributing wordmark + URL) and a ≤6s looping race GIF/MP4 exist and are referenced by the social cards
- [ ] #4 A launch kit with Show HN / Reddit / Bluesky copy and methodology-reply snippets is ready
- [ ] #5 ESLint passes before commit

<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->

Have the honest-methodology reply prepared before posting: link the live methodology expander, point to the "numbers from YOUR hardware" footer, and the "datasets/params are in the URL — try your own" permalink. Lead every channel with the race GIF.

The recorded race GIF/MP4 produced here also replaces the illustrative placeholder shipped in task-55.6. The non-WebGPU fallback `<figure>` (`#race-reference`) and its reveal logic already exist; swap `site/public/race-reference.svg` for the real ≤6s recording (and update the `<img>`/figcaption in `site/index.html` if moving to `<video>`), keeping the same reveal mechanism. Remove the "A recorded clip replaces this at launch" caveat from the figcaption once the real asset is in.

<!-- SECTION:NOTES:END -->
