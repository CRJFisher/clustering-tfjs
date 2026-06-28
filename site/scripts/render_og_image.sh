#!/usr/bin/env bash
# Rasterize the Open Graph social card from its SVG source.
# The PNG is committed (social scrapers can't render SVG og:images reliably);
# regenerate it whenever og-image.svg changes. Requires rsvg-convert (librsvg).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src="$here/public/og-image.svg"
out="$here/public/og-image.png"

rsvg-convert --width 1200 --height 630 --output "$out" "$src"
echo "wrote $out"
