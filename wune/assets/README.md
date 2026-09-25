# Wune icon

Based on the design sheet supplied by the repository owner for Issue #55
on 2026-09-25 (`codex-clipboard-7e28f5d0-57b9-40d2-b846-0280e1da8c82.png`).
The standalone artwork was adapted using Codex's built-in image generation tool.
No third-party icon library or stock image was added. This records provenance;
it does not assign a new license to the owner's supplied design.

- `wune.png`: 256 px artwork with the Wune name.
- `wune-window.png`: 64 px text-free artwork for pygame and non-Windows Tk.
- `Wune.ico`: 16, 24, 32, 48, 64, 96, 128 and 256 px RGBA frames.
  The 16–32 px frames use the text-free version; 48 px and larger include Wune.
  Windows Tk and the packaged executable use this ICO.

The design keeps the glossy dark rounded square, cyan rim and green/yellow/red
LED spectrum. The small variant simplifies the spectrum to seven columns.
Only downscaling and format encoding are performed by `tools/build_icons.py`.
To reproduce the ICO from checked-in PNGs (Pillow is only a development tool):

```powershell
python tools/build_icons.py wune/assets/wune.png wune/assets/wune-window.png
```

## Image generation prompts

Large artwork, using the user's sheet as reference:

> Create one production-ready standalone app icon based precisely on the large 256x256 icon at top left of supplied design sheet. Square canvas, transparent RGBA background outside rounded-square icon, tightly framed with only 2% transparent margin, no external glow outside silhouette. Preserve glossy black glass rounded square, thin polished silver/cyan rim, green-yellow-orange-red segmented LED spectrum, left tall peak and right smaller peak, pale cyan bold exact text 'Wune' beneath bars. Preserve design and proportions. No presentation sheet, no size labels, no other text, no mockup, no background. Single icon only. High-quality raster icon master for downsampling to 256px.

Small artwork, using the large artwork as reference:

> Create small-size version of this exact Wune app icon. Remove the Wune text entirely. Enlarge/reposition the equalizer to fill the central area, simplifying to 7 columns of sharply separated LED blocks so readable at 16/24/32 px, tall left peak, smaller right peak, green bottom yellow middle red top. Preserve glossy black rounded square and thin silver cyan rim; slightly simpler reflection for tiny icon. Single icon, square canvas, transparent RGBA outside rounded-square silhouette, no external glow or stray pixels. Tight frame 2% transparent margin. No text, no letters, no labels, no mockup, no surrounding background. Production icon master for downscaling.

Large artwork edge cleanup:

> Precise cleanup only. Preserve this Wune icon artwork, text, LED blocks, rounded-square rim, colors and layout exactly. The transparent area outside the outer rounded-square rim has stray white and cyan flecks especially across the top. Remove all stray pixels outside the icon silhouette, including outside glow. Exterior must be perfectly transparent alpha. Smooth clean antialiased rim silhouette. Do not alter or regenerate the interior design. Keep square canvas and centered icon, no other changes.
