"""Local image background rendering, independent of themes and window styles."""
import pygame as pg


class ImageBackground:
    def __init__(self):
        self.path = None
        self.source = None
        self.key = None
        self.scaled = None

    def draw(self, surface, cfg):
        surface.fill(cfg.theme.background)
        path = cfg.background_path if cfg.background_mode == "image" else ""
        if path != self.path:
            self.path, self.source, self.key, self.scaled = path, None, None, None
            if path:
                try:
                    loaded = pg.image.load(path)
                    # smoothscale needs 24/32-bit pixels; palette PNGs are common.
                    self.source = pg.Surface(loaded.get_size(), pg.SRCALPHA, 32)
                    self.source.blit(loaded, (0, 0))
                except (OSError, ValueError, pg.error):
                    pass  # Remember failure: no disk access or repeated errors per frame.
        if self.source is None:
            return
        size = surface.get_size()
        key = (size, cfg.background_sizing)
        if key != self.key:
            source = self.source
            w, h = source.get_size()
            width, height = size
            if cfg.background_sizing == "fill":
                # Crop first to avoid allocating an enormous intermediate surface.
                if w * height > h * width:
                    crop_w = max(1, round(h * width / height))
                    source = source.subsurface(((w - crop_w) // 2, 0, crop_w, h))
                else:
                    crop_h = max(1, round(w * height / width))
                    source = source.subsurface((0, (h - crop_h) // 2, w, crop_h))
                target = size
            else:
                ratio = min(width / w, height / h)
                target = (max(1, round(w * ratio)), max(1, round(h * ratio)))
            self.scaled = pg.transform.smoothscale(source, target)
            self.key = key
        surface.blit(self.scaled, self.scaled.get_rect(center=surface.get_rect().center))
