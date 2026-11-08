"""Non-modal application menu drawn in the spectrum's existing event loop."""
import pygame as pg
import sys
from .i18n import Translator


class ApplicationMenu:
    def __init__(self):
        self.anchor = None
        self.selected = 0
        self.submenu = None
        self.user_window_presets = {}
        self.hovered = False
        self.borderless = False

    def close(self):
        self.anchor = None
        self.submenu = None

    def items(self, language, fullscreen, borderless=None):
        t = Translator(language)
        mac = sys.platform == "darwin"
        is_borderless = self.borderless if borderless is None else borderless
        if self.submenu == "window_size":
            from .window_presets import get_all_window_presets
            presets = get_all_window_presets(self.user_window_presets)
            items = [("back", "< " + t("menu.back"), "")]
            for p in presets:
                name = t(p.name_key) if p.name_key.startswith("presets.") else (p.description or f"{p.width} × {p.height}")
                items.append((f"window_preset:{p.id}", name, f"{p.width}×{p.height}"))
            return tuple(items)

        items = [
            ("settings", t("menu.settings"), "Cmd+," if mac else "F2"),
            ("fullscreen", t("menu.exit_fullscreen" if fullscreen else "menu.enter_fullscreen"), "Cmd+F" if mac else "Alt+Enter"),
        ]
        if not fullscreen:
            items.append(("borderless", t("menu.exit_borderless" if is_borderless else "menu.enter_borderless"), "F10"))
            items.append(("window_size", t("menu.window_size"), ">"))
        items.append(("exit", t("menu.exit"), "Q" if fullscreen else "Q / Esc"))
        return tuple(items)

    @staticmethod
    def button_rect(font, language=None):
        scale = max(1.0, font.get_height() / 18.0)
        height = max(28, font.get_height() + round(10 * scale))
        x = max(16, round(16 * scale))
        y = max(14, round(14 * scale))
        return pg.Rect(x, y, height, height)

    def geometry(self, size, font, language, fullscreen, borderless=None):
        items = self.items(language, fullscreen, borderless)
        scale = max(1.0, font.get_height() / 18.0)
        pad_w = max(52, round(52 * scale))
        pad_h = max(12, round(12 * scale))
        width = max(font.size(label)[0] + font.size(shortcut)[0] + pad_w for _, label, shortcut in items)
        row_height = max(34, font.get_linesize() + pad_h)
        rect = pg.Rect(self.anchor or (0, 0), (width, row_height * len(items) + round(8 * scale)))
        rect.clamp_ip(pg.Rect((0, 0), size).inflate(-16, -16))
        rows = [pg.Rect(rect.x + 4, rect.y + 4 + i * row_height, width - 8, row_height)
                for i in range(len(items))]
        return rect, rows

    def handle(self, event, size, font, language, fullscreen, borderless=None):
        """Return (consumed, command); never run a nested event loop."""
        if event.type in (pg.WINDOWFOCUSLOST, pg.WINDOWMINIMIZED, pg.VIDEORESIZE, pg.WINDOWSIZECHANGED):
            self.close()
            return False, None
        button = self.button_rect(font, language)
        if event.type == pg.MOUSEBUTTONDOWN and event.button == 3:
            self.anchor, self.selected = event.pos, 0
            self.submenu = None
            return True, None
        if event.type == pg.MOUSEBUTTONDOWN and event.button == 1 and button.collidepoint(event.pos):
            self.anchor = None if self.anchor is not None else button.bottomleft
            self.selected = 0
            self.submenu = None
            return True, None
        if self.anchor is None:
            return False, None
        _, rows = self.geometry(size, font, language, fullscreen, borderless)
        if event.type == pg.MOUSEMOTION:
            self.selected = next((i for i, row in enumerate(rows) if row.collidepoint(event.pos)), -1)
            return True, None
        if event.type == pg.MOUSEBUTTONDOWN:
            index = next((i for i, row in enumerate(rows) if row.collidepoint(event.pos)), -1)
            if event.button == 1 and index >= 0:
                command = self.items(language, fullscreen, borderless)[index][0]
                if command == "window_size":
                    self.submenu = "window_size"
                    self.selected = 0
                    return True, None
                if command == "back":
                    self.submenu = None
                    self.selected = 0
                    return True, None
                self.close()
                return True, command
            self.close()
            return True, None
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                if self.submenu is not None:
                    self.submenu = None
                    self.selected = 0
                    return True, None
                self.close()
                return True, None
            if event.key in (pg.K_UP, pg.K_DOWN):
                self.selected = (self.selected + (1 if event.key == pg.K_DOWN else -1)) % len(rows)
                return True, None
            if event.key in (pg.K_RETURN, pg.K_KP_ENTER) and not getattr(event, "mod", 0) & pg.KMOD_ALT:
                if self.selected >= 0:
                    command = self.items(language, fullscreen, borderless)[self.selected][0]
                    if command == "window_size":
                        self.submenu = "window_size"
                        self.selected = 0
                        return True, None
                    if command == "back":
                        self.submenu = None
                        self.selected = 0
                        return True, None
                    self.close()
                    return True, command
                self.close()
                return True, None
            self.close()
        return False, None

    def draw(self, surface, font, language, fullscreen, theme, hovered=False, borderless=None):
        button = self.button_rect(font, language)
        scale = max(1.0, font.get_height() / 18.0)
        radius = max(6, round(6 * scale))
        is_active = hovered or getattr(self, "hovered", False) or (self.anchor is not None)
        if is_active:
            pg.draw.rect(surface, theme.info_background, button, border_radius=radius)
            pg.draw.rect(surface, theme.info_text, button, max(1, round(1 * scale)), border_radius=radius)
            icon_color = theme.info_text
        else:
            dim_border = tuple(round(c * 0.40) for c in theme.info_border)
            dim_text = tuple(round(c * 0.55) for c in theme.info_text)
            pg.draw.rect(surface, theme.info_background, button, border_radius=radius)
            pg.draw.rect(surface, dim_border, button, max(1, round(1 * scale)), border_radius=radius)
            icon_color = dim_text
        # 3 horizontal bars (hamburger icon)
        bar_w = max(12, round(14 * scale))
        bar_h = max(2, round(2 * scale))
        gap = max(3, round(3.5 * scale))
        total_icon_h = 3 * bar_h + 2 * gap
        start_y = button.centery - total_icon_h // 2
        start_x = button.centerx - bar_w // 2
        for i in range(3):
            bar_y = start_y + i * (bar_h + gap)
            pg.draw.rect(surface, icon_color, (start_x, bar_y, bar_w, bar_h), border_radius=max(1, bar_h // 2))
        if self.anchor is None:
            return
        rect, rows = self.geometry(surface.get_size(), font, language, fullscreen, borderless)
        pg.draw.rect(surface, theme.info_background, rect, border_radius=radius)
        pg.draw.rect(surface, theme.info_text, rect, max(1, round(1 * scale)), border_radius=radius)
        for index, ((_, label, shortcut), row) in enumerate(zip(self.items(language, fullscreen, borderless), rows)):
            if index == self.selected:
                pg.draw.rect(surface, theme.badge_background, row, border_radius=max(4, round(4 * scale)))
            y = row.centery - font.get_height() // 2
            surface.blit(font.render(label, True, theme.info_text), (row.x + max(10, round(10 * scale)), y))
            text = font.render(shortcut, True, theme.info_text)
            surface.blit(text, (row.right - text.get_width() - max(10, round(10 * scale)), y))
