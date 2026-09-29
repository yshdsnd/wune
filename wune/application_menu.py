"""Non-modal application menu drawn in the spectrum's existing event loop."""
import pygame as pg
from .i18n import Translator


class ApplicationMenu:
    def __init__(self):
        self.anchor = None
        self.selected = 0

    def close(self):
        self.anchor = None

    def items(self, language, fullscreen):
        t = Translator(language)
        return (("settings", t("menu.settings"), "F2"),
                ("fullscreen", t("menu.exit_fullscreen" if fullscreen else "menu.enter_fullscreen"), "Alt+Enter"),
                ("exit", t("menu.exit"), ""))

    @staticmethod
    def button_rect(font, language):
        return pg.Rect(24, 14, font.size(Translator(language)("menu.open"))[0] + 24, 30)

    def geometry(self, size, font, language, fullscreen):
        items = self.items(language, fullscreen)
        width = max(font.size(label)[0] + font.size(shortcut)[0] + 52 for _, label, shortcut in items)
        row_height = max(34, font.get_linesize() + 12)
        rect = pg.Rect(self.anchor or (0, 0), (width, row_height * len(items) + 8))
        rect.clamp_ip(pg.Rect((0, 0), size).inflate(-16, -16))
        rows = [pg.Rect(rect.x + 4, rect.y + 4 + i * row_height, width - 8, row_height)
                for i in range(len(items))]
        return rect, rows

    def handle(self, event, size, font, language, fullscreen):
        """Return (consumed, command); never run a nested event loop."""
        if event.type in (pg.WINDOWFOCUSLOST, pg.WINDOWMINIMIZED, pg.VIDEORESIZE, pg.WINDOWSIZECHANGED):
            self.close()
            return False, None
        button = self.button_rect(font, language)
        if event.type == pg.MOUSEBUTTONDOWN and event.button == 3:
            self.anchor, self.selected = event.pos, 0
            return True, None
        if event.type == pg.MOUSEBUTTONDOWN and event.button == 1 and button.collidepoint(event.pos):
            self.anchor = None if self.anchor is not None else button.bottomleft
            self.selected = 0
            return True, None
        if self.anchor is None:
            return False, None
        _, rows = self.geometry(size, font, language, fullscreen)
        if event.type == pg.MOUSEMOTION:
            self.selected = next((i for i, row in enumerate(rows) if row.collidepoint(event.pos)), -1)
            return True, None
        if event.type == pg.MOUSEBUTTONDOWN:
            index = next((i for i, row in enumerate(rows) if row.collidepoint(event.pos)), -1)
            self.close()
            command = self.items(language, fullscreen)[index][0] if event.button == 1 and index >= 0 else None
            return True, command
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                self.close()
                return True, None
            if event.key in (pg.K_UP, pg.K_DOWN):
                self.selected = (self.selected + (1 if event.key == pg.K_DOWN else -1)) % len(rows)
                return True, None
            if event.key in (pg.K_RETURN, pg.K_KP_ENTER) and not getattr(event, "mod", 0) & pg.KMOD_ALT:
                command = self.items(language, fullscreen)[self.selected][0] if self.selected >= 0 else None
                self.close()
                return True, command
            self.close()
        return False, None

    def draw(self, surface, font, language, fullscreen, theme):
        button = self.button_rect(font, language)
        pg.draw.rect(surface, theme.info_background, button, border_radius=6)
        pg.draw.rect(surface, theme.info_text, button, 1, border_radius=6)
        surface.blit(font.render(Translator(language)("menu.open"), True, theme.info_text),
                     (button.x + 12, button.centery - font.get_height() // 2))
        if self.anchor is None:
            return
        rect, rows = self.geometry(surface.get_size(), font, language, fullscreen)
        pg.draw.rect(surface, theme.info_background, rect, border_radius=6)
        pg.draw.rect(surface, theme.info_text, rect, 1, border_radius=6)
        for index, ((_, label, shortcut), row) in enumerate(zip(self.items(language, fullscreen), rows)):
            if index == self.selected:
                pg.draw.rect(surface, theme.badge_background, row, border_radius=4)
            y = row.centery - font.get_height() // 2
            surface.blit(font.render(label, True, theme.info_text), (row.x + 10, y))
            text = font.render(shortcut, True, theme.info_text)
            surface.blit(text, (row.right - text.get_width() - 10, y))
