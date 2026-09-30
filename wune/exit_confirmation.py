"""Keyboard-exit confirmation; keeps the existing capture/render loop running."""
import pygame as pg
from .i18n import Translator


class ExitConfirmation:
    def __init__(self):
        self.active = False
        self.dont_ask = False
        self.selected = 1  # Cancel is the safe default, including Enter.

    def open(self):
        self.active = True
        self.dont_ask = False
        self.selected = 1

    def geometry(self, size, font, language):
        t = Translator(language)
        width = max(300, font.size(t('exit.dont_ask'))[0] + 72,
                    font.size(t('exit.title'))[0] + 40)
        height = max(170, font.get_linesize() * 3 + 90)
        panel = pg.Rect(0, 0, width, height)
        panel.center = (size[0] // 2, size[1] // 2)
        checkbox = pg.Rect(panel.x + 20, panel.y + 55, width - 40, 32)
        cancel = pg.Rect(panel.x + 20, panel.bottom - 54, (width - 52) // 2, 34)
        confirm = cancel.copy()
        confirm.right = panel.right - 20
        return panel, (checkbox, cancel, confirm)

    def handle(self, event, size, font, language):
        """Return (confirmed, don't ask) only when the prompt is dismissed."""
        if event.type == pg.KEYDOWN:
            if getattr(event, 'repeat', False):
                return None
            if event.key == pg.K_ESCAPE:
                self.active = False
                return False, False
            if event.key == pg.K_TAB:
                self.selected = (self.selected + (-1 if getattr(event, 'mod', 0) & pg.KMOD_SHIFT else 1)) % 3
            elif event.key in (pg.K_LEFT, pg.K_UP):
                self.selected = (self.selected - 1) % 3
            elif event.key in (pg.K_RIGHT, pg.K_DOWN):
                self.selected = (self.selected + 1) % 3
            elif event.key in (pg.K_RETURN, pg.K_KP_ENTER, pg.K_SPACE):
                return self.activate(self.selected)
        elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            _, controls = self.geometry(size, font, language)
            for index, rect in enumerate(controls):
                if rect.collidepoint(event.pos):
                    self.selected = index
                    return self.activate(index)
        return None

    def activate(self, index):
        if index == 0:
            self.dont_ask = not self.dont_ask
            return None
        self.active = False
        return index == 2, self.dont_ask if index == 2 else False

    def draw(self, surface, font, language, theme):
        if not self.active:
            return
        t = Translator(language)
        panel, controls = self.geometry(surface.get_size(), font, language)
        pg.draw.rect(surface, theme.info_background, panel, border_radius=8)
        pg.draw.rect(surface, theme.info_text, panel, 2, border_radius=8)
        surface.blit(font.render(t('exit.title'), True, theme.info_text), (panel.x + 20, panel.y + 16))
        for index, rect in enumerate(controls):
            if index == self.selected:
                pg.draw.rect(surface, theme.badge_background, rect, border_radius=4)
            if index == 0:
                box = pg.Rect(rect.x + 5, rect.centery - 8, 16, 16)
                pg.draw.rect(surface, theme.info_text, box, 1)
                if self.dont_ask:
                    pg.draw.line(surface, theme.info_text, box.topleft, box.bottomright, 2)
                    pg.draw.line(surface, theme.info_text, box.topright, box.bottomleft, 2)
                label = font.render(t('exit.dont_ask'), True, theme.info_text)
                surface.blit(label, (box.right + 10, rect.centery - label.get_height() // 2))
            else:
                pg.draw.rect(surface, theme.info_text, rect, 1, border_radius=4)
                label = font.render(t('exit.cancel' if index == 1 else 'exit.confirm'), True, theme.info_text)
                surface.blit(label, label.get_rect(center=rect.center))
