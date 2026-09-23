# colors.py
from dataclasses import dataclass
from typing import Tuple

# LEDカラー定義（オン/オフ）
GREEN_ON  = ( 80, 255, 110)
YELLOW_ON = (255, 240,  90)
RED_ON    = (255,  80,  80)

# Preserve the previously rendered inactive color (formerly hard-coded).
GREEN_OFF = YELLOW_OFF = RED_OFF = (24, 24, 28)

BORDER_DARK = (12, 12, 12)
PANEL_BG    = (6, 6, 10)

@dataclass(frozen=True)
class Theme:
    th_yellow: float = 0.6
    th_red: float = 0.8
    green_on: Tuple[int,int,int]  = GREEN_ON
    green_off: Tuple[int,int,int] = GREEN_OFF
    yellow_on: Tuple[int,int,int] = YELLOW_ON
    yellow_off: Tuple[int,int,int]= YELLOW_OFF
    red_on: Tuple[int,int,int]    = RED_ON
    red_off: Tuple[int,int,int]   = RED_OFF
    background: Tuple[int,int,int] = PANEL_BG
    border: Tuple[int,int,int] = BORDER_DARK
    led_border: Tuple[int,int,int] = (18, 18, 20)
    led_outline: Tuple[int,int,int] = (0, 0, 0)
    highlight: Tuple[int,int,int] = (255, 255, 255)
    shadow: Tuple[int,int,int] = (0, 0, 0)
    peak: Tuple[int,int,int] = (255, 255, 255)
    peak_cutout: Tuple[int,int,int] = (8, 8, 10)
    scale_line: Tuple[int,int,int] = (70, 90, 100)
    scale_text: Tuple[int,int,int] = (150, 180, 190)
    edge_text: Tuple[int,int,int] = (180, 200, 210)
    db_text: Tuple[int,int,int] = (160, 180, 190)
    unit_text: Tuple[int,int,int] = (190, 210, 210)
    logo_text: Tuple[int,int,int] = (120, 120, 120)
    badge_text: Tuple[int,int,int] = (14, 230, 180)
    badge_glow: Tuple[int,int,int] = (10, 40, 36)
    badge_background: Tuple[int,int,int] = (30, 90, 80)
    info_background: Tuple[int,int,int] = (12, 22, 26)
    info_border: Tuple[int,int,int] = (20, 40, 44)
    info_text: Tuple[int,int,int] = (190, 220, 220)
    pause_text: Tuple[int,int,int] = (255, 255, 255)
    overlay: Tuple[int,int,int] = (0, 0, 0)

    def choose_color(self, level_ratio: float):
        # レベル位置に応じてLEDカラー（オン/オフ）を返す
        if level_ratio >= self.th_red:
            return self.red_on, self.red_off
        elif level_ratio >= self.th_yellow:
            return self.yellow_on, self.yellow_off
        return self.green_on, self.green_off
