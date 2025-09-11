# colors.py
from dataclasses import dataclass
from typing import Tuple

# LEDカラー定義（オン/オフ）
GREEN_ON  = ( 80, 255, 110)
YELLOW_ON = (255, 240,  90)
RED_ON    = (255,  80,  80)

GREEN_OFF  = ( 10,  28,  12)
YELLOW_OFF = ( 30,  28,  10)
RED_OFF    = ( 28,  10,  10)

BORDER_DARK = (12, 12, 12)
PANEL_BG    = (6, 6, 10)
GLASS_WHITE = (255, 255, 255)

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

    def choose_color(self, level_ratio: float):
        # レベル位置に応じてLEDカラー（オン/オフ）を返す
        if level_ratio >= self.th_red:
            return self.red_on, self.red_off
        elif level_ratio >= self.th_yellow:
            return self.yellow_on, self.yellow_off
        return self.green_on, self.green_off

    