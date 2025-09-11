# colors.py
from typing import Tuple
from .config import CFG

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

def choose_color(level_ratio: float) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """レベル位置に応じてLEDカラー（オン/オフ）を返す"""
    if level_ratio >= CFG.th_red:
        return RED_ON, RED_OFF
    elif level_ratio >= CFG.th_yellow:
        return YELLOW_ON, YELLOW_OFF
    else:
        return GREEN_ON, GREEN_OFF
    