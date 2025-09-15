from dataclasses import dataclass
from .colors import Theme

# ==========================
# 設定群（好みに合わせて調整）
# ==========================
@dataclass
class Config:
    theme: Theme = Theme()
    width: int = 1280
    height: int = 480
    fps: int = 60
    bars: int = 64                # バー本数
    leds_per_bar: int = 20        # 縦のLED個数
    led_gap: int = 1              # LEDの隙間(px)
    margin_lr: int = 40           # 左右マージン
    margin_tb: int = 96           # 上下マージン（インフォバー分を少し広めに）
    bar_gap: int = 6              # バー同士の隙間
    corner_radius: int = 4        # LED角の丸み

    # インフォバー（入力仕様）
    info_enabled: bool = True
    info_height: int = 28
    info_position: str = "bottom"    # "top" or "bottom"

    # 色分けしきい値（割合）
    th_yellow: float = 0.60
    th_red: float = 0.85

    # レベル挙動
    attack: float = 0.30          # 上昇スピード（0〜1）
    release: float = 0.08         # 下降スピード（0〜1）

    # ピークホールド
    peak_hold_frames: int = 24    # 頂点を保持するフレーム数
    peak_fall_per_frame: float = 0.9 / 32  # 1フレームごとの落下量（LED個数比率）

    # ほんのり残像（画面に黒を薄く重ねる）
    afterglow_alpha: int = 35     # 0で残像無し、値が大きいほど早く消える（0〜255）

    # 90s風ラベルなど
    show_badge: bool = True
    badge_text: str = "GROOVE"
    
    # 周波数スケール（表示用）
    show_freq_scale: bool = True
    min_freq_hz: float = 20.0
    max_freq_hz: float = 48000.0
    scale_ticks_hz: tuple = (31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 48000)
    show_freq_edge_labels: bool = True  # 左端/右端に最小・最大の周波数ラベルを描く

    # 縦軸（dB）ラベル
    show_db_scale: bool = True
    db_min: float = -66.0   # 下端
    db_max: float = 0.0     # 上端
    db_step: float = 10.0   # 間隔
    db_label_pad: int = 6   # ラベルの右端とバー領域の間隔(px)
    db_unit_offset: int = 10   # dB単位ラベルを上にずらす量

    # ---- ステレオ分割 ----
    channels: int = 2           # 1=mono, 2=stereo
    channel_gap: int = 60       # 上下の段の間隔(px)

    # 追加（目安値。画面の見え方に合わせて微調整OK）
    header_reserved: int = 36    # 上部のロゴ/バッジ/「dB」余白
    scale_reserved: int = 28     # 下部の周波数ラベル一式の高さ

    min_led_height: int = 3
    min_leds_per_bar: int = 12

    silence_rms_threshold: float = 5e-3     # 無音判定
    silence_decay: float = 0.90             # 無音時の減衰率
    agc_decay: float = 0.99                 # AGCの係数（大きい=ゆっくり）
    norm_lo_pct: float = 20.0               # 下側パーセンタイル(%)
    norm_hi_pct: float = 90.0               # 上側パーセンタイル(%)
    post_floor: float = 0.04                # 極小値カット閾値(0..1)
    quiet_dbfs_floor: float = -55.0         # これ未満は“静寂”扱いにする
    min_norm_span_db10: float = 0.6         # パーセンタイル正規化の最小スパン（log10(power)単位、0.6 ≒ 6 dB）

    #input_device = "ステレオ ミキサー (Realtek(R) Audio), Windows WASAPI"
    input_device = "CABLE Output (VB-Audio Virtual Cable), Windows WASAPI"

CFG = Config()
