from dataclasses import dataclass
from .colors import Theme

# ==========================
# 設定群（好みに合わせて調整）
# ==========================
@dataclass
class Config:
    language: str = "auto"  # auto or a catalog language code.
    theme: Theme = Theme()
    gauge_style: str = "flat"     # "flat" (original) or "box" (beveled LED)
    led_shape: str = "rounded"   # rectangle, rounded, ellipse
    led_aspect_ratio: float = 2.0  # width / height, independent of cell size
    initial_preset: str | None = "CLASSIC"  # Color theme only; None keeps custom colors.
    width: int = 1280
    height: int = 800
    fps: int = 60
    bars: int = 64                # バー本数
    leds_per_bar: int = 20        # 縦のLED個数
    led_gap: int = 1              # LEDの隙間(px)
    margin_lr: int = 40           # 左右マージン
    margin_tb: int = 16           # 上下の外側余白
    bar_gap: int = 6              # バー同士の隙間

    # インフォバー（入力仕様）
    info_enabled: bool = True
    info_height: int = 28
    info_position: str = "bottom"    # "top" or "bottom"

    # Time-based peak marker motion
    peak_hold_ms: float = 120
    peak_fall_per_second: float = 2.5  # full-scale spans per second

    # ほんのり残像（画面に黒を薄く重ねる）
    afterglow_alpha: int = 35     # 0で残像無し、値が大きいほど早く消える（0〜255）

    # 90s風ラベルなど
    show_badge: bool = True
    
    # 周波数スケール（表示用）
    show_freq_scale: bool = True
    min_freq_hz: float = 20.0
    max_freq_hz: float = 48000.0
    limit_to_20khz: bool = False  # Display/analysis range only; capture rate is unchanged.
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
    channel_layout: str = "vertical"  # "vertical" or "horizontal"
    spectrum_orientation: str = "frequency_horizontal"  # or "frequency_vertical"
    channel_gap: int = 24       # チャンネル間の余白(px)

    # 追加（目安値。画面の見え方に合わせて微調整OK）
    header_reserved: int = 36    # 上部のロゴ/バッジ/「dB」余白
    scale_reserved: int = 28     # 下部の周波数ラベル一式の高さ

    min_led_height: int = 3

    silence_rms_threshold: float = 5e-3     # 無音判定
    quiet_dbfs_floor: float = -55.0         # これ未満は“静寂”扱いにする

    # ビジュアルエンベロープの調整
    vis_attack_ms: int = 5
    vis_release_ms: int = 120

    # Windows render endpoint: None follows the default at startup.
    output_device: str | None = None
    sample_rate: int | None = None  # None: selected output's mix rate at startup.
    block_size: int = 4096
    output_floor: float = 0.05   # Display-only cutoff; envelope state is retained.

    def spectrum_upper_hz(self, samplerate: float, *, requested_max_hz=None) -> float:
        """Display policy, limited by the requested range and Nyquist safety."""
        policy_max = 20_000.0 if samplerate <= 48_000 else 40_000.0
        maximum = self.max_freq_hz if requested_max_hz is None else requested_max_hz
        cap = 20_000.0 if self.limit_to_20khz else policy_max
        return min(maximum, policy_max, cap, samplerate * 0.5 * 0.999)

CFG = Config()
