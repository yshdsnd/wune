# wune/spectrum_audio.py
from __future__ import annotations
from dataclasses import replace

import numpy as np
from .capture import CaptureBackend, create_capture_backend
from .config import Config

from .ballistics import LevelEnvelope

class AudioSpectrum:
    """PCM capture → Hann/rFFT → calibrated band power → display levels."""
    def __init__(
        self,
        cfg: Config,
        bars: int,
        channels: int = 2,
        samplerate: int | None = None,
        blocksize: int | None = None,
        capture_backend: CaptureBackend | None = None,
    ):
        self.cfg = cfg
        self.bars = int(bars)
        self.channels_req = int(channels)
        self.nfft = int(cfg.block_size if blocksize is None else blocksize)
        if self.nfft < 3 or self.bars < 1 or self.channels_req < 1:
            raise ValueError("FFT size, bars and channels must be positive (FFT size >= 3)")
        if cfg.db_max <= cfg.db_min:
            raise ValueError("db_max must be greater than db_min")
        capture_cfg = cfg if samplerate is None else replace(cfg, sample_rate=samplerate)
        self.capture = (capture_backend if capture_backend is not None
                        else create_capture_backend(capture_cfg, blocksize=self.nfft))
        self._closed = False
        try:
            self.sr = int(self.capture.sample_rate)
            if self.sr <= 0:
                raise ValueError("Sample rate must be positive.")
            self.channels_eff = int(self.capture.channels)
            if self.channels_eff < 1:
                raise ValueError("Capture channels must be positive.")
            self.device = self.capture.device_name
            self._initialize_analysis()
        except BaseException:
            self.close()
            raise

    def _initialize_analysis(self):
        cfg = self.cfg
        self.last_rms = 0.0
        self.gated = False

        # FFT 前処理
        self.window = np.hanning(self.nfft).astype(np.float32)
        self.freqs = np.fft.rfftfreq(self.nfft, d=1 / self.sr)

        # Parseval: positive-frequency Hann power of a unit-peak sine is
        # N * sum(window**2) / 4. Calibrate band sums to that reference.
        self._power_scale = 4.0 / (self.nfft * np.sum(self.window.astype(np.float64)**2))
        if cfg.db_max <= cfg.db_min:
            raise ValueError("db_max must be greater than db_min")

        # 初期の周波数レンジ（Configから上書き可）
        self.fmin = 20.0
        self.fmax = float(self.sr // 2)
        self._rebuild_bins()

        # 出力スムージングの状態
        self._channel_mode = self.cfg.channel_mode
        self._out = np.zeros((1 if self._channel_mode == "stereo_mix" else self.channels_req, self.bars), dtype=np.float32)

        # ビジュアルエンベロープの作成
        self._vis_env = LevelEnvelope(attack_ms=self.cfg.vis_attack_ms, release_ms=self.cfg.vis_release_ms)

    # --- public API ----------------------------------------------------------
    def set_display_mode(self, mode: str) -> None:
        """Reset display history only; keep the capture device and FFT mapping."""
        if mode not in ("stereo", "stereo_mix"):
            raise ValueError("Invalid channel display mode")
        if mode != self._channel_mode:
            self._channel_mode = mode
            count = 1 if mode == "stereo_mix" else self.channels_req
            self._out = np.zeros((count, self.bars), dtype=np.float32)
            self._vis_env.y = None

    def set_range(self, fmin: float, fmax: float) -> None:
        """外側（Configなど）から周波数レンジを合わせる用。"""
        fmin = max(1.0, float(fmin))
        nyq = self.sr * 0.5
        fmax = max(fmin * 1.01, float(fmax))
        fmax = min(fmax, nyq - 1.0)   # ここでクランプ
        changed = (self.fmin, self.fmax) != (fmin, fmax)
        self.fmin, self.fmax = fmin, fmax
        self._rebuild_bins()
        if changed and hasattr(self, "_vis_env"):
            # Historical levels belong to the previous band allocation.
            self._vis_env.y = None
            self._out.fill(0)

    def step(self, dt: float) -> np.ndarray:
        """
        1フレームぶん処理して (channels,bars) の 0..1 を返す。
        固定の正弦波フルスケール基準で表示レベルに変換する。
        """
        data = self.capture.record(self.nfft)     # shape: (nfft, C)
        if data.ndim == 1:
            data = data[:, None]

        # サイレンス判定（しきい値は環境で微調整）
        frame_rms = float(np.sqrt(np.mean(np.square(data.astype(np.float32)))))
        self.last_rms = frame_rms

        # dBFSでの静寂ガード（ほぼ無音なら正規化をスキップ）
        EPS = 1e-12
        frame_db = 20.0 * np.log10(max(frame_rms, EPS))
        QUIET_DB = self.cfg.quiet_dbfs_floor  # 設定から読み取れるように

        self.gated = frame_db < QUIET_DB or frame_rms < self.cfg.silence_rms_threshold
        if self.gated:
            return self._animate(np.zeros_like(self._out), dt)

        C_in = data.shape[1]
        input_channels = 2 if self._channel_mode == "stereo_mix" else self.channels_req

        # 入力chが要求と違う場合に対処（足りなければ繰り返し、余れば先頭だけ）
        if C_in < input_channels:
            data = np.repeat(data, repeats=input_channels, axis=1)[:, :input_channels]
        elif C_in > input_channels:
            data = data[:, :input_channels]

        norm = self._map_levels(data)

        return self._animate(norm, dt)

    def _animate(self, target, dt):
        self._vis_env.configure(self.cfg.vis_attack_ms, self.cfg.vis_release_ms)
        # Apply the display floor to a copy, never to the envelope state.
        self._out = self._vis_env.step(target, dt).copy()
        self._out[self._out < self.cfg.output_floor] = 0.0
        return self._out

    def _map_levels(self, data: np.ndarray) -> np.ndarray:
        """Band energy in sine-peak-equivalent dBFS, without temporal processing.

        Sum linear power before taking the log. No per-frame or historical
        gain reference: scaling input by A shifts every band by 20*log10(A).
        A tone split across band boundaries shares its energy between them.
        """
        count = 2 if self._channel_mode == "stereo_mix" else self.channels_req
        powers = []
        for ch in range(count):
            spec = np.fft.rfft(self.window * data[:, ch])
            power = np.abs(spec)**2 * self._power_scale
            # DC is excluded by the band mapping; Nyquist has no negative twin.
            if self.nfft % 2 == 0:
                power[-1] *= 0.5
            powers.append(power)
        if self._channel_mode == "stereo_mix":
            # Equal-power mean preserves identical/opposite-phase stereo;
            # a single active side measures 10*log10(1/2) = -3.0103 dB.
            powers = [(powers[0] + powers[1]) * 0.5]
        out = np.zeros((len(powers), self.bars), dtype=np.float32)
        for ch, power in enumerate(powers):
            for band, indices in enumerate(self._bin_idx):
                energy = float(np.sum(power[indices]))
                db = 10.0 * np.log10(max(energy, 1e-20))
                out[ch, band] = np.clip(
                    (db - self.cfg.db_min) / (self.cfg.db_max - self.cfg.db_min), 0.0, 1.0
                )
        return out

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.capture.close()

    def _rebuild_bins(self) -> None:
        """ログ等間隔のバー境界を作り、rFFT周波数→バー対応を前計算。
        狭い範囲に多数のバーを割り当てても表示上限を越えない。
        """
        freqs = self.freqs                        # len = nfft//2+1, 0..Nyquist
        nyq = self.sr * 0.5

        fmin = max(1.0, float(self.fmin))
        fmax = min(float(self.fmax), nyq * 0.999)  # Nyquist手前に倒す（安全側）

        # ログ等分の境界（bars → bars+1 個）
        edges = np.geomspace(fmin, fmax, self.bars + 1)

        # 各境界を「左側に最も近いビン」へ（整数化）
        edge_bins = np.searchsorted(freqs, edges, side="left")
        upper_bin = min(len(freqs) - 1, int(np.searchsorted(freqs, fmax, side="left")))
        edge_bins = np.clip(edge_bins, 1, upper_bin)  # Exclude DC and bins outside the requested range.

        # Prefer at least one bin, but keep exhausted bands empty at the upper limit.
        for i in range(1, len(edge_bins)):
            if edge_bins[i] <= edge_bins[i-1]:
                edge_bins[i] = min(edge_bins[i-1] + 1, upper_bin)

        starts = edge_bins[:-1]
        # Preserve the existing mapping: zip uses the first bars stops.
        # The appended endpoint is currently unused; changing it would retune bands.
        stops  = np.append(edge_bins[1:], len(freqs))

        # 各バーのビン配列（hi は排他）
        self._bin_idx = [np.arange(int(lo), int(hi), dtype=np.int32)
                        for lo, hi in zip(starts, stops)]
