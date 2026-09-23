# wune/spectrum_audio.py
from __future__ import annotations
from contextlib import ExitStack
import math

import numpy as np
import soundcard as sc
from .config import Config

# 視覚用エンベロープ（per-band）
class VisEnvelope:
    def __init__(self, attack_ms=35, release_ms=180, fps=60):
        # フレーム独立係数
        self.k_att = math.exp(-1.0 / max(1, (attack_ms/1000.0) * fps))
        self.k_rel = math.exp(-1.0 / max(1, (release_ms/1000.0) * fps))
        self.y = None
    def step(self, x):
        if self.y is None or self.y.shape != x.shape:
            self.y = np.zeros_like(x, dtype=np.float32)
        up = x > self.y
        self.y[up]  = self.k_att*self.y[up] + (1-self.k_att)*x[up]
        self.y[~up] = self.k_rel*self.y[~up] + (1-self.k_rel)*x[~up]
        return self.y

class AudioSpectrum:
    """Windows WASAPI loopback → Hann/rFFT → log-power bands → display levels."""
    def __init__(
        self,
        cfg: Config,
        bars: int,
        channels: int = 2,
        samplerate: int | None = None,
        blocksize: int | None = None,
        agc_decay: float | None = None,          # 大きいほどゆっくり追従（0.9〜0.99）
        smooth: float | None = None               # 出力の表示滑らかさ（0..1, 大きいほどヌル）
    ):
        self.cfg = cfg
        self.bars = int(bars)
        self.channels_req = int(channels)
        self.sr = int(cfg.sample_rate if samplerate is None else samplerate)
        self.nfft = int(cfg.block_size if blocksize is None else blocksize)
        self.smoothing = float(np.clip(cfg.smoothing if smooth is None else smooth, 0.0, 0.99))
        self._agc_decay = float(np.clip(cfg.agc_decay if agc_decay is None else agc_decay, 0.5, 0.999))
        self.last_rms = 0.0
        self.gated = False

        # FFT 前処理
        self.window = np.hanning(self.nfft).astype(np.float32)
        self.freqs = np.fft.rfftfreq(self.nfft, d=1 / self.sr)

        # 周波数ごとの重み付け配列を作成
        weights = np.ones(len(self.freqs), dtype=np.float32)
        cutoff_freq = 20000
        if self.sr > 48000:
            cutoff_freq = 48000

        for i, f in enumerate(self.freqs):
            if 10000 <= f < 16000:
                weights[i] = 1.8  # 10k-16kHzを強調 (倍率は好みで調整)
            elif 16000 <= f < cutoff_freq:
                weights[i] = 2.5  # 16kHz以上をさらに強調
            elif f >= cutoff_freq:
                weights[i] = 0.001 # カットオフ周波数以上は事実上カット (log10に通すためゼロにしない)
        self.weights = np.log10(weights) # ★対数パワーに加算するので、log10しておく

        # 初期の周波数レンジ（Configから上書き可）
        self.fmin = 20.0
        self.fmax = float(self.sr // 2)
        self._rebuild_bins()

        # 出力とAGC/スムージングの状態
        self._out = np.zeros((self.channels_req, self.bars), dtype=np.float32)
        self._agc = np.full((self.channels_req, self.bars), 1e-3, dtype=np.float32)

        # ビジュアルエンベロープの作成
        self._vis_env = VisEnvelope(attack_ms=self.cfg.vis_attack_ms, release_ms=self.cfg.vis_release_ms, fps=self.cfg.fps)

        # Open last so initialization errors cannot leave capture running.
        self._capture_context = ExitStack()
        self.stream = self._open_loopback()


    # --- public API ----------------------------------------------------------
    def set_range(self, fmin: float, fmax: float) -> None:
        """外側（Configなど）から周波数レンジを合わせる用。"""
        fmin = max(1.0, float(fmin))
        nyq = self.sr * 0.5
        fmax = max(fmin * 1.01, float(fmax))
        fmax = min(fmax, nyq - 1.0)   # ここでクランプ
        self.fmin, self.fmax = fmin, fmax
        self._rebuild_bins()

    def step(self, dt: float) -> np.ndarray:
        """
        1フレームぶん処理して (channels,bars) の 0..1 を返す。
        無音〜小音量でも 0 に張り付かないよう軽いAGCを入れている。
        """
        data = self.stream.record(numframes=self.nfft)     # shape: (nfft, C)
        if data.ndim == 1:
            data = data[:, None]

        # サイレンス判定（しきい値は環境で微調整）
        frame_rms = float(np.sqrt(np.mean(np.square(data.astype(np.float32)))))
        self.last_rms = frame_rms

        # dBFSでの静寂ガード（ほぼ無音なら正規化をスキップ）
        EPS = 1e-12
        frame_db = 20.0 * np.log10(max(frame_rms, EPS))
        QUIET_DB = self.cfg.quiet_dbfs_floor  # 設定から読み取れるように

        if frame_db < QUIET_DB:
            self.gated = True
            self._out *= self.cfg.silence_decay
            self._out[self._out < self.cfg.post_floor] = 0.0
            return self._out

        self.gated = (frame_rms < self.cfg.silence_rms_threshold)
        if frame_rms < self.cfg.silence_rms_threshold:
            # ゼロに落とす or 穏やかに減衰
            self._out *= self.cfg.silence_decay   # 無音時は既存の減衰率を維持
            self._out[self._out < self.cfg.post_floor] = 0.0
            return self._out

        C_in = data.shape[1]

        # 入力chが要求と違う場合に対処（足りなければ繰り返し、余れば先頭だけ）
        if C_in < self.channels_req:
            data = np.repeat(data, repeats=self.channels_req, axis=1)[:, : self.channels_req]
        elif C_in > self.channels_req:
            data = data[:, : self.channels_req]

        # 出力バッファ
        out = np.zeros_like(self._out)

        # chごとにFFT→バービニング（logパワー平均）
        for ch in range(self.channels_req):
            x = data[:, ch].astype(np.float32, copy=False)
            spec = np.fft.rfft(self.window * x)
            pwr = (spec.real**2 + spec.imag**2).astype(np.float32) + 1e-12  # power
            logp = np.log10(pwr)  # 聴感に寄せるため対数圧縮
            logp += self.weights # 周波数ごとの重みをここで加算

            # ビンに平均で落とし込み
            for b, idx in enumerate(self._bin_idx):
                if idx.size:
                    out[ch, b] = np.mean(logp[idx])
                else:
                    out[ch, b] = -12.0  # ほぼ無音扱い

        # --- 0..1 正規化（チャンネル独立の簡易AGC + スムージング） ---
        # 更新式: agc = max( out, agc*decay ) を各バーで
        self._agc = np.maximum(out, self._agc * self._agc_decay)

        # パーセンタイル正規化（極端値の影響を弱める）
        lo = np.percentile(out, self.cfg.norm_lo_pct, axis=1, keepdims=True)
        hi = np.percentile(self._agc, self.cfg.norm_hi_pct, axis=1, keepdims=True)  # 上側はAGC基準

        #    log10(パワー)スケールで 0.5 は約 +5 dB（十分“差”として認識できる量）
        min_span = self.cfg.min_norm_span_db10
        span = hi - lo
        use_abs = span < min_span   # (ch,1) ブール

        # ① 絶対dB基準の正規化（無条件で先に作る）
        dbmin10   = self.cfg.db_min / 10.0        # -60 dB → -6.0
        dbrange10 = (self.cfg.db_max - self.cfg.db_min) / 10.0  # 60 dB → 6.0
        abs_norm = (out - dbmin10) / (dbrange10 + 1e-6)
        abs_norm = np.clip(abs_norm, 0.0, 1.0)

        # 通常のパーセンタイル正規化
        pct_norm = (out - lo) / (np.maximum(span, min_span) + 1e-6)
        pct_norm = np.clip(pct_norm, 0.0, 1.0)

        # ③ スパンが小さいチャンネルは絶対dBにフォールバック
        #    use_abs は (ch,1) なので (ch,bins) に自動ブロードキャストされます
        norm = np.where(use_abs, abs_norm, pct_norm).astype(np.float32)

        knee = self.cfg.compression_knee
        gamma = self.cfg.compression_gamma
        x = norm
        x = x / (x + knee)
        x = np.clip(x, 0.0, 1.0)
        x = np.power(x, gamma, dtype=np.float32)
        norm = x

        norm = self._vis_env.step(norm)
        norm[norm < self.cfg.post_floor] = 0.0

        # 表示の滑らかさ
        self._out = self.smoothing * self._out + (1.0 - self.smoothing) * norm
        self._out[self._out < self.cfg.output_floor] = 0.0
        return self._out

    def close(self) -> None:
        self._capture_context.close()

    def _open_loopback(self):
        """Capture the render endpoint, never a microphone or an output player."""
        from .soundcard_compat import prepare_soundcard
        prepare_soundcard()
        speaker = (sc.default_speaker() if self.cfg.output_device is None
                   else sc.get_speaker(self.cfg.output_device))
        if speaker is None:
            raise RuntimeError("No Windows playback device is available.")
        if speaker.channels < 2:
            raise RuntimeError("Select a stereo Windows playback device for loopback.")

        # Resolve by endpoint ID: names can also match ordinary microphones.
        loopback = sc.get_microphone(id=speaker.id, include_loopback=True)
        if not loopback.isloopback:
            raise RuntimeError("The selected playback endpoint has no loopback capture.")
        self.device = speaker.name
        self.channels_eff = 2
        # Shared mode leaves normal playback running. Avoid SoundCard's known
        # single-channel WASAPI issue even when the display is configured mono.
        recorder = loopback.recorder(
            samplerate=self.sr, channels=[0, 1], blocksize=self.nfft,
            exclusive_mode=False,
        )
        return self._capture_context.enter_context(recorder)

    def _rebuild_bins(self) -> None:
        """ログ等間隔のバー境界を作り、rFFT周波数→バー対応を前計算。
        既存の境界計算を維持する（高域端の扱いの変更は別の調整作業）。
        """
        freqs = self.freqs                        # len = nfft//2+1, 0..Nyquist
        nyq = self.sr * 0.5

        fmin = max(1.0, float(self.fmin))
        fmax = min(float(self.fmax), nyq * 0.999)  # Nyquist手前に倒す（安全側）

        # ログ等分の境界（bars → bars+1 個）
        edges = np.geomspace(fmin, fmax, self.bars + 1)

        # 各境界を「左側に最も近いビン」へ（整数化）
        edge_bins = np.searchsorted(freqs, edges, side="left")
        edge_bins = np.clip(edge_bins, 1, len(freqs) - 1)  # DC(0)は避ける

        # 単調増加と最小幅=1を強制
        for i in range(1, len(edge_bins)):
            if edge_bins[i] <= edge_bins[i-1]:
                edge_bins[i] = min(edge_bins[i-1] + 1, len(freqs) - 1)

        starts = edge_bins[:-1]
        # Preserve the existing mapping: zip uses the first bars stops.
        # The appended endpoint is currently unused; changing it would retune bands.
        stops  = np.append(edge_bins[1:], len(freqs))

        # 各バーのビン配列（hi は排他）
        self._bin_idx = [np.arange(int(lo), int(hi), dtype=np.int32)
                        for lo, hi in zip(starts, stops)]
