# wune/spectrum_audio.py
from __future__ import annotations
import numpy as np
import sounddevice as sd
from .config import Config
from .utils import build_band_map

# 追加：視覚用エンベロープ（per-band）
class VisEnvelope:
    def __init__(self, attack_ms=35, release_ms=180, fps=60, bars=90):
        import math
        # フレーム独立係数
        self.k_att = math.exp(-1.0 / max(1, (attack_ms/1000.0) * fps))
        self.k_rel = math.exp(-1.0 / max(1, (release_ms/1000.0) * fps))
        self.y = None
        self.n = bars
    def step(self, x):
        import numpy as np
        if self.y is None or self.y.shape != x.shape:
            self.y = np.zeros_like(x, dtype=np.float32)
        up = x > self.y
        self.y[up]  = self.k_att*self.y[up] + (1-self.k_att)*x[up]
        self.y[~up] = self.k_rel*self.y[~up] + (1-self.k_rel)*x[~up]
        return self.y

class AudioSpectrum:
    """
    マイク/ライン入力（デフォルトデバイス）→ FFT → (channels,bars) の 0..1 を返す最小実装。
    ・cross-platform（PortAudio）
    ・Hann窓 + rFFT
    ・周波数はログ等間隔で bars 個にビニング
    ・軽いAGC＋滑らかさ用のEMA
    """
    def __init__(
        self,
        cfg: Config,
        bars: int,
        channels: int = 2,
        samplerate: int = 48_000,
        blocksize: int = 2048,
        agc_decay: float = 0.98,          # 大きいほどゆっくり追従（0.9〜0.99）
        smooth: float = 0.7               # 出力の表示滑らかさ（0..1, 大きいほどヌル）
    ):
        self.cfg = cfg
        self.bars = int(bars)
        self.channels_req = int(channels)
        self.sr = int(samplerate)
        self.nfft = int(blocksize)
        self.device = cfg.input_device
        self.smooth = float(np.clip(smooth, 0.0, 0.99))
        self._agc_decay = float(np.clip(agc_decay, 0.5, 0.999))
        self.last_rms = 0.0
        self.gated = False

        # ★ デバイス情報を見て「開ける入力チャンネル数」を決める
        try:
            if self.device is None:
                in_index = sd.default.device[0]  # (input, output) の input
                if in_index is None or in_index < 0:
                    in_index = sd.query_hostapis(sd.default.hostapi)['default_input_device']
                devinfo = sd.query_devices(in_index)
            else:
                devinfo = sd.query_devices(self.device)
        except Exception:
            # 取得に失敗したら最終的に1chで試す
            devinfo = {"max_input_channels": 1}

        open_channels = max(1, min(self.channels_req, int(devinfo.get("max_input_channels", 1))))

        # ★ まず open_channels で開く。失敗したら 1ch でリトライ。
        try:
            self.stream = sd.InputStream(
                device=self.device,
                channels=open_channels,
                samplerate=self.sr,
                blocksize=self.nfft,
                dtype="float32",
            )
            self.stream.start()
        except sd.PortAudioError:
            # samplerate不一致やドライバ事情もあるので最終手段で1ch
            self.stream = sd.InputStream(
                device=self.device,
                channels=1,
                samplerate=self.sr,
                blocksize=self.nfft,
                dtype="float32",
            )
            self.stream.start()
            open_channels = 1

        self.channels_eff = open_channels  # 実際に開けたch数（後段で使う）

        # FFT 前処理
        self.window = np.hanning(self.nfft).astype(np.float32)
        self.freqs = np.fft.rfftfreq(self.nfft, d=1 / self.sr)

        # 初期の周波数レンジ（Configから上書き可）
        self.fmin = 20.0
        self.fmax = float(self.sr // 2)
        self._rebuild_bins()

        # 出力とAGC/スムージングの状態
        self._out = np.zeros((self.channels_req, self.bars), dtype=np.float32)
        self._agc = np.full((self.channels_req, self.bars), 1e-3, dtype=np.float32)

        # ビジュアルエンベロープの作成
        self._vis_env = VisEnvelope(attack_ms=35, release_ms=220, fps=self.cfg.fps, bars=self.bars)

        self.slices, self.band_mask, self.band_labels = build_band_map(self.sr, self.nfft, self.bars)

    # --- public API ----------------------------------------------------------
    def set_range(self, fmin: float, fmax: float) -> None:
        """外側（Configなど）から周波数レンジを合わせる用。"""
        fmin = max(1.0, float(fmin))
        nyq = self.sr * 0.5
        fmax = max(fmin * 1.01, float(fmax))
        fmax = min(fmax, nyq - 1.0)   # ★ ここでクランプ
        self.fmin, self.fmax = fmin, fmax
        self._rebuild_bins()

    def step(self, dt: float) -> np.ndarray:
        """
        1フレームぶん処理して (channels,bars) の 0..1 を返す。
        無音〜小音量でも 0 に張り付かないよう軽いAGCを入れている。
        """
        data, _ = self.stream.read(self.nfft)     # shape: (nfft, C)
        if data.ndim == 1:
            data = data[:, None]

        # ★ サイレンス判定（しきい値は環境で微調整）
        frame_rms = float(np.sqrt(np.mean(np.square(data.astype(np.float32)))))
        self.last_rms = frame_rms

        # ★追加：dBFSでの静寂ガード（ほぼ無音なら正規化をスキップ）
        EPS = 1e-12
        frame_db = 20.0 * np.log10(max(frame_rms, EPS))
        QUIET_DB = getattr(self.cfg, "quiet_dbfs_floor", -55.0)  # 設定から読み取れるように

        if frame_db < QUIET_DB:
            self.gated = True
            self._out *= self.cfg.silence_decay
            self._out[self._out < self.cfg.post_floor] = 0.0
            return self._out

        self.gated = (frame_rms < self.cfg.silence_rms_threshold)
        if frame_rms < self.cfg.silence_rms_threshold:
            # ゼロに落とす or 穏やかに減衰
            self._out *= self.cfg.silence_decay   # ←残像っぽく消えていく
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
        lo = np.percentile(out, 10, axis=1, keepdims=True)
        hi = np.percentile(self._agc, 95, axis=1, keepdims=True)  # 上側はAGC基準

        #    log10(パワー)スケールで 0.5 は約 +5 dB（十分“差”として認識できる量）
        min_span = getattr(self.cfg, "min_norm_span_db10", 0.6)
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

        knee = 0.25
        gamma = 1.15
        x = norm
        x = x / (x + knee)
        x = np.clip(x, 0.0, 1.0)
        x = np.power(x, gamma, dtype=np.float32)
        norm = x

        norm = self._vis_env.step(norm)
        norm[norm < self.cfg.post_floor] = 0.0

        # 表示の滑らかさ
        self._out = self.smooth * self._out + (1.0 - self.smooth) * norm
        self._out[self._out < 0.05] = 0.0
        return self._out

    def close(self) -> None:
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass

    # --- internals -----------------------------------------------------------
    def _rebuild_bins(self) -> None:
        """ログ等間隔のバー境界を作り、rFFT周波数→バーへの対応を前計算。
        各バーに最低1ビン以上必ず割り当てる（ゼロ幅禁止）。
        """
        freqs = self.freqs                         # len = nfft//2 + 1, 0..Nyquist
        nyq = self.sr * 0.5
        fmin = max(1.0, float(self.fmin))
        fmax = min(float(self.fmax), nyq * 0.999)  # Nyquist直前にクリップ

        # ログ等分の境界（bars本 → bars+1個の境界）
        edges = np.geomspace(fmin, fmax, self.bars + 1)

        # 各境界を左側に割り当て（ビン番号）
        edge_bins = np.searchsorted(freqs, edges, side="left")

        # DC(0Hz)ビンは避ける・範囲内に
        edge_bins = np.clip(edge_bins, 1, len(freqs) - 1)

        # ★単調増加＆最低幅=1を強制
        for i in range(1, len(edge_bins)):
            if edge_bins[i] <= edge_bins[i - 1]:
                edge_bins[i] = min(edge_bins[i - 1] + 1, len(freqs) - 1)

        starts = edge_bins[:-1]
        stops  = edge_bins[1:]

        # 各バーのインデックス配列を作成（必ず hi>lo）
        idx = [np.arange(int(lo), int(hi), dtype=np.int32) for lo, hi in zip(starts, stops)]
        self._bin_idx = idx

