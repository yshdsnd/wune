# wune/spectrum_audio.py
from __future__ import annotations
import numpy as np
import sounddevice as sd
from .config import Config

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
        device: int | None = None,        # None=デフォルト
        agc_decay: float = 0.98,          # 大きいほどゆっくり追従（0.9〜0.99）
        smooth: float = 0.7               # 出力の表示滑らかさ（0..1, 大きいほどヌル）
    ):
        self.cfg = cfg
        self.bars = int(bars)
        self.channels_req = int(channels)
        self.sr = int(samplerate)
        self.nfft = int(blocksize)
        self.device = device
        self._smooth = float(np.clip(smooth, 0.0, 0.99))
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
        norm = (out - lo) / (hi - lo + 1e-6)
        norm = np.clip(norm, 0.0, 1.0).astype(np.float32)
        norm[norm < self.cfg.post_floor] = 0.0

        # 表示の滑らかさ
        self._out = self._smooth * self._out + (1.0 - self._smooth) * norm
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
        """ログ等間隔のバー境界を作り、rFFT周波数→バーへの対応を前計算。"""
        lo, hi = np.log10(self.fmin), np.log10(self.fmax)
        edges = np.logspace(lo, hi, self.bars + 1, base=10.0)
        idx = []
        for i in range(self.bars):
            mask = (self.freqs >= edges[i]) & (self.freqs < edges[i + 1])
            idx.append(np.nonzero(mask)[0])
        self._bin_idx = idx
