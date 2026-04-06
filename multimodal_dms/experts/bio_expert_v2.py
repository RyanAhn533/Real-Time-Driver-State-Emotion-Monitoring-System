"""
Bio Expert V2 — Incremental physiological feature extraction

Optimizations over bio_expert.py:
1. IncrementalBioProcessor with ring buffers for PPG/EDA/Temp/HR
2. neurokit2 / scipy imported once at init (not per call)
3. HRV (LF/HF) computed every 5 seconds, not every 100ms frame
4. Simple stats (mean, std, range) computed every frame (cheap)
5. Backward-compatible extract_bio_features_v2() interface

Ring buffer sizes:
  - PPG/BVP: 5 min @ 32 Hz = 9600 samples (meaningful HRV)
  - EDA:     5 min @ 4 Hz  = 1200 samples
  - Temp:    5 min @ 4 Hz  = 1200 samples
  - HR:      5 min @ 1 Hz  = 300 samples

Author: Auto-generated optimization
"""

import time
import numpy as np
import warnings
from collections import deque

warnings.filterwarnings("ignore")

# ── Lazy one-time imports ──────────────────────────────────────────────
_nk = None
_welch = None


def _ensure_imports():
    """Import neurokit2 and scipy.signal.welch exactly once."""
    global _nk, _welch
    if _nk is None:
        try:
            import neurokit2 as nk
            _nk = nk
        except ImportError:
            _nk = False  # sentinel: not available
    if _welch is None:
        try:
            from scipy.signal import welch
            _welch = welch
        except ImportError:
            _welch = False


# ── Ring Buffer ────────────────────────────────────────────────────────
class RingBuffer:
    """Fixed-size numpy ring buffer — O(1) append, O(1) snapshot."""

    __slots__ = ("_buf", "_maxlen", "_idx", "_full")

    def __init__(self, maxlen: int, dtype=np.float64):
        self._buf = np.zeros(maxlen, dtype=dtype)
        self._maxlen = maxlen
        self._idx = 0
        self._full = False

    def extend(self, data: np.ndarray):
        """Append a 1-D array of samples."""
        n = len(data)
        if n == 0:
            return
        if n >= self._maxlen:
            # Only keep the last maxlen samples
            self._buf[:] = data[-self._maxlen:]
            self._idx = 0
            self._full = True
            return
        end = self._idx + n
        if end <= self._maxlen:
            self._buf[self._idx:end] = data
        else:
            first = self._maxlen - self._idx
            self._buf[self._idx:] = data[:first]
            self._buf[:n - first] = data[first:]
        self._idx = end % self._maxlen
        if end >= self._maxlen:
            self._full = True

    def get(self) -> np.ndarray:
        """Return contiguous view of stored samples (oldest→newest)."""
        if not self._full:
            return self._buf[:self._idx].copy()
        return np.roll(self._buf, -self._idx).copy()

    @property
    def count(self) -> int:
        return self._maxlen if self._full else self._idx

    def clear(self):
        self._buf[:] = 0
        self._idx = 0
        self._full = False


# ── Incremental Bio Processor ─────────────────────────────────────────
class IncrementalBioProcessor:
    """
    Stateful bio-signal processor with ring buffers.

    Parameters
    ----------
    ppg_sr : int   – PPG sampling rate (default 32 Hz)
    eda_sr : int   – EDA sampling rate (default 4 Hz)
    temp_sr : int  – Temperature sampling rate (default 4 Hz)
    hr_sr : int    – HR sampling rate (default 1 Hz)
    hrv_interval : float – Minimum seconds between HRV recomputations (default 5.0)
    buffer_minutes : float – Ring buffer duration in minutes (default 5.0)
    """

    def __init__(
        self,
        ppg_sr: int = 32,
        eda_sr: int = 4,
        temp_sr: int = 4,
        hr_sr: int = 1,
        hrv_interval: float = 5.0,
        buffer_minutes: float = 5.0,
    ):
        self.ppg_sr = ppg_sr
        self.eda_sr = eda_sr
        self.temp_sr = temp_sr
        self.hr_sr = hr_sr
        self.hrv_interval = hrv_interval

        # Ring buffers
        buf_sec = int(buffer_minutes * 60)
        self.ppg_buf = RingBuffer(ppg_sr * buf_sec)
        self.eda_buf = RingBuffer(eda_sr * buf_sec)
        self.temp_buf = RingBuffer(temp_sr * buf_sec)
        self.hr_buf = RingBuffer(hr_sr * buf_sec)

        # Cached HRV results
        self._last_hrv_time = 0.0
        self._cached_hrv = {
            "mean_hr": 0.0,
            "sdnn": 0.0,
            "rmssd": 0.0,
            "lf_hf": 0.0,
        }

        # One-time import
        _ensure_imports()

    # ── Internal: HRV from BVP (expensive) ────────────────────────────
    def _compute_hrv(self, bvp: np.ndarray) -> dict:
        """Compute HRV metrics from BVP signal. Uses neurokit2 if available."""
        result = {"mean_hr": 0.0, "sdnn": 0.0, "rmssd": 0.0, "lf_hf": 0.0}

        if _nk is False or len(bvp) < 64:
            return result

        try:
            bvp_clean = _nk.ppg_clean(bvp, sampling_rate=self.ppg_sr)
            peaks_info = _nk.ppg_findpeaks(bvp_clean, sampling_rate=self.ppg_sr)
            peak_locs = peaks_info.get("PPG_Peaks", [])

            if len(peak_locs) < 3:
                return result

            ibi = np.diff(peak_locs) / float(self.ppg_sr) * 1000.0  # ms

            # Filter physiologically implausible IBIs (< 300ms or > 2000ms)
            ibi = ibi[(ibi > 300) & (ibi < 2000)]
            if len(ibi) < 3:
                return result

            result["mean_hr"] = 60000.0 / np.mean(ibi)
            result["sdnn"] = float(np.std(ibi))
            result["rmssd"] = float(np.sqrt(np.mean(np.diff(ibi) ** 2)))

            # LF/HF — only meaningful with sufficient data (>= ~60s of IBI)
            if _welch is not False and len(ibi) >= 16:
                fs_ibi = 1000.0 / np.mean(ibi)
                nperseg = min(len(ibi), 128)
                f, psd = _welch(ibi, fs=fs_ibi, nperseg=nperseg)
                lf_mask = (f >= 0.04) & (f < 0.15)
                hf_mask = (f >= 0.15) & (f < 0.4)
                lf = float(np.trapz(psd[lf_mask], f[lf_mask])) if lf_mask.any() else 0.0
                hf = float(np.trapz(psd[hf_mask], f[hf_mask])) if hf_mask.any() else 0.0
                result["lf_hf"] = lf / (hf + 1e-8)

        except Exception:
            pass

        return result

    # ── Internal: Simple stats (cheap) ────────────────────────────────
    @staticmethod
    def _simple_stats(arr: np.ndarray) -> tuple:
        """Return (mean, std, range) — O(n) numpy ops."""
        if len(arr) < 2:
            return (0.0, 0.0, 0.0)
        return (float(np.mean(arr)), float(np.std(arr)), float(np.ptp(arr)))

    # ── Internal: EDA features (moderate cost) ────────────────────────
    @staticmethod
    def _eda_features(eda: np.ndarray) -> tuple:
        """Return (mean_scl, std_scl, n_peaks, mean_amp, auc)."""
        if len(eda) < 4:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        mean_scl = float(np.mean(eda))
        std_scl = float(np.std(eda))

        # Simple peak detection (same logic as v1)
        eda_diff = np.diff(eda)
        peaks = np.where((eda_diff[:-1] > 0) & (eda_diff[1:] <= 0))[0] + 1
        threshold = mean_scl + 0.5 * std_scl
        peaks = peaks[eda[peaks] > threshold]
        n_peaks = float(len(peaks))
        mean_amp = float(np.mean(eda[peaks] - mean_scl)) if len(peaks) > 0 else 0.0
        auc = float(np.trapz(np.abs(eda - mean_scl)))

        return (mean_scl, std_scl, n_peaks, mean_amp, auc)

    # ── Internal: Temp features ───────────────────────────────────────
    @staticmethod
    def _temp_features(temp: np.ndarray) -> tuple:
        """Return (mean, slope, range)."""
        if len(temp) < 3:
            return (0.0, 0.0, 0.0)
        mean_t = float(np.mean(temp))
        range_t = float(np.ptp(temp))
        x = np.arange(len(temp), dtype=np.float64)
        slope = float(np.polyfit(x, temp, 1)[0]) if np.std(temp) > 1e-10 else 0.0
        return (mean_t, slope, range_t)

    # ── Public: feed new data ─────────────────────────────────────────
    def feed(self, npz_data: dict):
        """
        Append new segment data to ring buffers.
        npz_data: dict with keys bvp/eda/temp/hr, each shape (N, 2).
        """
        for key, buf in [
            ("bvp", self.ppg_buf),
            ("eda", self.eda_buf),
            ("temp", self.temp_buf),
            ("hr", self.hr_buf),
        ]:
            if key in npz_data:
                arr = npz_data[key]
                if arr.ndim == 2:
                    arr = arr[:, 0]
                valid = arr[~(np.isnan(arr) | np.isinf(arr))]
                if len(valid) > 0:
                    buf.extend(valid)

    # ── Public: extract features ──────────────────────────────────────
    def extract(self) -> np.ndarray:
        """
        Extract 15-dim feature vector from current ring buffer state.
        HRV recomputed only every `hrv_interval` seconds.

        Returns: (15,) float32
        """
        now = time.monotonic()
        features = []

        # === BVP / HRV (4 features) ===
        bvp = self.ppg_buf.get()
        if len(bvp) > 10:
            if (now - self._last_hrv_time) >= self.hrv_interval:
                self._cached_hrv = self._compute_hrv(bvp)
                self._last_hrv_time = now
            hrv = self._cached_hrv
            features.extend([hrv["mean_hr"], hrv["sdnn"], hrv["rmssd"], hrv["lf_hf"]])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])

        # === EDA (5 features) ===
        eda = self.eda_buf.get()
        features.extend(self._eda_features(eda))

        # === TEMP (3 features) ===
        temp = self.temp_buf.get()
        features.extend(self._temp_features(temp))

        # === HR (3 features) ===
        hr = self.hr_buf.get()
        features.extend(self._simple_stats(hr))

        result = np.array(features, dtype=np.float32)
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result

    def reset(self):
        """Clear all buffers and cached HRV."""
        self.ppg_buf.clear()
        self.eda_buf.clear()
        self.temp_buf.clear()
        self.hr_buf.clear()
        self._last_hrv_time = 0.0
        self._cached_hrv = {"mean_hr": 0.0, "sdnn": 0.0, "rmssd": 0.0, "lf_hf": 0.0}


# ── Global singleton (lazy init) ──────────────────────────────────────
_processor: IncrementalBioProcessor = None


def _get_processor() -> IncrementalBioProcessor:
    global _processor
    if _processor is None:
        _processor = IncrementalBioProcessor()
    return _processor


# ── Backward-compatible API ────────────────────────────────────────────

def extract_bio_features(npz_data: dict) -> np.ndarray:
    """
    Drop-in replacement for bio_expert.extract_bio_features().

    Feeds new data into the incremental processor and returns (15,) features.
    """
    proc = _get_processor()
    proc.feed(npz_data)
    return proc.extract()


def extract_bio_features_v2(npz_data: dict) -> dict:
    """
    Drop-in replacement for bio_expert.extract_bio_features_v2().

    Returns dict with 'features', validity flags, and quality score.
    """
    features = extract_bio_features(npz_data)

    bvp_valid = not np.allclose(features[0:4], 0.0)
    eda_valid = not np.allclose(features[4:9], 0.0)
    hr_valid = not np.allclose(features[12:15], 0.0)

    # NaN/Inf quality check on input
    nan_count = 0
    total_count = 0
    for key in ["bvp", "eda", "temp", "hr"]:
        if key in npz_data:
            arr = npz_data[key]
            total_count += arr.size
            nan_count += np.isnan(arr).sum() + np.isinf(arr).sum()

    nan_ratio = nan_count / max(total_count, 1)
    quality_from_nan = 1.0 - nan_ratio

    n_valid = sum([bvp_valid, eda_valid, hr_valid])
    quality_from_valid = n_valid / 3.0
    bio_quality = float(0.5 * quality_from_nan + 0.5 * quality_from_valid)

    return {
        "features": features,
        "bvp_valid": bvp_valid,
        "eda_valid": eda_valid,
        "hr_valid": hr_valid,
        "bio_quality": bio_quality,
        "valid": n_valid >= 1,
    }


def get_processor() -> IncrementalBioProcessor:
    """Access the global IncrementalBioProcessor instance."""
    return _get_processor()


def reset_processor():
    """Reset the global processor state (e.g. on session restart)."""
    global _processor
    if _processor is not None:
        _processor.reset()


# Feature names — same as bio_expert.py
BIO_FEATURE_NAMES = [
    "bvp_mean_hr", "bvp_sdnn", "bvp_rmssd", "bvp_lf_hf",
    "eda_mean_scl", "eda_std_scl", "eda_n_peaks", "eda_mean_amp", "eda_auc",
    "temp_mean", "temp_slope", "temp_range",
    "hr_mean", "hr_std", "hr_range",
]
