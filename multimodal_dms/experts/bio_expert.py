"""
Bio Expert — Hand-crafted physiological features

BVP → HRV features (mean_hr, sdnn, rmssd, lf_hf_ratio)
EDA → SCR features (mean_scl, std_scl, n_peaks, mean_amp, auc)
TEMP → trend features (mean, slope, range)
HR → direct features (mean, std, range)

Total: 15 features per segment
Based on: Kreibig (2010), Healey & Picard (2005)
"""
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def extract_bio_features(npz_data: dict) -> np.ndarray:
    """
    npz_data: dict with keys bvp, eda, temp, hr (each shape (160, 2))
    Returns: (15,) float32 feature vector
    """
    features = []

    # === BVP (4 features) ===
    try:
        bvp = npz_data["bvp"][:, 0] if "bvp" in npz_data else None
        if bvp is not None and len(bvp) > 10:
            import neurokit2 as nk
            # Clean BVP signal
            bvp_clean = nk.ppg_clean(bvp, sampling_rate=32)
            # Find peaks
            peaks_info = nk.ppg_findpeaks(bvp_clean, sampling_rate=32)
            peak_locs = peaks_info.get("PPG_Peaks", [])

            if len(peak_locs) >= 3:
                ibi = np.diff(peak_locs) / 32.0 * 1000  # ms
                mean_hr = 60000.0 / np.mean(ibi) if np.mean(ibi) > 0 else 0
                sdnn = np.std(ibi)
                rmssd = np.sqrt(np.mean(np.diff(ibi) ** 2))

                # LF/HF (simplified)
                if len(ibi) > 8:
                    from scipy import signal as sig
                    f, psd = sig.welch(ibi, fs=1000.0 / np.mean(ibi), nperseg=min(len(ibi), 64))
                    lf_mask = (f >= 0.04) & (f < 0.15)
                    hf_mask = (f >= 0.15) & (f < 0.4)
                    lf = np.trapz(psd[lf_mask], f[lf_mask]) if lf_mask.any() else 0
                    hf = np.trapz(psd[hf_mask], f[hf_mask]) if hf_mask.any() else 0
                    lf_hf = lf / (hf + 1e-8)
                else:
                    lf_hf = 0.0

                features.extend([mean_hr, sdnn, rmssd, lf_hf])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
    except Exception:
        features.extend([0.0, 0.0, 0.0, 0.0])

    # === EDA (5 features) ===
    try:
        eda = npz_data["eda"][:, 0] if "eda" in npz_data else None
        if eda is not None and len(eda) > 10:
            mean_scl = float(np.mean(eda))
            std_scl = float(np.std(eda))

            # Simple peak detection
            eda_diff = np.diff(eda)
            peaks = np.where((eda_diff[:-1] > 0) & (eda_diff[1:] <= 0))[0] + 1
            threshold = np.mean(eda) + 0.5 * np.std(eda)
            peaks = peaks[eda[peaks] > threshold]
            n_peaks = len(peaks)

            if n_peaks > 0:
                mean_amp = float(np.mean(eda[peaks] - np.mean(eda)))
            else:
                mean_amp = 0.0
            auc = float(np.trapz(np.abs(eda - np.mean(eda))))

            features.extend([mean_scl, std_scl, float(n_peaks), mean_amp, auc])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    except Exception:
        features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

    # === TEMP (3 features) ===
    try:
        temp = npz_data["temp"][:, 0] if "temp" in npz_data else None
        if temp is not None and len(temp) > 2:
            mean_temp = float(np.mean(temp))
            temp_range = float(np.max(temp) - np.min(temp))
            # Slope (linear regression)
            x = np.arange(len(temp))
            if np.std(x) > 0:
                slope = float(np.polyfit(x, temp, 1)[0])
            else:
                slope = 0.0
            features.extend([mean_temp, slope, temp_range])
        else:
            features.extend([0.0, 0.0, 0.0])
    except Exception:
        features.extend([0.0, 0.0, 0.0])

    # === HR (3 features) ===
    try:
        hr = npz_data["hr"][:, 0] if "hr" in npz_data else None
        if hr is not None and len(hr) > 2:
            mean_hr = float(np.mean(hr))
            hr_std = float(np.std(hr))
            hr_range = float(np.max(hr) - np.min(hr))
            features.extend([mean_hr, hr_std, hr_range])
        else:
            features.extend([0.0, 0.0, 0.0])
    except Exception:
        features.extend([0.0, 0.0, 0.0])

    result = np.array(features, dtype=np.float32)
    # Replace NaN/Inf
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result


def extract_bio_features_v2(npz_data: dict) -> dict:
    """
    Enhanced bio feature extraction with per-channel quality scores.

    Returns:
        {
            "features":   (15,) float32 — same as extract_bio_features
            "bvp_valid":  bool  — BVP signal usable
            "eda_valid":  bool  — EDA signal usable
            "hr_valid":   bool  — HR signal usable
            "bio_quality": float — overall quality [0, 1]
            "valid":       bool  — overall validity
        }
    """
    features = extract_bio_features(npz_data)

    # Per-channel validity
    bvp_valid = not np.allclose(features[0:4], 0.0)
    eda_valid = not np.allclose(features[4:9], 0.0)
    hr_valid = not np.allclose(features[12:15], 0.0)
    # TEMP is passive (always somewhat valid if present)

    # Check for NaN/extreme values in raw signals
    nan_count = 0
    total_count = 0
    for key in ["bvp", "eda", "temp", "hr"]:
        if key in npz_data:
            arr = npz_data[key]
            total_count += arr.size
            nan_count += np.isnan(arr).sum() + np.isinf(arr).sum()

    nan_ratio = nan_count / max(total_count, 1)
    quality_from_nan = 1.0 - nan_ratio

    # Quality from feature validity
    n_valid = sum([bvp_valid, eda_valid, hr_valid])
    quality_from_valid = n_valid / 3.0

    # Combined quality
    bio_quality = float(0.5 * quality_from_nan + 0.5 * quality_from_valid)
    overall_valid = n_valid >= 1  # at least 1 channel valid

    return {
        "features": features,
        "bvp_valid": bvp_valid,
        "eda_valid": eda_valid,
        "hr_valid": hr_valid,
        "bio_quality": bio_quality,
        "valid": overall_valid,
    }


# Feature names for reference
BIO_FEATURE_NAMES = [
    "bvp_mean_hr", "bvp_sdnn", "bvp_rmssd", "bvp_lf_hf",
    "eda_mean_scl", "eda_std_scl", "eda_n_peaks", "eda_mean_amp", "eda_auc",
    "temp_mean", "temp_slope", "temp_range",
    "hr_mean", "hr_std", "hr_range",
]
