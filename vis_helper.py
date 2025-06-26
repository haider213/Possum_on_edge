"""
vis_helper.py – core helper functions for the Invasive-Species bio-acoustics dataset
------------------------------------------------------------------------------
This module purposefully **excludes** any Gradio/UI code.  It only contains the
utility routines you’ll reuse in notebooks, scripts, or a separate `app.py`.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import List, Tuple

import librosa
import librosa.display as ld
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# 1. Dataset configuration                                                    #
# --------------------------------------------------------------------------- #

ROOT_DIR = Path("./data/invasive-species").resolve()
TRAIN_DIR: Path = ROOT_DIR / "training" / "training"
TEST_DIR: Path = ROOT_DIR / "test" / "test"
ANNOT_CSV: Path = ROOT_DIR / "annotations.csv"

SR: int = 16_000              # audio sample-rate (Hz)
DEFAULT_CLIP_SEC: float = 1.0  # one-second window


# --------------------------------------------------------------------------- #
# 2. File / annotation helpers                                                #
# --------------------------------------------------------------------------- #

def load_annotations(path: Path = ANNOT_CSV) -> pd.DataFrame:
    """Load the annotations CSV into a DataFrame."""
    return pd.read_csv(path)


def wav_files(directory: Path) -> List[str]:
    """All *.wav files in *directory* (non-recursive)."""
    return [f for f in os.listdir(directory) if f.lower().endswith(".wav")]


def audio_path(fname: str) -> Path:
    """
    Resolve *fname* to its absolute path in TRAIN_DIR or TEST_DIR.

    Raises
    ------
    FileNotFoundError
        if the file isn’t present in either directory.
    """
    p_train = TRAIN_DIR / fname
    p_test = TEST_DIR / fname
    if p_train.exists():
        return p_train
    if p_test.exists():
        return p_test
    raise FileNotFoundError(f"{fname!r} not found in training or test dirs")


def load_audio(fname: str, sr: int = SR) -> Tuple[np.ndarray, int]:
    """Load a WAV file (mono) at *sr*.  Returns (samples, sr)."""
    path = audio_path(fname)
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y, sr


# --------------------------------------------------------------------------- #
# 3. Visualisation utilities (Matplotlib)                                     #
# --------------------------------------------------------------------------- #

def plot_waveform(
    y: np.ndarray, sr: int,
    start: float | None = None,
    end: float | None = None,
    figsize: Tuple[int, int] = (10, 3)
) -> plt.Figure:
    """Waveform with optional start/end markers (in seconds)."""
    t = np.arange(len(y)) / sr
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t, y, lw=0.6)
    if start is not None:
        ax.axvline(start, ls="--", c="tab:red", label=f"start={start:.2f}s")
    if end is not None:
        ax.axvline(end, ls="--", c="tab:green", label=f"end={end:.2f}s")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    if start is not None or end is not None:
        ax.legend()
    fig.tight_layout()
    return fig

def rolling_energy_anim(
    y: np.ndarray,
    sr: int,
    window_sec: float = 1.0,
    interval_ms: int = 50,
) -> str:
    """Return a JavaScript‑based HTML animation that sweeps a *window_sec*
    rectangle across the waveform and displays its instantaneous energy.

    The HTML string can be fed directly to a **gr.HTML** component.
    """
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Rectangle

    duration = len(y) / sr
    t = np.arange(len(y)) / sr

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, y, color="lightgray", linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, duration)
    ymin, ymax = np.min(y) * 1.1, np.max(y) * 1.1
    ax.set_ylim(ymin, ymax)

    # Rolling window rectangle
    window_rect = Rectangle((0, ymin), window_sec, ymax - ymin,
                            color="tab:orange", alpha=0.3)
    ax.add_patch(window_rect)

    # Energy text overlay
    energy_text = ax.text(0.01, ymax * 0.9, "", color="black",
                          fontsize=10, ha="left")

    n_frames = int(np.ceil(duration * 1000 / interval_ms))

    def init():
        window_rect.set_x(0)
        energy_text.set_text("")
        return window_rect, energy_text

    def update(frame):
        current_time = (frame * interval_ms) / 1000.0
        if current_time > duration:
            current_time = duration
        win_end = current_time
        win_start = max(0.0, win_end - window_sec)

        window_rect.set_x(win_start)
        window_rect.set_width(win_end - win_start)

        s_idx = int(win_start * sr)
        e_idx = int(win_end * sr)
        energy = float(np.sum(y[s_idx:e_idx] ** 2)) if e_idx > s_idx else 0.0

        energy_text.set_x(win_start + 0.02)
        energy_text.set_text(f"Energy: {energy:.2f}")
        return window_rect, energy_text

    anim = FuncAnimation(fig, update, frames=n_frames + 1,
                         init_func=init, blit=True, interval=interval_ms)
    html = anim.to_jshtml()
    plt.close(fig)
    return html
def plot_spectrogram(
    y: np.ndarray, sr: int,
    n_mels: int = 128,
    figsize: Tuple[int, int] = (10, 3)
) -> plt.Figure:
    """Return a Mel-spectrogram (dB) figure."""
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=sr // 2)

    S_dB = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=figsize)
    img = ld.specshow(S_dB, sr=sr, fmax=sr // 2,
                      x_axis="time", y_axis="mel", ax=ax)
    ax.set_title("Mel-Spectrogram (dB)")
    fig.colorbar(img, ax=ax, format="%.0f dB")
    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig: plt.Figure) -> bytes:
    """Convert a Matplotlib figure to raw PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# --------------------------------------------------------------------------- #
# 4. Signal manipulation                                                      #
# --------------------------------------------------------------------------- #

def clip_audio_segment(
    y: np.ndarray,
    sr: int,
    start_sec: float,
    duration_sec: float = DEFAULT_CLIP_SEC
) -> np.ndarray:
    """
    Slice out a fixed-length segment.

    Parameters
    ----------
    y : np.ndarray
        The full-length mono signal.
    sr : int
        Sample rate (Hz).
    start_sec : float
        Start time of the window (seconds).
    duration_sec : float, default 1.0
        Length of the segment to return (seconds).

    Returns
    -------
    np.ndarray
        Samples in the [start, start+duration) window, clipped to y’s length.
    """
    start_idx = int(max(0, start_sec) * sr)
    end_idx = int(min(len(y), start_idx + duration_sec * sr))
    return y[start_idx:end_idx]

def fig_to_bytes(fig: plt.Figure) -> bytes:
    """Convert a Matplotlib figure to raw PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()