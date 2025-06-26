import csv
import io
from pathlib import Path

import gradio as gr
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ENERGY_LOG = Path("energy_log.csv")

class Viewer:
    def __init__(self):
        self.df = pd.DataFrame()
        self.root = Path(".")
        self.idx = 0
        self.current_y_norm = np.array([])
        self.current_sr = 16000

    def load_dataset(self, data_root: str, annot_csv: str):
        self.root = Path(data_root)
        self.df = pd.read_csv(annot_csv)
        if 'Label' in self.df.columns:
            self.df = self.df[self.df['Label'] == 1].reset_index(drop=True)
        self.idx = 0
        return self._prepare_output()

    def next_clip(self):
        if self.df.empty:
            raise gr.Error("Dataset not loaded")
        self.idx = (self.idx + 1) % len(self.df)
        return self._prepare_output()

    def prev_clip(self):
        if self.df.empty:
            raise gr.Error("Dataset not loaded")
        self.idx = (self.idx - 1) % len(self.df)
        return self._prepare_output()

    def detect_calls(self, threshold: float):
        """Return start times (s) of 1s windows with energy >= threshold."""
        if self.current_y_norm.size == 0:
            raise gr.Error("No clip loaded")
        win = int(self.current_sr)
        starts = []
        for s in range(0, len(self.current_y_norm) - win + 1, win):
            seg = self.current_y_norm[s:s + win]
            energy = float(np.sum(seg ** 2))
            if energy >= threshold:
                starts.append(s / self.current_sr)
        if not starts:
            return "None"
        return ", ".join(f"{st:.2f}" for st in starts)

    # Internal helpers --------------------------------------------------
    def _audio_path(self, fname: str) -> Path:
        p_train = self.root / "training" / "training" / fname
        p_test = self.root / "test" / "test" / fname
        if p_train.exists():
            return p_train
        if p_test.exists():
            return p_test
        raise FileNotFoundError(f"{fname} not found")

    @staticmethod
    def _fig_to_bytes(fig: plt.Figure) -> bytes:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    def _prepare_output(self):
        row = self.df.iloc[self.idx]
        fname = row['Filename'] if 'Filename' in row else row.iloc[0]
        path = self._audio_path(str(fname))
        y, sr = librosa.load(path, sr=16000, mono=True)

        power = float(np.mean(y ** 2))
        t = np.arange(len(y)) / sr
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(t, y)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        fig.tight_layout()
        wav_png = self._fig_to_bytes(fig)

        y_norm = y
        if np.max(y) != np.min(y):
            y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.plot(t, y_norm)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Normalised")
        fig2.tight_layout()
        norm_png = self._fig_to_bytes(fig2)

        # store current normalised signal for EDA tab
        self.current_y_norm = y_norm
        self.current_sr = sr

        energy = float(np.sum(y_norm ** 2))
        self._log_energy(str(fname), energy)

        label = f"{fname} ({self.idx + 1}/{len(self.df)})"
        return label, wav_png, f"{power:.4f}", norm_png, f"{energy:.4f}", (sr, y)

    def _log_energy(self, fname: str, energy: float):
        first = not ENERGY_LOG.exists()
        with ENERGY_LOG.open('a', newline='') as f:
            writer = csv.writer(f)
            if first:
                writer.writerow(['Filename', 'Energy'])
            writer.writerow([fname, energy])


viewer = Viewer()

with gr.Blocks(title="Possum Audio Viewer") as demo:
    gr.Markdown("# Possum Audio Viewer")
    with gr.Tabs():
        with gr.Tab("Viewer"):
            data_root_in = gr.Textbox(label="Data root folder")
            annot_in = gr.Textbox(label="Annotation CSV path")
            load_btn = gr.Button("Load")

            out_fname = gr.Textbox(label="File", interactive=False)
            out_wave = gr.Image(label="Waveform")
            out_power = gr.Textbox(label="Power", interactive=False)
            out_norm_wave = gr.Image(label="Normalised Waveform")
            out_energy = gr.Textbox(label="Normalised Energy", interactive=False)
            out_audio = gr.Audio(label="Playback")

            prev_btn = gr.Button("◀ Prev")
            next_btn = gr.Button("Next ▶")

            load_btn.click(viewer.load_dataset,
                           inputs=[data_root_in, annot_in],
                           outputs=[out_fname, out_wave, out_power,
                                    out_norm_wave, out_energy, out_audio])

            next_btn.click(viewer.next_clip, outputs=[out_fname, out_wave, out_power,
                                                      out_norm_wave, out_energy, out_audio])
            prev_btn.click(viewer.prev_clip, outputs=[out_fname, out_wave, out_power,
                                                      out_norm_wave, out_energy, out_audio])

        with gr.Tab("EDA"):
            gr.Markdown("## Energy-based Call Detection")
            threshold_in = gr.Number(label="Energy threshold", value=1.0)
            detect_btn = gr.Button("Detect Calls")
            call_times = gr.Textbox(label="Call start times (s)", interactive=False)

            detect_btn.click(viewer.detect_calls, inputs=threshold_in, outputs=call_times)

if __name__ == "__main__":
    demo.launch()
