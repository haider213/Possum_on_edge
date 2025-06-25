"""
app.py – random Invasive‑Species clip explorer
---------------------------------------------
* User supplies **data root** (folder holding `training/` + `test/`) and the
  **annotations CSV** path.
* Click **Random Clip** → waveform PNG, mel‑spectrogram PNG, rolling‑window‑
  energy animation, and audio playback.
* Relies on `vis_helper.py` (must be importable).
"""

import os
import random
import importlib
from pathlib import Path

import gradio as gr

import vis_helper as vh  # make sure PYTHONPATH picks up the local module


def load_random_clip(data_root: str, annot_csv: str):
    """Callback that picks a random WAV and returns assets for the UI."""
    # 1) Wire vis_helper to the chosen dataset location
    os.environ["INVASIVE_SPECIES_ROOT"] = data_root
    importlib.reload(vh)  # refresh ROOT_DIR/TRAIN_DIR/… constants

    # 2) Read annotations
    df = vh.load_annotations(Path(annot_csv))
    if df.empty:
        raise gr.Error("Annotation CSV is empty or unreadable.")

    fname = random.choice(df["Filename"].tolist())

    # 3) Generate visuals & audio
    y, sr = vh.load_audio(fname)
    wav_png = vh.fig_to_bytes(vh.plot_waveform(y, sr))
    spec_png = vh.fig_to_bytes(vh.plot_spectrogram(y, sr))
    energy_html = vh.rolling_energy_anim(y, sr)

    return fname, wav_png, spec_png, energy_html, (sr, y)


with gr.Blocks(title="Invasive‑Species Random Clip Explorer") as demo:
    gr.Markdown("""# 🐾 Random Clip Explorer
Choose your dataset location, then let the app draw a random 5‑second clip and
visualise it.
""")

    data_root_in = gr.Textbox(label="Data root folder", value="/mnt/d/GrokkingGrok/Bat_detectio/pythonProject1/data/invasive-species")
    annot_path_in = gr.Textbox(label="Annotation CSV path", value="/mnt/d/GrokkingGrok/Bat_detectio/pythonProject1/data/invasive-species")
    load_btn = gr.Button("Random Clip")

    out_fname = gr.Textbox(label="Selected file", interactive=False)
    out_wave = gr.Image(label="Waveform")
    out_spec = gr.Image(label="Mel‑Spectrogram")
    out_energy = gr.HTML(label="Rolling‑Window Energy")
    out_audio = gr.Audio(label="Playback")

    load_btn.click(load_random_clip,
                   inputs=[data_root_in, annot_path_in],
                   outputs=[out_fname, out_wave, out_spec, out_energy, out_audio])


if __name__ == "__main__":
    demo.launch()
