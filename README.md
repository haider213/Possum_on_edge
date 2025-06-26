# Possum Audio Exploration Tools

This repository provides a small set of utilities and Gradio apps for exploring the Invasive Species dataset.  The main entry point is `viewer_app.py`, which lets you browse annotated possum clips, examine window energies and optionally save one‑second segments above a chosen threshold.

## Setup
1. Create a Python 3.10+ environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Obtain the dataset.  You can download it via Kaggle using:
   ```bash
   python data_downloader.py
   ```
   By default the files will be placed under `./data/invasive-species` with `training/` and `test/` subdirectories.

## Running the viewer
Launch the Gradio interface with:
```bash
python viewer_app.py
```
The UI will prompt for:
- **Data root folder** – the directory containing the `training/` and `test/` folders.
- **Annotation CSV path** – path to `annotations.csv` (from the dataset).

Once loaded you can page through clips, compute window energies, and clip high‑energy segments via the provided tabs. Results of the normalised window energies are logged to `energy_log.csv`.

## Additional app
`app.py` contains a minimal random‑clip explorer that uses `vis_helper.py` for visualisation.  You can run it the same way:
```bash
python app.py
```
It will display a waveform, mel‑spectrogram and rolling energy animation for a random clip each time you press the button.
