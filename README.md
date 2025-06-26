# Possum on Edge

This repository contains a collection of small utilities and Gradio apps for working with the *invasive-species* audio dataset.

## Requirements

A Python 3.8+ environment is recommended. Install dependencies using `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The apps depend on additional packages such as `gradio`, `pandas`, `numpy` and `matplotlib`. If they are missing simply install them manually:

```bash
pip install gradio pandas numpy matplotlib
```

## Downloading the dataset

Use `data_downloader.py` to pull the latest dataset via KaggleHub. Files will be copied into `./data/invasive-species` by default:

```bash
python -m data_downloader
```

## Running the applications

Two entry points are provided:

- `viewer_app.py` – a simple clip viewer with next/previous navigation.
- `app.py` – draws a random clip and shows waveform, spectrogram and a rolling energy animation.

Launch either one with Python to start a local Gradio server:

```bash
python viewer_app.py
# or
python app.py
```

By default the apps expect the dataset under `./data/invasive-species`. You can specify a custom data root and annotation CSV path through the UI once the app is running.

