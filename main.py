from pathlib import Path
from data_downloader import list_files
from vis_helper import load_audio, plot_waveform, plot_spectrogram


# Example usage
if __name__ == "__main__":
    dataset_path = Path("./data/invasive-species").resolve()

    print("\nFirst few files:")
    for f in list_files(dataset_path)[:10]:
        print(" •", f)
    y, sr = load_audio(f)
    plot_waveform(y, sr)
    plot_spectrogram(y, sr)
