import subprocess
import json
from collections import defaultdict
from essentia.standard import MusicExtractor
import essentia
essentia.log.infoActive = False
essentia.log.warningActive = False
essentia.log.errorActive = False
import sys
import IPython.display as ipd
from yt_dlp import YoutubeDL
import os
import contextlib
from sklearn.feature_extraction import DictVectorizer
import extract_json, use_models

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def download_mp3(url, output_name="song"):
    print("\n--- Downloading MP3 ---")
    print(f"🔄 Downloading MP3 from URL: {url}")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f"{output_name}.%(ext)s",
        'ffmpeg_location': '/opt/homebrew/bin/ffmpeg',  # Update this path if needed
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    with suppress_output():
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    print(f"✅ MP3 downloaded and saved as: {output_name}.mp3")

# Step 1: Convert MP3 to WAV
def convert_to_wav(input_file, output_file="song.wav"):
    print("\n--- Converting MP3 to WAV ---")
    print(f"🔄 Converting MP3 to WAV: {input_file} -> {output_file}")
    with suppress_output():
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", input_file, output_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    print(f"✅ Conversion complete: {output_file}")

# Step 2: Load classifier labels from file
def load_classifier_labels(filename="classifiers.txt"):
    print(f"🔄 Loading classifier labels from: {filename}")
    with open(filename, "r") as f:
        labels = set(line.strip() for line in f if line.strip())
    print(f"✅ Loaded {len(labels)} classifier labels")
    return labels

# Step 3: Filter extracted features by classifier keys
def filter_keys_by_classifier_list(pool, classifier_keys):
    print(f"🔄 Filtering features by classifier keys")
    def nested_dict():
        return defaultdict(nested_dict)

    root = nested_dict()
    for key in pool.descriptorNames():
        if any(key.startswith(label) for label in classifier_keys):
            parts = key.split('.')
            current = root
            for part in parts[:-1]:
                current = current[part]
            val = pool[key]
            current[parts[-1]] = val.tolist() if hasattr(val, 'tolist') else val

    print(f"✅ Features filtered")
    return json.loads(json.dumps(root))  # Convert nested defaultdict to regular dict

# Step 4: Extract features and save filtered output
def extract_filtered_features(mp3_path, classifiers_path="../classifiers.txt", output_json="features.json"):
    print("\n--- Extracting Audio Features ---")
    print(f"🔄 Starting feature extraction for: {mp3_path}")
    if not os.path.exists(mp3_path):
        raise FileNotFoundError(f"Input MP3 file does not exist: {mp3_path}")
    wav_path = "temp.wav"
    convert_to_wav(mp3_path, wav_path)

    print(f"🔄 Extracting features from WAV file: {wav_path}")
    with suppress_output():
        extractor = MusicExtractor()
        features, audio = extractor(wav_path)
    print(f"✅ Features extracted")

    print("\n--- Filtering Selected Features ---")
    print(f"🔄 Filtering features using classifier labels")
    classifier_keys = load_classifier_labels(classifiers_path)
    filtered = filter_keys_by_classifier_list(features, classifier_keys)

    print("\n--- Saving Output to JSON ---")
    print(f"🔄 Saving filtered features to JSON: {output_json}")
    with open(output_json, "w") as f:
        json.dump(filtered, f, indent=2)
    print(f"✅ Saved filtered features to: {output_json}")

# Step 5: analyze with model
def analyze_with_model(features_json):
    print("\n--- Analyzing Features with Model ---")
    print(f"🔄 Loading features from JSON: {features_json}")
    if not os.path.exists(features_json):
        raise FileNotFoundError(f"Features JSON file does not exist: {features_json}")

    df = extract_json.extract_single_json(features_json)


    print("\n--- Analyzing with GMM ---")
    use_models.predict_gmm(df)

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test.py <mp3_path>")
        print("Or:")
        print("  python test.py (to download a YouTube song and process it)")
        url = input("Enter the YouTube URL: ")
        output_name = "song"
        mp3_path = f"{output_name}.mp3"
        download_mp3(url, output_name)
    else:
        mp3_path = sys.argv[1]

    # Ensure the MP3 file exists before proceeding
    try:
        print(f"🔄 Processing MP3 file: {mp3_path}")
        extract_filtered_features(mp3_path) 
    
    except FileNotFoundError:
        print(f"❌ Error: File '{mp3_path}' not found.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")

    # Analyze the extracted features with the model
    features_json = "features.json"
    try:
        analyze_with_model(features_json)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ An error occurred during analysis: {e}")

    # Clean up temporary files
    try:
        if os.path.exists("temp.wav"):
            os.remove("temp.wav")
        if os.path.exists("features.json"):
            os.remove("features.json")
    except Exception as e:
        print(f"⚠️ Warning: Could not delete temporary files: {e}")
    

if __name__ == "__main__":
    main()
