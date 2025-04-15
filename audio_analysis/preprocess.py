import pandas as pd
import numpy as np
import json
import re
import joblib


def GMM_extract_labels(data):
    features = []
    labels = []

    def add_feature(value, label):
        features.append(value)
        labels.append(label)

    def add_feature_list(value_list, base_label):
        for i, v in enumerate(value_list):
            features.append(v)
            labels.append(f"{base_label}_{i}")

    try:
        add_feature_list(data['lowlevel']['mfcc']['mean'], "mfcc")
    except KeyError:
        print("Missing: mfcc")

    try:
        add_feature_list(data['lowlevel']['gfcc']['mean'], "gfcc")
    except KeyError:
        print("Missing: gfcc")

    try:
        add_feature(data['lowlevel']['hfc']['mean'], "hfc")
    except KeyError:
        print("Missing: hfc")

    try:
        add_feature(data['tonal']['chords_changes_rate'], "chords_changes_rate")
    except KeyError:
        print("Missing: chords_changes_rate")

    try:
        scale = data['tonal'].get('key_scale', 'major')
        add_feature(1 if scale == 'minor' else 0, "key_scale")
    except KeyError:
        print("Missing: key_scale (defaulted to major)")
        add_feature(0, "key_scale")

    try:
        add_feature(data['lowlevel']['pitch_salience']['mean'], "pitch_salience")
    except KeyError:
        print("Missing: pitch_salience")

    try:
        add_feature(data['lowlevel']['dissonance']['mean'], "dissonance")
    except KeyError:
        print("Missing: dissonance")

    try:
        add_feature(data['rhythm']['bpm'], "bpm")
    except KeyError:
        print("Missing: bpm")

    try:
        add_feature(data['rhythm']['onset_rate'], "onset_rate")
    except KeyError:
        print("Missing: onset_rate")

    try:
        add_feature(data['lowlevel']['spectral_centroid']['mean'], "spectral_centroid")
    except KeyError:
        print("Missing: spectral_centroid")

    try:
        add_feature(data['lowlevel']['spectral_complexity']['mean'], "spectral_complexity")
    except KeyError:
        print("Missing: spectral_complexity")

    try:
        add_feature(data['lowlevel']['spectral_rolloff']['mean'], "spectral_rolloff")
    except KeyError:
        print("Missing: spectral_rolloff")

    try:
        add_feature(data['lowlevel']['spectral_flux']['mean'], "spectral_flux")
    except KeyError:
        print("Missing: spectral_flux")

    try:
        add_feature(data['lowlevel']['zerocrossingrate']['mean'], "zerocrossingrate")
    except KeyError:
        print("Missing: zerocrossingrate")

    try:
        add_feature_list(data['lowlevel']['spectral_contrast_coeffs']['mean'], "spectral_contrast")
    except KeyError:
        print("Missing: spectral_contrast_coeffs")

    try:
        add_feature(data['lowlevel']['average_loudness'], "average_loudness")
    except KeyError:
        print("Missing: average_loudness")

    try:
        add_feature(data['lowlevel']['dynamic_complexity'], "dynamic_complexity")
    except KeyError:
        print("Missing: dynamic_complexity")

    try:
        add_feature(data['rhythm']['beats_loudness']['mean'], "beats_loudness")
    except KeyError:
        print("Missing: beats_loudness")

    try:
        add_feature(data['lowlevel']['spectral_energyband_low']['mean'], "spectral_energyband_low")
    except KeyError:
        print("Missing: spectral_energyband_low")

    try:
        add_feature(data['lowlevel']['spectral_energyband_high']['mean'], "spectral_energyband_high")
    except KeyError:
        print("Missing: spectral_energyband_high")

    try:
        add_feature(data['tonal']['hpcp_entropy']['mean'], "hpcp_entropy")
    except KeyError:
        print("Missing: hpcp_entropy")

    try:
        add_feature(data['tonal']['key_strength'], "key_strength")
    except KeyError:
        print("Missing: key_strength")

    try:
        add_feature(data['lowlevel']['spectral_entropy']['mean'], "spectral_entropy")
    except KeyError:
        print("Missing: spectral_entropy")

    try:
        add_feature(data['lowlevel']['spectral_strongpeak']['mean'], "spectral_strongpeak")
    except KeyError:
        print("Missing: spectral_strongpeak")

    return features, labels

def build_feature_labels(data_sample):
    labels = []

    labels += [f"mfcc_{i}" for i in range(len(data_sample['lowlevel']['mfcc']['mean']))]
    labels += [f"gfcc_{i}" for i in range(len(data_sample['lowlevel']['gfcc']['mean']))]
    labels += ["hfc"]
    labels += ["chords_changes_rate"]
    
    labels += ["key_scale"]
    labels += ["pitch_salience"]
    labels += ["dissonance"]
    labels += ["bpm", "onset_rate"]
    labels += ["spectral_centroid", "spectral_complexity", "spectral_rolloff", "spectral_flux", "zerocrossingrate"]
    labels += [f"spectral_contrast_{i}" for i in range(len(data_sample['lowlevel']['spectral_contrast_coeffs']['mean']))]
    labels += ["average_loudness", "dynamic_complexity"]

    labels += ["beats_loudness"]
    labels += ["spectral_energyband_low", "spectral_energyband_high"]
    labels += ["hpcp_entropy", "key_strength"]
    labels += ["spectral_entropy", "spectral_strongpeak"]

    return labels


def GMM(file_path):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        features, labels = GMM_extract_labels(data)
        if features is None:
            return None

        df = pd.DataFrame([features], columns=labels)
        return df
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        return None
    

def get_value(d, keys):
    try:
        for key in keys:
            d = d[key]
        return d
    except (KeyError, TypeError):
        return np.nan

    
def SVM(full_path):
    with open('../models/SVMdata/idsGenre.txt', 'r') as f:
        loaded_dict = json.load(f)

    try:
        with open(full_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {full_path}: {e}")
        return None

    entry = {}

    recording_id = get_value(data, ['metadata', 'tags', 'musicbrainz_recordingid'])
    entry['id'] = recording_id[0] if isinstance(recording_id, list) and recording_id else np.nan

    # --- RHYTHM ---
    entry["bpm"] = get_value(data, ["rhythm", "bpm"])
    entry["beats_count"] = get_value(data, ["rhythm", "beats_count"])
    entry["danceability"] = get_value(data, ["rhythm", "danceability"])
    entry["onset_rate"] = get_value(data, ["rhythm", "onset_rate"])
    entry["bpm_first_peak"] = get_value(data, ["rhythm", "bpm_histogram_first_peak_bpm", "mean"])
    entry["bpm_second_peak"] = get_value(data, ["rhythm", "bpm_histogram_second_peak_bpm", "mean"])
    entry["bpm_first_peak_weight"] = get_value(data, ["rhythm", "bpm_histogram_first_peak_weight", "mean"])

    # --- TONAL / HARMONY ---
    entry["key_key"] = get_value(data, ["tonal", "key_key"])
    entry["key_scale"] = get_value(data, ["tonal", "key_scale"])
    entry["key_strength"] = get_value(data, ["tonal", "key_strength"])
    entry["chords_key"] = get_value(data, ["tonal", "chords_key"])
    entry["chords_scale"] = get_value(data, ["tonal", "chords_scale"])
    entry["chords_strength"] = get_value(data, ["tonal", "chords_strength", "mean"])
    entry["chords_changes_rate"] = get_value(data, ["tonal", "chords_changes_rate"])
    entry["hpcp_entropy"] = get_value(data, ["tonal", "hpcp_entropy", "mean"])

    # HPCP mean and var
    hpcp = get_value(data, ["tonal", "hpcp"])
    if isinstance(hpcp, dict):
        mean_vals = hpcp.get("mean", [])
        var_vals = hpcp.get("var", [])
        entry["hpcp_mean"] = np.mean(mean_vals) if mean_vals else np.nan
        entry["hpcp_var"] = np.mean(var_vals) if var_vals else np.nan

    # --- LOWLEVEL / TIMBRE ---
    for feature in ["mfcc", "gfcc"]:
        coeffs = get_value(data, ["lowlevel", feature])
        if isinstance(coeffs, dict):
            mean_vals = coeffs.get("mean", [])
            var_vals = coeffs.get("var", [])
            entry[f"{feature}_mean"] = np.mean(mean_vals) if mean_vals else np.nan
            entry[f"{feature}_var"] = np.mean(var_vals) if var_vals else np.nan

    spectral_features = [
        "spectral_centroid", "spectral_rolloff", "spectral_flux",
        "spectral_entropy", "zerocrossingrate"
    ]
    for feat in spectral_features:
        entry[feat] = get_value(data, ["lowlevel", feat, "mean"])

    entry['dynamic_complexity'] = get_value(data, ["lowlevel", "dynamic_complexity"])
    entry["average_loudness"] = get_value(data, ["lowlevel", "average_loudness"])
    entry["tuning_frequency"] = get_value(data, ["tonal", "tuning_frequency"])
    entry["tuning_equal_tempered_deviation"] = get_value(data, ["tonal", "tuning_equal_tempered_deviation"])

    # Genre assignment
    if entry['id'] in loaded_dict:
        entry['genre'] = loaded_dict[entry['id']]
    else:
        g = get_value(data, ['metadata', 'tags', 'genre'])
        entry['genre'] = ', '.join(g) if isinstance(g, list) else ""

    df =  pd.DataFrame([entry])

    # Load the saved OneHotEncoder
    encoder_path = "../models/SVMdata/onehot_encoder.pkl"
    encoder = joblib.load(encoder_path)


    # Apply encoder to expected columns
    categorical_cols = ['key_key', 'key_scale', 'chords_key', 'chords_scale']
    encoded = encoder.transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))

    # Drop the object columns (if you did that)
    dropCols = ['genre', 'is_rnb', 'key_key', 'key_scale', 'chords_key', 'chords_scale']
    result = encoded_df.drop(columns=dropCols)

    return result


if __name__ == "__main__":
    json_path = "features.json" 
    result_df = SVM(json_path)