import pandas as pd
import numpy as np
import os
import json

def extract_features_from_json(data):
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


def extract_single_json(file_path):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        features, labels = extract_features_from_json(data)
        if features is None:
            return None

        df = pd.DataFrame([features], columns=labels)
        return df
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        return None

def process_dataset(root_folder):
    all_features = []
    file_ids = []
    labels_initialized = False
    feature_labels = []

    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(subdir, file)
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)

                    features, labels = extract_features_from_json(data)
                    if features is None:
                        continue

                    if not labels_initialized:
                        feature_labels = labels
                        labels_initialized = True

                    all_features.append(features)
                    file_ids.append(file.replace('.json', ''))

                except Exception as e:
                    print(f"Failed on {file}: {e}")

    df = pd.DataFrame(all_features, columns=feature_labels)
    df['recordingmbid'] = file_ids
    return df