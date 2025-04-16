import os
import json
import pandas as pd

root_folder = "data"  
data = []
for root, dirs, files in os.walk(root_folder):
    for file in files:
        if file.endswith(".json"):
            path = os.path.join(root, file)
            with open(path, 'r') as f:
                js = json.load(f)

            # Pull the features you want
            features = {
                'filename': file,
                'bpm': js.get('rhythm', {}).get('bpm', None),
                'danceability': js.get('rhythm', {}).get('danceability', None),
                'onset_rate': js.get('rhythm', {}).get('onset_rate', None),
                'average_loudness': js.get('lowlevel', {}).get('average_loudness', None),
                'spectral_centroid': js.get('lowlevel', {}).get('spectral_centroid', {}).get('mean', None),
                'chords_strength': js.get('tonal', {}).get('chords_strength', {}).get('mean', None),
                'key': js.get('tonal', {}).get('key_key', None),
                'scale': js.get('tonal', {}).get('key_scale', None),
                'genre': js.get('metadata', {}).get('tags', {}).get('genre', ['unknown'])[0],
            }

            if features['genre'] != 'unknown':
                data.append(features)

df = pd.DataFrame(data)
df.to_csv('rnb_features.csv', index=False)