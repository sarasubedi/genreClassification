import joblib
import numpy as np
from collections import Counter


def prepare_features_for_model(df, scaler):
    expected_columns = scaler.feature_names_in_
    df = df[[col for col in expected_columns if col in df.columns]]

    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_columns]
    scaled_features = scaler.transform(df)
    
    return scaled_features

def predict_gmm(df):
    """
    Predicts whether input features represent RnB or Non-RnB using pre-trained GMM and logistic regression models.
    
    Args:
        df (pd.DataFrame): A dataframe containing feature vectors to classify.

    Returns:
        dict: A dictionary containing the probabilities, binary predictions, and the final genre.
    """
    # Load the pre-trained models and scaler

    gmm_rnb = joblib.load("../models/GMM_data/model_gmm_rnb.joblib")
    gmm_nonrnb = joblib.load("../models/GMM_data/model_gmm_nonrnb.joblib")
    clf = joblib.load("../models/GMM_data/model_logistic_rnb.joblib")
    scaler = joblib.load("../models/GMM_data/scaler.joblib")
    best_t = joblib.load("../models/GMM_data/optimal_threshold.joblib")

    # Scale the features using the new function
    scaled_features = prepare_features_for_model(df, scaler)

    # Compute GMM log-likelihood features
    ll_rnb = gmm_rnb.score_samples(scaled_features)
    ll_nonrnb = gmm_nonrnb.score_samples(scaled_features)

    log_diff = ll_rnb - ll_nonrnb
    log_ratio = np.log((np.exp(ll_rnb) + 1e-9) / (np.exp(ll_nonrnb) + 1e-9))

    X_fused = np.column_stack([ll_rnb, ll_nonrnb, log_diff, log_ratio])

    # Predict probabilities using the logistic regression model
    probabilities = clf.predict_proba(X_fused)[:, 1]

    # Apply the optimal threshold to classify as RnB or non-RnB
    predictions = (probabilities >= best_t).astype(int)

    # Determine the most common genre prediction
    genre_prediction = "RnB" if Counter(predictions).most_common(1)[0][0] == 1 else "Non-RnB"
    print(f"🎵 Predicted Genre: {genre_prediction}")

    confidence = max(np.mean(probabilities), 1 - np.mean(probabilities))
    print(f"🔍 Confidence Score: {confidence:.4f}")

    return {
        "probabilities": probabilities.tolist(),
        "predictions": predictions.tolist(),
        "genre": genre_prediction,
        "confidence": confidence
    }

# Optional test block
if __name__ == "__main__":
    import pandas as pd
    print("🔍 Running example with dummy input")
    dummy_df = pd.DataFrame([[0] * 53])  # Adjust column count as needed
    result = predict_gmm(dummy_df)
    print(result)