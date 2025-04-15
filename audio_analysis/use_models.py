import joblib
import numpy as np
import pandas as pd
import preprocess

def prepare_features_for_model(df, scaler):
    # Ensure the dataframe contains only the columns expected by the scaler
    expected_columns = scaler.feature_names_in_
    df = df[[col for col in expected_columns if col in df.columns]]

    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_columns]
    scaled_features = scaler.transform(df)
    
    return scaled_features

    

def predict_gmm(df, verbose=True):
    """
    Predicts whether input features represent RnB or Non-RnB using pre-trained GMM and logistic regression models.

    Args:
        df (pd.DataFrame): A dataframe containing feature vectors to classify.
        verbose (bool): Whether to print diagnostic output.

    Returns:
        dict: A dictionary containing genre predictions and model outputs.
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

    # Determine genre from GMM and logistic classifier
    gmm_genre_prediction = "RnB" if np.mean(ll_rnb) > np.mean(ll_nonrnb) else "Non-RnB"

    genre_prediction = "RnB" if np.mean(predictions) >= 0.5 else "Non-RnB"

    confidence = float(np.max([np.mean(probabilities), 1 - np.mean(probabilities)]))

    if verbose:
        print(f"🎵 GMM Genre Prediction: {gmm_genre_prediction}")
        print(f"🎵 GMM + Logistic Predicted Genre: {genre_prediction}")
        print(f"🔍 Confidence Score: {confidence:.4f}")

    return {
        "gmm_genre": gmm_genre_prediction,
        "logistic_genre": genre_prediction,
        "probabilities": probabilities.tolist(),
        "predictions": predictions.tolist(),
        "confidence": confidence
    }


def predict_svm(df):
    
    model_paths = {
        "linear": "../models/SVMdata/linear_kernal_model.pkl",
        "poly": "../models/SVMdata/poly_kernal_model.pkl",
        "rbf": "../models/SVMdata/rbf_kernal_model.pkl",
        "sigmoid": "../models/SVMdata/sigmoid_kernal_model.pkl"
    }

    # Load scaler
    scaler = joblib.load("../models/SVMdata/scaler.pkl")

    results = {}

    for kernel, path in model_paths.items():
        # Load the SVM model
        model = joblib.load(path)

        # Prepare features for the model
        scaled_features = prepare_features_for_model(df, scaler)

        # Make predictions
        predictions = model.predict(scaled_features)
        probabilities = model.predict_proba(scaled_features)[:, 1] if hasattr(model, "predict_proba") else None

        # Print the prediction
        print(f"Kernel: {kernel}")
        print(f"Predictions: {predictions}")
        if probabilities is not None:
            print(f"Probabilities: {probabilities}")

        # Store results
        results[kernel] = {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist() if probabilities is not None else None
        }

    return results


if __name__ == "__main__":
    df = preprocess.SVM("features.json")
    results = predict_svm(df)
   