import joblib
import numpy as np
import pandas as pd
import preprocess
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)

def preprocess_gmm(df, scaler):
    expected_columns = scaler.feature_names_in_
    df = df[[col for col in expected_columns if col in df.columns]]

    for col in expected_columns:
        if col not in df.columns:
            df.loc[:, col] = 0

    df = df[expected_columns]
    scaled_features = scaler.transform(df)
    
    return scaled_features

def preprocess_svm(df, scaler):
    expected_columns = scaler.feature_names_in_
    df = df[[col for col in expected_columns if col in df.columns]]
    df = df.copy()

    for col in expected_columns:
        if col not in df.columns:
            df.loc[:, col] = 0

    df = df[expected_columns]
    df = df.fillna(0)  # Replace NaN values with 0
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
    scaled_features = preprocess_gmm(df, scaler)

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
    gmm_confience = np.mean(ll_rnb) - np.mean(ll_nonrnb)

    genre_prediction = "RnB" if np.mean(predictions) >= 0.5 else "Non-RnB"
    gl_confidence = float(np.max([np.mean(probabilities), 1 - np.mean(probabilities)]))

    if verbose:
        print(f"🎵 GMM Genre Prediction: {gmm_genre_prediction}")
        print(f"🔍 Confidence Score: {gmm_confience:.4f}\n")

        print(f"🎵 GMM + Logistic Predicted Genre: {genre_prediction}")
        print(f"🔍 Confidence Score: {gl_confidence:.4f}\n")

    return {
        "gmm_genre": gmm_genre_prediction,
        "logistic_genre": genre_prediction,
        "probabilities": probabilities.tolist(),
        "predictions": predictions.tolist(),
        "confidence": (gmm_confience, gl_confidence)
    }

def predict_svm(df, verbose=True):
    """
    Predicts genre using multiple pre-trained SVM models with different kernels.

    Args:
        df (pd.DataFrame): Feature dataframe.
        verbose (bool): Whether to print diagnostic output.

    Returns:
        dict: Dictionary with predictions, confidence scores, and genre results for each kernel.
    """
    model_paths = {
        "linear": "../models/SVMdata/linear_kernal_model.pkl",
        "poly": "../models/SVMdata/poly_kernal_model.pkl",
        "rbf": "../models/SVMdata/rbf_kernal_model.pkl",
        "rbf low gamma": "../models/SVMdata/rbf_modGamma_kernal_model.pkl",
        "sigmoid": "../models/SVMdata/sigmoid_kernal_model.pkl"
    }

    # Load scaler
    scaler = joblib.load("../models/SVMdata/scaler.pkl")
    scaled_features = preprocess_svm(df, scaler)

    results = {}

    for kernel, path in model_paths.items():
        try:
            model = joblib.load(path)
            prediction = model.predict(scaled_features)
            decision_scores = model.decision_function(scaled_features)

            genre_prediction = "RnB" if prediction[0] == 1 else "Non-RnB"
            confidence = float(np.max([decision_scores[0], -decision_scores[0]]))

            if verbose:
                print(f"🎧 SVM ({kernel}) Prediction: {genre_prediction}")
                print(f"📏 Distance from Decision Boundary: {confidence:.4f}\n")

            results[kernel] = {
                "svm_genre": genre_prediction,
                "svm_prediction": int(prediction[0]),
                "decision_score": float(decision_scores[0]),
                "confidence": confidence
            }
        except Exception as e:
            print(f"⚠️ Error with {kernel} model: {e}")
            results[kernel] = {"error": str(e)}

    return results

if __name__ == "__main__":
    df = preprocess.GMM("features.json")
    results = predict_gmm(df)