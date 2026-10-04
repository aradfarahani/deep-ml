import numpy as np

def bagging_classifier(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, n_estimators: int = 10, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    n_samples = X_train.shape[0]
    
    all_predictions = []
    
    for _ in range(n_estimators):
        # Bootstrap sampling with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X_train[indices]
        y_boot = y_train[indices]
        
        # Train a decision stump
        best_feature, best_threshold, best_polarity = _train_stump(X_boot, y_boot)
        
        # Make predictions with this stump
        if best_polarity == 1:
            predictions = (X_test[:, best_feature] > best_threshold).astype(int)
        else:
            predictions = (X_test[:, best_feature] <= best_threshold).astype(int)
        
        all_predictions.append(predictions)
    
    # Majority voting
    all_predictions = np.array(all_predictions)
    final_predictions = (np.mean(all_predictions, axis=0) >= 0.5).astype(int)
    return final_predictions


def _train_stump(X, y):
    n_samples, n_features = X.shape
    best_error = float('inf')
    best_feature = 0
    best_threshold = 0
    best_polarity = 1
    
    for feature_idx in range(n_features):
        feature_values = X[:, feature_idx]
        thresholds = np.unique(feature_values)
        
        for threshold in thresholds:
            for polarity in [1, -1]:
                if polarity == 1:
                    predictions = (feature_values > threshold).astype(int)
                else:
                    predictions = (feature_values <= threshold).astype(int)
                
                error = np.sum(predictions != y)
                if error < best_error:
                    best_error = error
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_polarity = polarity
    
    return best_feature, best_threshold, best_polarity