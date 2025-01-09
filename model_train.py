from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def load_and_merge_features(data_dir):
    """Load and merge all feature files"""
    features = ['ndvi', 'rvi', 'evi', 'rdvi', 'dpsvi', 'rfdi', 'vsi']
    all_features = {}

    for feature in features:
        file_path = os.path.join(data_dir, f'{feature}_timeseries.csv')
        df = pd.read_csv(file_path)
        all_features[feature] = df

    # Ensure sample_id and class consistency across all feature files
    base_samples = all_features[features[0]][['sample_id', 'class']]
    return all_features, base_samples


def extract_time_series_features(df, feature_name):
    """Extract statistical features for each time series"""
    # Get time series data (excluding sample_id and class columns)
    time_series = df.iloc[:, 2:]

    # Create feature dictionary
    features_dict = {}

    # Extract features for each sample
    for idx in range(len(df)):
        series = time_series.iloc[idx]
        # Exclude -9999 values
        valid_values = series[series != -9999]

        if len(valid_values) > 0:
            features_dict[f'{feature_name}_mean'] = valid_values.mean()
            features_dict[f'{feature_name}_std'] = valid_values.std()
            features_dict[f'{feature_name}_max'] = valid_values.max()
            features_dict[f'{feature_name}_min'] = valid_values.min()
            features_dict[f'{feature_name}_range'] = valid_values.max() - valid_values.min()
            features_dict[f'{feature_name}_median'] = valid_values.median()
            features_dict[f'{feature_name}_q25'] = valid_values.quantile(0.25)
            features_dict[f'{feature_name}_q75'] = valid_values.quantile(0.75)
            features_dict[f'{feature_name}_iqr'] = features_dict[f'{feature_name}_q75'] - features_dict[
                f'{feature_name}_q25']
            # Calculate ratio of valid values
            features_dict[f'{feature_name}_valid_ratio'] = len(valid_values) / len(series)
        else:
            # Fill with 0 if no valid values
            for stat in ['mean', 'std', 'max', 'min', 'range', 'median', 'q25', 'q75', 'iqr', 'valid_ratio']:
                features_dict[f'{feature_name}_{stat}'] = 0

    return features_dict


def prepare_features(all_features, base_samples):
    """Prepare feature matrix for training"""
    # Create feature DataFrame
    feature_rows = []

    for idx in range(len(base_samples)):
        row_features = {'sample_id': base_samples.iloc[idx]['sample_id'],
                       'class': base_samples.iloc[idx]['class']}

        # Extract statistics for each feature
        for feature_name, feature_df in all_features.items():
            sample_series = feature_df.iloc[idx:idx + 1]
            stats = extract_time_series_features(sample_series, feature_name)
            row_features.update(stats)

        feature_rows.append(row_features)

    return pd.DataFrame(feature_rows)


def train_and_save_model(X, y, model_dir, model_type='xgboost'):
    """
    Train and save model, supporting XGBoost and RandomForest, using grid search for optimal parameters

    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target variable
    model_dir : str
        Model save path
    model_type : str
        Model type selection, 'xgboost' or 'randomforest'
    """
    # Split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Set parameter grid based on model type
    if model_type.lower() == 'xgboost':
        base_model = XGBClassifier(
            objective='multi:softproba',
            random_state=42
        )
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.1, 0.3],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }

    elif model_type.lower() == 'randomforest':
        base_model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        )
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True],
            'class_weight': [None, 'balanced']
        }

    else:
        raise ValueError("model_type must be one of: 'xgboost', 'randomforest'")

    # Use grid search
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    # Train model
    print(f"\nTraining {model_type} model with grid search...")
    grid_search.fit(X_train_scaled, y_train_encoded)

    # Get best model
    best_model = grid_search.best_estimator_

    # Evaluate model
    train_score = best_model.score(X_train_scaled, y_train_encoded)
    test_score = best_model.score(X_test_scaled, y_test_encoded)

    # Print results
    print("\nBest parameters:", grid_search.best_params_)
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Testing accuracy: {test_score:.4f}")

    # Print class mapping
    print("\nClass mapping:")
    for original, encoded in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
        print(f"Original class {original} -> Encoded class {encoded}")

    # Save model related files
    os.makedirs(model_dir, exist_ok=True)
    model_filename = 'randomforest_model.pkl' if model_type.lower() == 'randomforest' else 'xgboost_model.pkl'

    joblib.dump(best_model, os.path.join(model_dir, model_filename))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(label_encoder, os.path.join(model_dir, 'label_encoder.pkl'))

    # Save training results summary
    results = {
        'best_parameters': grid_search.best_params_,
        'train_accuracy': train_score,
        'test_accuracy': test_score,
        'cv_results': pd.DataFrame(grid_search.cv_results_)
    }

    # Save training results to CSV
    results['cv_results'].to_csv(os.path.join(model_dir, 'grid_search_results.csv'))

    return best_model, scaler, label_encoder, results


def main():
    # Set paths
    root_data_path = '/Users/xxxx'
    data_dir = os.path.join(root_data_path, 'integrated_data_standard')
    model_dir = os.path.join(root_data_path, 'models')

    # Load data
    all_features, base_samples = load_and_merge_features(data_dir)

    # Prepare features
    feature_df = prepare_features(all_features, base_samples)

    # Split features and labels
    X = feature_df.drop(['sample_id', 'class'], axis=1)
    y = feature_df['class']

    # Print class distribution
    print("\nClass distribution in training data:")
    class_counts = y.value_counts()
    for class_label, count in class_counts.items():
        print(f"Class {class_label}: {count} samples ({count/len(y)*100:.2f}%)")
    print(f"Total samples: {len(y)}")

    # Train and save model
    model, scaler, label_encoder, results = train_and_save_model(
        X, y,
        model_dir=model_dir,
        model_type='randomforest'
    )

    # Print cross-validation results summary
    print("\nCross-validation results summary:")
    print(f"Mean CV score: {results['cv_results']['mean_test_score'].mean():.4f}")
    print(f"Std CV score: {results['cv_results']['std_test_score'].mean():.4f}")


if __name__ == "__main__":
    main()
