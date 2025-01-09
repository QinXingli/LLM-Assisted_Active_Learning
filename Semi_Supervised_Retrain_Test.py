import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from model_train import (
    load_and_merge_features,
    prepare_features,
    train_and_save_model
)


def split_target_samples(target_df, selected_sample_ids):
    """
    Split target year samples into augmentation set and validation set

    Parameters:
    -----------
    target_df : DataFrame
        All samples from target year
    selected_sample_ids : list
        List of selected sample IDs

    Returns:
    --------
    augment_samples : DataFrame
        Augmentation sample set
    validation_samples : DataFrame
        Validation sample set
    """
    augment_samples = target_df[target_df['sample_id'].isin(selected_sample_ids)]
    validation_samples = target_df[~target_df['sample_id'].isin(selected_sample_ids)]

    print(f"Augment samples: {len(augment_samples)}")
    print(f"Validation samples: {len(validation_samples)}")

    return augment_samples, validation_samples


def get_pseudo_labels(model, scaler, label_encoder, validation_data, confidence_threshold=0.9):
    """
    Get pseudo-labeled samples with high confidence
    """
    # Prepare features
    X_val = validation_data.drop(['sample_id', 'class'], axis=1)
    X_val_scaled = scaler.transform(X_val)

    # Predict
    y_pred = model.predict(X_val_scaled)
    y_pred_proba = model.predict_proba(X_val_scaled)
    max_probabilities = np.max(y_pred_proba, axis=1)

    # Transform predicted labels to original classes
    y_pred = label_encoder.inverse_transform(y_pred)

    # Select high confidence samples
    high_confidence_mask = max_probabilities >= confidence_threshold
    pseudo_labeled_samples = validation_data[high_confidence_mask].copy()
    pseudo_labeled_samples['class'] = y_pred[high_confidence_mask]

    print(f"Found {len(pseudo_labeled_samples)} high confidence samples")
    return pseudo_labeled_samples


def semi_supervised_training(train_data_dir, target_data_dir, selected_sample_ids,
                           model_dir, confidence_threshold=0.9, model_type='catboost'):
    """
    Execute semi-supervised learning process

    Parameters:
    -----------
    train_data_dir : str
        Original training data directory
    target_data_dir : str
        Target year data directory
    selected_sample_ids : list
        List of selected sample IDs
    model_dir : str
        Model save directory
    confidence_threshold : float
        Confidence threshold
    model_type : str
        Model type
    """
    # 1. Load data
    print("\nLoading training data...")
    train_features, train_samples = load_and_merge_features(train_data_dir)
    train_df = prepare_features(train_features, train_samples)

    print("\nLoading target year data...")
    target_features, target_samples = load_and_merge_features(target_data_dir)
    target_df = prepare_features(target_features, target_samples)

    # 2. Split target year samples
    augment_samples, validation_samples = split_target_samples(target_df, selected_sample_ids)

    # 3. Train initial model
    print("\nTraining initial model...")
    X_train = train_df.drop(['sample_id', 'class'], axis=1)
    y_train = train_df['class']

    initial_model, initial_scaler, label_encoder, _ = train_and_save_model(
        X_train, y_train,
        model_dir=os.path.join(model_dir, 'initial'),
        model_type=model_type
    )

    # 4. Initial prediction on validation data
    print("\nInitial prediction on validation data...")
    X_val = validation_samples.drop(['sample_id', 'class'], axis=1)
    y_val = validation_samples['class']
    X_val_scaled = initial_scaler.transform(X_val)

    initial_predictions = initial_model.predict(X_val_scaled)
    initial_predictions = label_encoder.inverse_transform(initial_predictions)
    initial_accuracy = accuracy_score(y_val, initial_predictions)
    print(f"\nInitial model accuracy: {initial_accuracy:.4f}")
    print("\nInitial classification report:")
    print(classification_report(y_val, initial_predictions))

    # 5. Get pseudo-labeled samples
    pseudo_labeled_samples = get_pseudo_labels(
        initial_model,
        initial_scaler,
        label_encoder,
        validation_samples,
        confidence_threshold
    )

    # 6. Combine new training set
    combined_train_df = pd.concat([
        augment_samples,
        pseudo_labeled_samples
    ], ignore_index=True)

    print(f"\nNew training set size: {len(combined_train_df)}")

    # 7. Train new model
    print("\nTraining new model with pseudo-labels...")
    X_new_train = combined_train_df.drop(['sample_id', 'class'], axis=1)
    y_new_train = combined_train_df['class']

    new_model, new_scaler, new_label_encoder, _ = train_and_save_model(
        X_new_train, y_new_train,
        model_dir=os.path.join(model_dir, 'semi_supervised'),
        model_type=model_type
    )

    # 8. Predict with new model
    print("\nPredicting with new model...")
    X_val_scaled_new = new_scaler.transform(X_val)
    new_predictions = new_model.predict(X_val_scaled_new)
    new_predictions = new_label_encoder.inverse_transform(new_predictions)
    new_accuracy = accuracy_score(y_val, new_predictions)

    print(f"\nNew model accuracy: {new_accuracy:.4f}")
    print("\nNew model classification report:")
    print(classification_report(y_val, new_predictions))

    return {
        'initial_accuracy': initial_accuracy,
        'new_accuracy': new_accuracy,
        'pseudo_labeled_count': len(pseudo_labeled_samples),
        'initial_model': initial_model,
        'new_model': new_model
    }


def main():
    # Set paths
    root_data_path = '/Users/XXX'
    train_data_dir = os.path.join(root_data_path, 'integrated_data_standard')
    target_data_dir = os.path.join(root_data_path, 'integrated_data_target')
    model_dir = os.path.join(root_data_path, 'models')
    model_type_list = ['xgboost', 'randomforest']
    model_type = model_type_list[0]
    confidence_threshold = 0.95

    # Sample numbers recognized by LLM
    selected_sample_ids = [

    ]

    # Execute semi-supervised learning
    results = semi_supervised_training(
        train_data_dir=train_data_dir,
        target_data_dir=target_data_dir,
        selected_sample_ids=selected_sample_ids,
        model_dir=model_dir,
        confidence_threshold=confidence_threshold,
        model_type=model_type
    )

    # Print results comparison
    print("\nResults Summary:")
    print(f"Initial model accuracy: {results['initial_accuracy']:.4f}")
    print(f"New model accuracy: {results['new_accuracy']:.4f}")
    print(f"Improvement: {(results['new_accuracy'] - results['initial_accuracy']) * 100:.2f}%")
    print(f"Number of pseudo-labeled samples used: {results['pseudo_labeled_count']}")


if __name__ == "__main__":
    main()
