# Cross-Year Crop Classification with LLM-Assisted Active Learning

This repository contains the implementation of a novel approach for cross-year crop classification using Large Language Models (LLMs) combined with traditional remote sensing techniques, as described in our paper "Towards Cross-Year Knowledge Transfer for Crop Classification: A Perspective from LLM-Assisted Active Learning".

## Overview

Our framework addresses the challenge of maintaining classification accuracy across different growing seasons while reducing the need for extensive manual labeling. The method combines:

- Traditional remote sensing feature extraction
- LLM-based temporal pattern recognition
- Semi-supervised learning techniques

## Key Features

- High-confidence sample selection using temporal consistency checking
- LLM-assisted crop type identification through temporal pattern analysis
- Multi-sensor feature integration (Sentinel-1 SAR and Sentinel-2 optical data)
- Support for multiple machine learning classifiers (XGBoost, Random Forest, etc.)
- Triple-voting mechanism for reliable LLM-based pattern recognition

## Requirements

```python
pandas>=1.2.0
numpy>=1.19.0
scikit-learn>=0.24.0
xgboost>=1.4.0
lightgbm>=3.2.0
catboost>=0.24
```

## Installation

```bash
git clone https://github.com/yourusername/crop-classification-llm
cd crop-classification-llm
pip install -r requirements.txt
```

## Usage

1. Data Preparation:
```python
# Prepare your data directory structure
root_data_path = 'path/to/your/data'
train_data_dir = os.path.join(root_data_path, 'integrated_data_standard')
target_data_dir = os.path.join(root_data_path, 'integrated_data_target')
```

2. Run Semi-supervised Training:
```python
from semi_supervised import semi_supervised_training

results = semi_supervised_training(
    train_data_dir=train_data_dir,
    target_data_dir=target_data_dir,
    selected_sample_ids=selected_sample_ids,
    model_dir=model_dir,
    confidence_threshold=0.95,
    model_type='xgboost'
)
```

## Results

Our method achieves significant improvements in classification performance:
- XGBoost: 11.7% F1-score improvement
- Random Forest: 13.0% F1-score improvement
