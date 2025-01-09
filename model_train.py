
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier  # 添加这行导入
from lightgbm import LGBMClassifier

def load_and_merge_features(data_dir):
    """读取所有特征文件并合并"""
    features = ['ndvi', 'rvi', 'evi', 'rdvi', 'dpsvi', 'rfdi', 'vsi']
    all_features = {}

    for feature in features:
        file_path = os.path.join(data_dir, f'{feature}_timeseries.csv')
        df = pd.read_csv(file_path)
        all_features[feature] = df

    # 确保所有特征文件的sample_id和class一致
    base_samples = all_features[features[0]][['sample_id', 'class']]
    return all_features, base_samples


def extract_time_series_features(df, feature_name):
    """为每个时间序列提取统计特征"""
    # 获取时间序列数据（排除sample_id和class列）
    time_series = df.iloc[:, 2:]

    # 创建特征字典
    features_dict = {}

    # 对每个样本进行特征提取
    for idx in range(len(df)):
        series = time_series.iloc[idx]
        # 排除-9999值
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
            # 计算有效值的比例
            features_dict[f'{feature_name}_valid_ratio'] = len(valid_values) / len(series)
        else:
            # 如果没有有效值，填充为0
            for stat in ['mean', 'std', 'max', 'min', 'range', 'median', 'q25', 'q75', 'iqr', 'valid_ratio']:
                features_dict[f'{feature_name}_{stat}'] = 0

    return features_dict


def prepare_features(all_features, base_samples):
    """准备用于训练的特征矩阵"""
    # 创建特征DataFrame
    feature_rows = []

    for idx in range(len(base_samples)):
        row_features = {'sample_id': base_samples.iloc[idx]['sample_id'],
                        'class': base_samples.iloc[idx]['class']}

        # 为每个特征提取统计量
        for feature_name, feature_df in all_features.items():
            sample_series = feature_df.iloc[idx:idx + 1]
            stats = extract_time_series_features(sample_series, feature_name)
            row_features.update(stats)

        feature_rows.append(row_features)

    return pd.DataFrame(feature_rows)





def train_and_save_model(X, y, model_dir, model_type='xgboost'):
    """
    训练模型并保存，支持XGBoost和CatBoost，使用网格搜索寻找最佳参数

    Parameters:
    -----------
    X : array-like
        特征矩阵
    y : array-like
        目标变量
    model_dir : str
        模型保存路径
    model_type : str
        选择模型类型，'xgboost' 或 'catboost'
    """
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 对标签进行编码
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # 根据选择的模型类型设置参数网格
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

    elif model_type.lower() == 'catboost':
        base_model = CatBoostClassifier(
            random_state=42,
            verbose=False,
            bootstrap_type='Bernoulli'  # 添加这个参数
        )

        param_grid = {
            'iterations': [100, 200, 300],
            'depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1, 0.3],
            'l2_leaf_reg': [1, 3, 5, 7],
            'subsample': [0.8, 0.9, 1.0],  # 需要 bootstrap_type='Bernoulli' 才能使用
            'bootstrap_type': ['Bernoulli']  # 可以尝试其他类型，但不是所有类型都支持 subsample
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
    elif model_type.lower() == 'lightgbm':
        base_model = LGBMClassifier(
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 63, 127],  # 2^depth - 1
            'min_child_samples': [20, 50, 100],
            'colsample_bytree': [0.8, 1.0],

        }

    else:
        raise ValueError("model_type must be one of: 'xgboost', 'catboost', 'randomforest', or 'lightgbm'")


    # 使用网格搜索
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    # 训练模型
    print(f"\nTraining {model_type} model with grid search...")
    grid_search.fit(X_train_scaled, y_train_encoded)

    # 获取最佳模型
    best_model = grid_search.best_estimator_

    # 评估模型
    train_score = best_model.score(X_train_scaled, y_train_encoded)
    test_score = best_model.score(X_test_scaled, y_test_encoded)

    # 打印结果
    print("\nBest parameters:", grid_search.best_params_)
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Testing accuracy: {test_score:.4f}")

    # 打印类别映射关系
    print("\nClass mapping:")
    for original, encoded in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
        print(f"Original class {original} -> Encoded class {encoded}")

    # 如果是CatBoost模型，保存特征名称
    if model_type.lower() == 'catboost':
        feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in
                                                                                range(X.shape[1])]
        best_model.set_feature_names(feature_names)

    # 保存模型相关文件
    os.makedirs(model_dir, exist_ok=True)
    model_filename = 'catboost_model.pkl' if model_type.lower() == 'catboost' else 'xgboost_model.pkl'

    joblib.dump(best_model, os.path.join(model_dir, model_filename))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(label_encoder, os.path.join(model_dir, 'label_encoder.pkl'))

    # 保存训练结果摘要
    results = {
        'best_parameters': grid_search.best_params_,
        'train_accuracy': train_score,
        'test_accuracy': test_score,
        'cv_results': pd.DataFrame(grid_search.cv_results_)
    }

    # 保存训练结果到CSV
    results['cv_results'].to_csv(os.path.join(model_dir, 'grid_search_results.csv'))

    return best_model, scaler, label_encoder, results



def main():
    # 设置路径
    root_data_path = '/Users/qinxl/Documents/GEE_Output/Experiment_HLJ'
    data_dir = os.path.join(root_data_path, 'integrated_data_standard')
    model_dir = os.path.join(root_data_path, 'models')

    # 加载数据
    all_features, base_samples = load_and_merge_features(data_dir)

    # 准备特征
    feature_df = prepare_features(all_features, base_samples)

    # 分离特征和标签
    X = feature_df.drop(['sample_id', 'class'], axis=1)
    y = feature_df['class']

    # 打印各类别的样本数量
    print("\nClass distribution in training data:")
    class_counts = y.value_counts()
    for class_label, count in class_counts.items():
        print(f"Class {class_label}: {count} samples ({count/len(y)*100:.2f}%)")
    print(f"Total samples: {len(y)}")

    # 训练和保存模型
    model, scaler, label_encoder, results = train_and_save_model(
        X, y,
        model_dir=model_dir,
        model_type='randomforest'
    )

    # 打印最佳模型的交叉验证结果摘要
    print("\nCross-validation results summary:")
    print(f"Mean CV score: {results['cv_results']['mean_test_score'].mean():.4f}")
    print(f"Std CV score: {results['cv_results']['std_test_score'].mean():.4f}")

if __name__ == "__main__":
    main()
