"""
main.py
Credit Risk Project — Full pipeline runner.
Run this to reproduce all results end-to-end.
"""

import sys
sys.path.insert(0, "src")

from preprocess import load_data, preprocess
from train import train_models, add_cluster_features, evaluate, save_models
from visualize import (
    plot_class_distribution, plot_roc_curves, plot_confusion_matrices,
    plot_feature_importance, plot_shap, plot_pca_clusters, plot_metrics_comparison
)

print("=" * 55)
print("  Credit Risk Default Prediction — Full Pipeline")
print("=" * 55)

# 1. Load & preprocess
df = load_data()
X_train, X_test, y_train, y_test, feature_names, scaler = preprocess(df)

# 2. Hybrid pipeline: add cluster features
X_train_c, X_test_c, km = add_cluster_features(X_train, X_test)

# 3. Train
models = train_models(X_train_c, y_train)

# 4. Evaluate
results = evaluate(models, X_test_c, y_test)

# 5. Save models
save_models(models, path="models/")

# 6. Figures
print("[5/5] Generating figures...")
plot_class_distribution(y_train)
plot_roc_curves(results, y_test)
plot_confusion_matrices(results, y_test)
plot_feature_importance(models["Random Forest"], X_train_c.columns.tolist())
plot_shap(models["XGBoost"], X_test_c)
plot_pca_clusters(X_train_c, X_train_c["cluster"].values)
plot_metrics_comparison(results)

print("=" * 55)
print("  Pipeline complete. Results:")
for name, res in results.items():
    print(f"  {name:25s} AUC={res['auc_roc']:.3f}  F1={res['f1']:.3f}  Acc={res['accuracy']:.3f}")
print("=" * 55)
print("  Figures saved to: figures/")
print("  Models saved to:  models/")
print("=" * 55)
