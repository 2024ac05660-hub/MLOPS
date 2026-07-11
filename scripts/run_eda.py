"""
Run EDA and generate all screenshots (headless, no display needed).
Used in CI and as a standalone script.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from data_processing import load_raw_data, clean_data, save_processed

plt.rcParams.update({"figure.dpi": 120, "font.size": 11})
sns.set_theme(style="whitegrid", palette="muted")

BASE = os.path.join(os.path.dirname(__file__), "..")
RAW_PATH = os.path.join(BASE, "data", "raw", "processed.cleveland.data")
OUT = os.path.join(BASE, "screenshots")
os.makedirs(OUT, exist_ok=True)

LABEL_NO_DISEASE = "No Disease"
LABEL_DISEASE = "Disease"
COLOR_NO_DISEASE = "#2ecc71"
COLOR_DISEASE = "#e74c3c"
CLASS_PALETTE = {LABEL_NO_DISEASE: COLOR_NO_DISEASE, LABEL_DISEASE: COLOR_DISEASE}


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close()
    print(f"  saved {name}")


df_raw = load_raw_data(RAW_PATH)
df = clean_data(df_raw)
save_processed(df, os.path.join(BASE, "data", "processed", "heart_disease_clean.csv"))

# 1. Missing values
missing_pct = (df_raw.isnull().sum() / len(df_raw)) * 100
fig, ax = plt.subplots(figsize=(8, 3))
missing_pct[missing_pct > 0].plot(kind="bar", ax=ax, color=COLOR_DISEASE)
ax.set_title("Missing Value Percentage per Feature", fontweight="bold")
ax.set_ylabel("% Missing")
save("eda_missing_values.png")

# 2. Class balance
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
counts = df["target"].value_counts()
bar_labels = [f"{LABEL_NO_DISEASE} (0)", f"{LABEL_DISEASE} (1)"]
colors = [COLOR_NO_DISEASE, COLOR_DISEASE]
axes[0].bar(bar_labels, counts.values, color=colors, edgecolor="white", linewidth=1.5)
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 2, str(v), ha="center", fontweight="bold")
axes[0].set_title("Class Distribution (Count)", fontweight="bold")
axes[0].set_ylabel("Count")
axes[1].pie(counts.values, labels=bar_labels, colors=colors, autopct="%1.1f%%",
            startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[1].set_title("Class Distribution (Proportion)", fontweight="bold")
save("eda_class_balance.png")

# 3. Continuous feature distributions
continuous = ["age", "trestbps", "chol", "thalach", "oldpeak"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, col in enumerate(continuous):
    for label, color in [(0, COLOR_NO_DISEASE), (1, COLOR_DISEASE)]:
        axes[i].hist(df[df["target"] == label][col], bins=20, alpha=0.6,
                     color=color, label=f"Target={label}", edgecolor="white")
    axes[i].set_title(col, fontweight="bold")
    axes[i].legend(fontsize=9)
axes[-1].axis("off")
fig.suptitle("Continuous Feature Distributions by Target Class", fontsize=14, fontweight="bold")
save("eda_continuous_distributions.png")

# 4. Categorical features
categorical = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
for i, col in enumerate(categorical):
    ct = df.groupby([col, "target"]).size().unstack(fill_value=0)
    ct.plot(kind="bar", ax=axes[i], color=[COLOR_NO_DISEASE, COLOR_DISEASE], edgecolor="white")
    axes[i].set_title(col, fontweight="bold")
    axes[i].tick_params(axis="x", rotation=0)
    axes[i].legend([LABEL_NO_DISEASE, LABEL_DISEASE], fontsize=8)
fig.suptitle("Categorical Feature Counts by Target Class", fontsize=14, fontweight="bold")
save("eda_categorical_features.png")

# 5. Correlation heatmap
fig, ax = plt.subplots(figsize=(12, 9))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, square=True, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=15)
save("eda_correlation_heatmap.png")

# 6. Boxplots
fig, axes = plt.subplots(1, 5, figsize=(18, 5))
for i, col in enumerate(continuous):
    df.boxplot(column=col, by="target", ax=axes[i],
               boxprops=dict(color="#2c3e50"),
               medianprops=dict(color=COLOR_DISEASE, linewidth=2))
    axes[i].set_title(col, fontweight="bold")
    axes[i].set_xlabel(f"Target (0={LABEL_NO_DISEASE}, 1={LABEL_DISEASE})")
plt.suptitle("Continuous Features by Target Class", fontsize=13, fontweight="bold")
save("eda_boxplots.png")

# 7. Feature relationship pairplot (continuous features coloured by target)
pair_cols = continuous + ["target"]
pair_df = df[pair_cols].copy()
pair_df["target"] = pair_df["target"].map({0: LABEL_NO_DISEASE, 1: LABEL_DISEASE})
pair_grid = sns.pairplot(pair_df, hue="target", palette=CLASS_PALETTE,
                         diag_kind="kde", plot_kws={"alpha": 0.6, "s": 25})
pair_grid.fig.suptitle("Feature Relationship Analysis (Continuous Features by Target)",
                        y=1.01, fontsize=13, fontweight="bold")
pair_grid.fig.savefig(os.path.join(OUT, "eda_feature_relationships.png"), bbox_inches="tight")
plt.close("all")
print("  saved eda_feature_relationships.png")

print("\nAll EDA screenshots saved to screenshots/")
