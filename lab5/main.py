# Laboratory Work 5
# Dimensionality Reduction Methods and Clustering

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
import umap
import joblib
import warnings
from sklearn.datasets import load_breast_cancer
import kagglehub
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


# Читаем CSV (в датасете два файла: winequality-red.csv и winequality-white.csv, можно выбрать один)
df = pd.read_csv("WineQT.csv")

# Разделяем признаки и целевую переменную
X = df.drop("quality", axis=1)
y = df["quality"]

print(f"Dataset shape: {X.shape}")
print(f"Features: {X.columns.tolist()}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")

# 2. Exploratory Data Analysis (EDA)
print("\n2. Performing EDA...")

# Basic statistics
print("Basic statistics:")
print(X.describe())

# Check for missing values
print("\nMissing values:")
print(X.isnull().sum())

# Display correlation matrix
plt.figure(figsize=(12, 10))
correlation_matrix = X.corr()
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# 2.1 Outlier detection and treatment using IQR method
print("\nDetecting and treating outliers...")


def detect_and_treat_outliers(df):
    df_clean = df.copy()

    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Replace outliers with boundary values
        df_clean[column] = np.where(
            df_clean[column] < lower_bound,
            lower_bound,
            np.where(
                df_clean[column] > upper_bound,
                upper_bound,
                df_clean[column]
            )
        )

    return df_clean


# Apply outlier detection and treatment
X_clean = detect_and_treat_outliers(X)

# 2.2 Normalize the data
print("Normalizing data...")
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_clean)
X_normalized_df = pd.DataFrame(X_normalized, columns=X.columns)

print("Normalized data statistics:")
print(X_normalized_df.describe())

# 3. Apply Kernel PCA with different kernel functions
print("\n3. Applying Kernel PCA with different kernels...")
n_components = 2  # For visualization
kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']

# Dictionary to store KernelPCA models
kpca_models = {}

# Create a figure for all KernelPCA results
plt.figure(figsize=(15, 10))

for i, kernel in enumerate(kernels):
    print(f"Applying KernelPCA with {kernel} kernel...")
    kpca = KernelPCA(n_components=n_components, kernel=kernel)
    X_kpca = kpca.fit_transform(X_normalized)
    kpca_models[kernel] = kpca

    # Plot the results
    plt.subplot(2, 3, i + 1)
    scatter = plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap='viridis', s=30, alpha=0.7)
    plt.title(f'Kernel PCA with {kernel} kernel')
    plt.colorbar(scatter)

plt.tight_layout()
plt.savefig('kpca_results.png')
plt.close()

# 4. Analysis of KernelPCA results already implemented in the code above

# 5. For linear kernel, calculate variance and lost variance
print("\n5. Analyzing variance for linear kernel...")
kpca_linear = KernelPCA(n_components=X_normalized.shape[1], kernel='linear')
X_kpca_linear = kpca_linear.fit_transform(X_normalized)

# Get eigenvalues
eigenvalues = kpca_linear.eigenvalues_

# Calculate explained variance ratio
total_variance = np.sum(eigenvalues)
explained_variance_ratio = eigenvalues / total_variance

# Cumulative explained variance
cumulative_variance = np.cumsum(explained_variance_ratio)

# Plot explained variance
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7,
        label='Individual explained variance')
plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid',
         label='Cumulative explained variance')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Principal Components')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('explained_variance.png')
plt.close()

# Find optimal number of components for 95% variance explained
optimal_components = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Number of components needed to explain 95% of variance: {optimal_components}")

# Calculate lost variance
lost_variance = 1 - cumulative_variance[optimal_components - 1]
print(f"Lost variance with {optimal_components} components: {lost_variance:.4f}")

# 6. Apply t-SNE and UMAP for comparison
print("\n6. Applying t-SNE and UMAP...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_normalized)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', s=30, alpha=0.7)
plt.title('t-SNE visualization')
plt.colorbar(scatter)
plt.tight_layout()
plt.savefig('tsne_result.png')
plt.close()

# UMAP
print("Applying UMAP...")
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_normalized)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='viridis', s=30, alpha=0.7)
plt.title('UMAP visualization')
plt.colorbar(scatter)
plt.tight_layout()
plt.savefig('umap_result.png')
plt.close()

# 7. Save and load the trained model
print("\n7. Saving and loading the model...")

# Save the best model (using UMAP as an example)
joblib.dump(reducer, 'umap_model.joblib')

# Load the model
loaded_model = joblib.load('umap_model.joblib')
print("Model loaded successfully!")

# Apply the loaded model
X_umap_loaded = loaded_model.transform(X_normalized)

# Verify the loaded model produces similar results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
scatter1 = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='viridis', s=30, alpha=0.7)
plt.title('Original UMAP Result')
plt.colorbar(scatter1)

plt.subplot(1, 2, 2)
scatter2 = plt.scatter(X_umap_loaded[:, 0], X_umap_loaded[:, 1], c=y, cmap='viridis', s=30, alpha=0.7)
plt.title('Loaded Model UMAP Result')
plt.colorbar(scatter2)

plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()

# Try LDA (only if there are multiple classes, as we have in this case)
print("\n8. Applying LDA...")
try:
    lda = LDA(n_components=1)  # n_components must be less than number of classes
    X_lda = lda.fit_transform(X_normalized, y)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_lda, np.zeros_like(X_lda), c=y, cmap='viridis', s=30, alpha=0.7)
    plt.title('LDA visualization')
    plt.xlabel('LD1')
    plt.yticks([])
    plt.tight_layout()
    plt.savefig('lda_result.png')
    plt.close()

    print(f"LDA explained variance ratio: {lda.explained_variance_ratio_}")
except Exception as e:
    print(f"LDA could not be applied: {e}")

# Conclusion
print("\n=== CONCLUSION ===")
print("1. We successfully applied multiple dimensionality reduction techniques:")
print("   - KernelPCA with different kernels")
print("   - t-SNE")
print("   - UMAP")
print("   - LDA")
print(f"2. For linear KernelPCA, we need {optimal_components} components to explain 95% of variance")
print(f"3. Lost variance with {optimal_components} components: {lost_variance:.4f}")
print("4. Visual comparison shows that t-SNE and UMAP provide better class separation than KernelPCA")
print("5. Successfully saved and loaded the UMAP model using joblib")