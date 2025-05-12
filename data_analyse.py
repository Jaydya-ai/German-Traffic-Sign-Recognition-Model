import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import skew, entropy
from scipy.special import kl_div
import multiprocessing
from sklearn.manifold import TSNE
import tqdm

# paths
train_path = os.path.join("GTSRB", "Final_Training", "Images")
test_path = os.path.join("GTSRB", "Final_Test", "Images")
train_hog_path = os.path.join("GTSRB", "Final_Training", "HOG")
test_hog_path = os.path.join("GTSRB", "Final_Test", "HOG")
train_huehist_path = os.path.join("GTSRB", "Final_Training", "HueHist")
test_huehist_path = os.path.join("GTSRB", "Final_Test", "HueHist")
test_csv_path = os.path.join("GTSRB", "GT-final_test.csv")

# workers to help speed up processing
num_workers = multiprocessing.cpu_count()

# load test data
print("Loading test CSV...")
df_test = pd.read_csv(test_csv_path, sep=";")

# ==============================
# CLASS DISTRIBUTION
# ==============================

print("Analyzing class distribution...")
train_class_counts = {}
for class_id in os.listdir(train_path):
    class_dir = os.path.join(train_path, class_id)
    if os.path.isdir(class_dir):
        numeric_class_id = int(class_id.lstrip('0')) if class_id.lstrip('0') else 0
        train_class_counts[numeric_class_id] = len([f for f in os.listdir(class_dir) if f.endswith('.ppm')])

train_class_counts = pd.Series(train_class_counts).sort_index()
test_class_counts = df_test["ClassId"].value_counts().sort_index()

# class distribution stats
print("\nClass Distribution Summary:")
print(f"Training set: {len(train_class_counts)} classes, {train_class_counts.sum()} images")
print(f"Test set: {len(test_class_counts)} classes, {test_class_counts.sum()} images")
print(f"Training set - Min class size: {train_class_counts.min()} (Class {train_class_counts.idxmin()})")
print(f"Training set - Max class size: {train_class_counts.max()} (Class {train_class_counts.idxmax()})")
class_imbalance_ratio = train_class_counts.max() / train_class_counts.min()
print(f"Class imbalance ratio (max/min): {class_imbalance_ratio:.2f}x")
if class_imbalance_ratio > 10:
    print("⚠️ Significant class imbalance detected. Consider oversampling or weighted loss.")

# ==============================
# IMAGE SIZE DISTRIBUTION
# ==============================

print("\nChecking image sizes...")
train_image_sizes = []
test_image_sizes = []
invalid_images = []
brightness_values = []
contrast_values = []

def process_image(img_path):
    """process a single image for size, brightness, and contrast."""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None, None, None, img_path
        size = img.shape[:2]
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray_img)
        contrast = np.std(gray_img)
        return size, brightness, contrast, None
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return None, None, None, img_path

def process_class_dir(class_id):
    """process all images in a class directory."""
    local_sizes = []
    local_brightness = []
    local_contrast = []
    local_invalid = []
    class_dir = os.path.join(train_path, class_id)
    if os.path.isdir(class_dir):
        img_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.ppm', '.jpg', '.png'))]
        print(f"Processing class {class_id} ({len(img_files)} images)...")
        for img_file in img_files:
            img_path = os.path.join(class_dir, img_file)
            size, brightness, contrast, invalid = process_image(img_path)
            if size:
                local_sizes.append(size)
                local_brightness.append(brightness)
                local_contrast.append(contrast)
            if invalid:
                local_invalid.append(invalid)
    return local_sizes, local_brightness, local_contrast, local_invalid

# processing for training images
print(f"Processing training images with {num_workers} workers...")
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    results = list(executor.map(process_class_dir, [class_id for class_id in os.listdir(train_path)]))
for sizes, brightness, contrast, invalid in results:
    train_image_sizes.extend(sizes)
    brightness_values.extend(brightness)
    contrast_values.extend(contrast)
    invalid_images.extend(invalid)

# processing for test images
test_image_files = [f for f in os.listdir(test_path) if f.lower().endswith(('.ppm', '.jpg', '.png'))]
print(f"Processing {len(test_image_files)} test images with {num_workers} workers...")
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    results = list(executor.map(process_image, [os.path.join(test_path, f) for f in test_image_files]))
for size, brightness, contrast, invalid in results:
    if size:
        test_image_sizes.append(size)
    if invalid:
        invalid_images.append(invalid)

# convert to numpy arrays
train_image_sizes = np.array(train_image_sizes)
test_image_sizes = np.array(test_image_sizes)
brightness_values = np.array(brightness_values)
contrast_values = np.array(contrast_values)

# training set analysis
train_unique_sizes, train_counts = np.unique(train_image_sizes, axis=0, return_counts=True)
train_sorted_indices = np.argsort(-train_counts)
train_top_sizes = train_unique_sizes[train_sorted_indices][:5]
train_top_counts = train_counts[train_sorted_indices][:5]

# test set analysis
test_unique_sizes, test_counts = np.unique(test_image_sizes, axis=0, return_counts=True)
test_sorted_indices = np.argsort(-test_counts)
test_top_sizes = test_unique_sizes[test_sorted_indices][:5]
test_top_counts = test_counts[test_sorted_indices][:5]

# calc avg dimensions and variability
train_avg_height = np.mean(train_image_sizes[:, 0])
train_avg_width = np.mean(train_image_sizes[:, 1])
train_height_cv = np.std(train_image_sizes[:, 0]) / train_avg_height * 100
train_width_cv = np.std(train_image_sizes[:, 1]) / train_avg_width * 100
test_avg_height = np.mean(test_image_sizes[:, 0])
test_avg_width = np.mean(test_image_sizes[:, 1])
test_height_cv = np.std(test_image_sizes[:, 0]) / test_avg_height * 100
test_width_cv = np.std(test_image_sizes[:, 1]) / test_avg_width * 100

# img size stats
print("\nImage Size Summary:")
print(f"Training set: {len(train_image_sizes)} images")
print(f"  Average size: {train_avg_height:.1f}x{train_avg_width:.1f} pixels")
print(f"  Size variability: Height Coefficient of Variation = {train_height_cv:.1f}%, Width Coefficient of Variation = {train_width_cv:.1f}%")
print("  Top 5 most common sizes:")
for size, count in zip(train_top_sizes, train_top_counts):
    print(f"    {size[0]}x{size[1]} - {count} images ({count/len(train_image_sizes)*100:.2f}%)")
print(f"Test set: {len(test_image_sizes)} images")
print(f"  Average size: {test_avg_height:.1f}x{test_avg_width:.1f} pixels")
print(f"  Size variability: Height Coefficient of Variation = {test_height_cv:.1f}%, Width Coefficient of Variation = {test_width_cv:.1f}%")
print("  Top 5 most common sizes:")
for size, count in zip(test_top_sizes, test_top_counts):
    print(f"    {size[0]}x{size[1]} - {count} images ({count/len(test_image_sizes)*100:.2f}%)")
if train_height_cv > 20 or train_width_cv > 20:
    print("⚠️ High variability in training image sizes. Consider resizing for consistency.")
if abs(train_avg_height - test_avg_height) > 10 or abs(train_avg_width - test_avg_width) > 10:
    print("⚠️ Training and test set average sizes differ significantly. Verify dataset alignment.")

# ==============================
# BRIGHTNESS & CONTRAST
# ==============================

# brightness and contrast stats
brightness_skewness = skew(brightness_values)
contrast_skewness = skew(contrast_values)

print("\nBrightness & Contrast Summary:")
print(f"Brightness (pixel intensity, 0-255):")
print(f"  Mean: {np.mean(brightness_values):.2f}")
print(f"  Range: {np.min(brightness_values):.2f} to {np.max(brightness_values):.2f}")
print(f"  Skewness: {brightness_skewness:.2f}")
if abs(brightness_skewness) > 1:
    print("⚠️ Brightness distribution is skewed. Consider normalization.")
print(f"Contrast (std of pixel intensity):")
print(f"  Mean: {np.mean(contrast_values):.2f}")
print(f"  Range: {np.min(contrast_values):.2f} to {np.max(contrast_values):.2f}")
print(f"  Skewness: {contrast_skewness:.2f}")
if abs(contrast_skewness) > 1:
    print("⚠️ Contrast distribution is skewed. Consider contrast stretching.")
if np.min(brightness_values) < 50 or np.max(brightness_values) > 200:
    print("⚠️ Extreme brightness values detected. May affect model performance.")

# ==============================
# HOG FEATURES ANALYSIS
# ==============================

print("\nAnalyzing HOG features...")
hog_configs = ["HOG_01", "HOG_02", "HOG_03"]
hog_dims = {"HOG_01": 1568, "HOG_02": 1568, "HOG_03": 2916}
train_hog_features = {config: [] for config in hog_configs}
train_hog_labels = {config: [] for config in hog_configs}
test_hog_features = {config: [] for config in hog_configs}
invalid_hog_files = []

# max samples per class for subsampling
max_samples_per_class = 700
max_tsne_samples = 7000

# loads single HOG feature file
def load_hog_file(args):
    file_path, expected_dim, class_id = args
    try:
        features = np.loadtxt(file_path)
        if features.shape[0] != expected_dim:
            print(f"Invalid dimension in {file_path}: expected {expected_dim}, got {features.shape[0]}")
            return None, None
        return features, class_id
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

# load training HOG features
for config in hog_configs:
    config_path = os.path.join(train_hog_path, config)
    print(f"Processing training {config} features...")
    file_args = []
    class_sample_counts = {}
    for class_id in os.listdir(config_path):
        class_dir = os.path.join(config_path, class_id)
        if os.path.isdir(class_dir):
            numeric_class_id = int(class_id.lstrip('0')) if class_id.lstrip('0') else 0
            class_sample_counts[numeric_class_id] = 0
            hog_files = [f for f in os.listdir(class_dir) if f.endswith('.txt')]
            np.random.shuffle(hog_files)  # randomize for subsampling
            for hog_file in hog_files[:max_samples_per_class]:
                file_path = os.path.join(class_dir, hog_file)
                file_args.append((file_path, hog_dims[config], numeric_class_id))
    print(f"Loading {len(file_args)} training {config} files with {num_workers} workers...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm.tqdm(executor.map(load_hog_file, file_args), total=len(file_args)))
    for features, class_id in results:
        if features is not None:
            train_hog_features[config].append(features)
            train_hog_labels[config].append(class_id)
        else:
            invalid_hog_files.append(file_path)
    train_hog_features[config] = np.array(train_hog_features[config])
    train_hog_labels[config] = np.array(train_hog_labels[config])

# load test HOG features
for config in hog_configs:
    config_path = os.path.join(test_hog_path, config)
    print(f"Processing test {config} features...")
    hog_files = [f for f in os.listdir(config_path) if f.endswith('.txt')]
    np.random.shuffle(hog_files)
    file_args = [(os.path.join(config_path, hog_file), hog_dims[config], None) 
                 for hog_file in hog_files[:max_samples_per_class * 10]]  # roughly match training size
    print(f"Loading {len(file_args)} test {config} files with {num_workers} workers...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm.tqdm(executor.map(load_hog_file, file_args), total=len(file_args)))
    for features, _ in results:
        if features is not None:
            test_hog_features[config].append(features)
        else:
            invalid_hog_files.append(file_path)
    test_hog_features[config] = np.array(test_hog_features[config])

# compute HOG stats
hog_stats = {}
for config in hog_configs:
    train_features = train_hog_features[config]
    test_features = test_hog_features[config]
    stats = {
        "train_mean": np.mean(train_features, axis=0).mean() if train_features.size else 0,
        "train_var": np.var(train_features, axis=0).mean() if train_features.size else 0,
        "train_range": (np.min(train_features), np.max(train_features)) if train_features.size else (0, 0),
        "test_mean": np.mean(test_features, axis=0).mean() if test_features.size else 0,
        "test_var": np.var(test_features, axis=0).mean() if test_features.size else 0,
        "test_range": (np.min(test_features), np.max(test_features)) if test_features.size else (0, 0)
    }
    hog_stats[config] = stats

# class discriminability analysis (training set only)
class_discriminability = {}
for config in hog_configs:
    features = train_hog_features[config]
    labels = train_hog_labels[config]
    class_vars = []
    for class_id in np.unique(labels):
        class_features = features[labels == class_id]
        class_var = np.var(class_features, axis=0).mean() if class_features.size else 0
        class_vars.append(class_var)
    inter_class_var = np.var([np.mean(features[labels == class_id], axis=0) 
                             for class_id in np.unique(labels)], axis=0).mean() if features.size else 0
    class_discriminability[config] = {
        "mean_intra_class_var": np.mean(class_vars) if class_vars else 0,
        "inter_class_var": inter_class_var
    }

# print HOG stats
print("\nHOG Features Summary defamation(Subsampled):")
for config in hog_stats:
    stats = hog_stats[config]
    print(f"{config}:")
    print(f"  Training set: {train_hog_features[config].shape[0]} features (subsampled), dimension {hog_dims[config]}")
    print(f"    Mean: {stats['train_mean']:.4f}, Variance: {stats['train_var']:.4f}")
    print(f"    Range: {stats['train_range'][0]:.4f} to {stats['train_range'][1]:.4f}")
    print(f"  Test set: {test_hog_features[config].shape[0]} features (subsampled), dimension {hog_dims[config]}")
    print(f"    Mean: {stats['test_mean']:.4f}, Variance: {stats['test_var']:.4f}")
    print(f"    Range: {stats['test_range'][0]:.4f} to {stats['test_range'][1]:.4f}")
    disc = class_discriminability[config]
    print(f"  Class Discriminability:")
    print(f"    Mean Intra-class Variance: {disc['mean_intra_class_var']:.4f}")
    print(f"    Inter-class Variance: {disc['inter_class_var']:.4f}")
    if disc['inter_class_var'] > disc['mean_intra_class_var']:
        print(f"    ✅ {config} features show good class separability.")
    else:
        print(f"    ⚠️ {config} features may have limited class separability.")
if invalid_hog_files:
    print(f"⚠️ {len(invalid_hog_files)} HOG files failed to load. Check for corruption.")

# ==============================
# HUE HISTOGRAM FEATURES ANALYSIS
# ==============================

print("\nAnalyzing Hue Histogram features...")
huehist_dim = 256
train_huehist_features = []
train_huehist_labels = []
test_huehist_features = []
invalid_huehist_files = []

# loads a single HueHist feature file
def load_huehist_file(args):
    file_path, expected_dim, class_id = args
    try:
        features = np.loadtxt(file_path)
        if features.shape[0] != expected_dim:
            print(f"Invalid dimension in {file_path}: expected {expected_dim}, got {features.shape[0]}")
            return None, None
        # normalize histogram to sum to 1
        features = features / (np.sum(features) + 1e-10)
        return features, class_id
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

# load training HueHist features
print(f"Processing training HueHist features...")
file_args = []
class_sample_counts = {}
for class_id in os.listdir(train_huehist_path):
    class_dir = os.path.join(train_huehist_path, class_id)
    if os.path.isdir(class_dir):
        numeric_class_id = int(class_id.lstrip('0')) if class_id.lstrip('0') else 0
        class_sample_counts[numeric_class_id] = 0
        huehist_files = [f for f in os.listdir(class_dir) if f.endswith('.txt')]
        np.random.shuffle(huehist_files)
        for huehist_file in huehist_files[:max_samples_per_class]:
            file_path = os.path.join(class_dir, huehist_file)
            file_args.append((file_path, huehist_dim, numeric_class_id))
print(f"Loading {len(file_args)} training HueHist files with {num_workers} workers...")
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    results = list(tqdm.tqdm(executor.map(load_huehist_file, file_args), total=len(file_args)))
for features, class_id in results:
    if features is not None:
        train_huehist_features.append(features)
        train_huehist_labels.append(class_id)
    else:
        invalid_huehist_files.append(file_path)
train_huehist_features = np.array(train_huehist_features)
train_huehist_labels = np.array(train_huehist_labels)

# load test HueHist features
print(f"Processing test HueHist features...")
huehist_files = [f for f in os.listdir(test_huehist_path) if f.endswith('.txt')]
np.random.shuffle(huehist_files)
file_args = [(os.path.join(test_huehist_path, huehist_file), huehist_dim, None) 
             for huehist_file in huehist_files[:max_samples_per_class * 10]]
print(f"Loading {len(file_args)} test HueHist files with {num_workers} workers...")
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    results = list(tqdm.tqdm(executor.map(load_huehist_file, file_args), total=len(file_args)))
for features, _ in results:
    if features is not None:
        test_huehist_features.append(features)
    else:
        invalid_huehist_files.append(file_path)
test_huehist_features = np.array(test_huehist_features)

# compute HueHist stats
huehist_stats = {
    "train_mean": np.mean(train_huehist_features, axis=0).mean() if train_huehist_features.size else 0,
    "train_var": np.var(train_huehist_features, axis=0).mean() if train_huehist_features.size else 0,
    "train_entropy": np.mean([entropy(hist + 1e-10) for hist in train_huehist_features]) if train_huehist_features.size else 0,
    "test_mean": np.mean(test_huehist_features, axis=0).mean() if test_huehist_features.size else 0,
    "test_var": np.var(test_huehist_features, axis=0).mean() if test_huehist_features.size else 0,
    "test_entropy": np.mean([entropy(hist + 1e-10) for hist in test_huehist_features]) if test_huehist_features.size else 0
}

# class discriminability analysis using KL divergence
huehist_kl_divergence = []
for class_id in np.unique(train_huehist_labels):
    class_hist = train_huehist_features[train_huehist_labels == class_id]
    mean_class_hist = np.mean(class_hist, axis=0) + 1e-10
    for other_class_id in np.unique(train_huehist_labels):
        if other_class_id > class_id:
            other_hist = train_huehist_features[train_huehist_labels == other_class_id]
            mean_other_hist = np.mean(other_hist, axis=0) + 1e-10
            kl = np.sum(kl_div(mean_class_hist, mean_other_hist))
            huehist_kl_divergence.append(kl)
huehist_avg_kl = np.mean(huehist_kl_divergence) if huehist_kl_divergence else 0

# print HueHist stats
print("\nHue Histogram Features Summary (Subsampled):")
print(f"Training set: {train_huehist_features.shape[0]} features (subsampled), dimension {huehist_dim}")
print(f"  Mean: {huehist_stats['train_mean']:.4f}, Variance: {huehist_stats['train_var']:.4f}")
print(f"  Entropy: {huehist_stats['train_entropy']:.4f}")
print(f"Test set: {test_huehist_features.shape[0]} features (subsampled), dimension {huehist_dim}")
print(f"  Mean: {huehist_stats['test_mean']:.4f}, Variance: {huehist_stats['test_var']:.4f}")
print(f"  Entropy: {huehist_stats['test_entropy']:.4f}")
print(f"Class Discriminability:")
print(f"  Average KL Divergence between classes: {huehist_avg_kl:.4f}")
if huehist_avg_kl > 1.0:
    print(f"  ✅ HueHist features show good class separability.")
else:
    print(f"  ⚠️ HueHist features may have limited class separability.")
if invalid_huehist_files:
    print(f"⚠️ {len(invalid_huehist_files)} HueHist files failed to load. Check for corruption.")

# ==============================
# GRAPHS
# ==============================

print("\nGenerating plots...")
fig, ax = plt.subplots(4, 2, figsize=(14, 20))

# training set distribution
ax[0, 0].bar(train_class_counts.index, train_class_counts.values, color='blue')
ax[0, 0].set_title("Training Set Class Distribution")
ax[0, 0].set_xlabel("Traffic Sign Class")
ax[0, 0].set_ylabel("Number of Images")
ax[0, 0].grid(axis='y', linestyle='--', alpha=0.7)

# test set distribution
ax[0, 1].bar(test_class_counts.index, test_class_counts.values, color='red')
ax[0, 1].set_title("Test Set Class Distribution")
ax[0, 1].set_xlabel("Traffic Sign Class")
ax[0, 1].grid(axis='y', linestyle='--', alpha=0.7)

# img size distribution
num_display_sizes = min(10, len(train_unique_sizes))
x = np.arange(num_display_sizes)
width = 0.35
ax[1, 0].bar(x - width/2, train_counts[train_sorted_indices][:num_display_sizes], 
             width, label='Training', color='green')
ax[1, 0].bar(x + width/2, test_counts[test_sorted_indices][:num_display_sizes], 
             width, label='Test', color='red')
ax[1, 0].set_xticks(x)
ax[1, 0].set_xticklabels([f"{s[0]}x{s[1]}" for s in train_unique_sizes[train_sorted_indices][:num_display_sizes]], 
                         rotation=45)
ax[1, 0].set_xlabel("Image Dimensions")
ax[1, 0].set_ylabel("Number of Images")
ax[1, 0].set_title("Top Image Sizes in Both Sets")
ax[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
ax[1, 0].legend()

# brightness and contrast histograms
ax[1, 1].hist(brightness_values, bins=30, color='orange', alpha=0.5, label='Brightness')
ax[1, 1].hist(contrast_values, bins=30, color='purple', alpha=0.5, label='Contrast')
ax[1, 1].set_title("Brightness & Contrast Distribution")
ax[1, 1].set_xlabel("Value")
ax[1, 1].set_ylabel("Frequency")
ax[1, 1].legend()

# HOG feature histograms (for HOG_01 as example)
hog_example = train_hog_features["HOG_01"]
if hog_example.size:
    ax[2, 0].hist(hog_example.ravel(), bins=50, color='cyan', alpha=0.7)
    ax[2, 0].set_title("HOG_01 Feature Distribution (Training, Subsampled)")
    ax[2, 0].set_xlabel("Feature Value")
    ax[2, 0].set_ylabel("Frequency")
    ax[2, 0].grid(axis='y', linestyle='--', alpha=0.7)

# t-SNE visualization for HOG_01 (subsampled)
if hog_example.shape[0] > max_tsne_samples:
    indices = np.random.choice(hog_example.shape[0], max_tsne_samples, replace=False)
    tsne_features = hog_example[indices]
    tsne_labels = train_hog_labels["HOG_01"][indices]
else:
    tsne_features = hog_example
    tsne_labels = train_hog_labels["HOG_01"]
if tsne_features.size:
    print("Computing t-SNE for HOG_01 (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, n_jobs=num_workers)
    tsne_results = tsne.fit_transform(tsne_features)
    scatter = ax[2, 1].scatter(tsne_results[:, 0], tsne_results[:, 1], c=tsne_labels, cmap='viridis', alpha=0.6)
    ax[2, 1].set_title("t-SNE of HOG_01 Features (Training, Subsampled)")
    ax[2, 1].set_xlabel("t-SNE Component 1")
    ax[2, 1].set_ylabel("t-SNE Component 2")
    plt.colorbar(scatter, ax=ax[2, 1], label="Class ID")

# HueHist average histogram for selected classes
selected_classes = [1, 2, 13, 14]
for class_id in selected_classes:
    class_hist = train_huehist_features[train_huehist_labels == class_id]
    mean_hist = np.mean(class_hist, axis=0) if class_hist.size else np.zeros(huehist_dim)
    ax[3, 0].plot(mean_hist, label=f"Class {class_id}", alpha=0.7)
ax[3, 0].set_title("Average Hue Histograms for Selected Classes")
ax[3, 0].set_xlabel("Hue Bin (0-255)")
ax[3, 0].set_ylabel("Probability")
ax[3, 0].legend()
ax[3, 0].grid(axis='y', linestyle='--', alpha=0.7)

# t-SNE visualization for HueHist (subsampled)
if train_huehist_features.shape[0] > max_tsne_samples:
    indices = np.random.choice(train_huehist_features.shape[0], max_tsne_samples, replace=False)
    tsne_features = train_huehist_features[indices]
    tsne_labels = train_huehist_labels[indices]
else:
    tsne_features = train_huehist_features
    tsne_labels = train_huehist_labels
if tsne_features.size:
    print("Computing t-SNE for HueHist (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, n_jobs=num_workers)
    tsne_results = tsne.fit_transform(tsne_features)
    scatter = ax[3, 1].scatter(tsne_results[:, 0], tsne_results[:, 1], c=tsne_labels, cmap='plasma', alpha=0.6)
    ax[3, 1].set_title("t-SNE of HueHist Features (Training, Subsampled)")
    ax[3, 1].set_xlabel("t-SNE Component 1")
    ax[3, 1].set_ylabel("t-SNE Component 2")
    plt.colorbar(scatter, ax=ax[3, 1], label="Class ID")

plt.tight_layout()
print("Displaying plots...")
plt.savefig('dataset_analysis_plots.png')
plt.show()

# ==============================
# DESCRIPTIVE ANALYSIS SUMMARY
# ==============================

print("\nDescriptive Analysis Summary:")
print("1. Class Distribution:")
print(f"   - The dataset contains {len(train_class_counts)} traffic sign classes.")
print(f"   - Training set has {train_class_counts.sum()} images, test set has {test_class_counts.sum()} images.")
print(f"   - Class imbalance ratio is {class_imbalance_ratio:.2f}x, indicating {'significant' if class_imbalance_ratio > 10 else 'moderate'} variation.")
print("2. Image Sizes:")
print(f"   - Training images average {train_avg_height:.1f}x{train_avg_width:.1f} pixels with {train_height_cv:.1f}% height variability.")
print(f"   - Test images average {test_avg_height:.1f}x{test_avg_width:.1f} pixels with {test_height_cv:.1f}% height variability.")
print(f"   - {'High' if train_height_cv > 20 else 'Moderate'} size variability suggests {'resizing' if train_height_cv > 20 else 'optional resizing'} for model training.")
print("3. Brightness & Contrast:")
print(f"   - Brightness averages {np.mean(brightness_values):.2f} with {'skewed' if abs(brightness_skewness) > 1 else 'normal'} distribution.")
print(f"   - Contrast averages {np.mean(contrast_values):.2f} with {'skewed' if abs(contrast_skewness) > 1 else 'normal'} distribution.")
print(f"   - {'Normalization recommended' if abs(brightness_skewness) > 1 or np.min(brightness_values) < 50 else 'Standard preprocessing sufficient'}.")
print("4. HOG Features (Subsampled):")
for config in hog_configs:
    stats = hog_stats[config]
    disc = class_discriminability[config]
    print(f"   - {config}: Dimension {hog_dims[config]}, Training samples {train_hog_features[config].shape[0]} (subsampled), Test samples {test_hog_features[config].shape[0]} (subsampled)")
    print(f"     Mean feature value: {stats['train_mean']:.4f} (train), {stats['test_mean']:.4f} (test)")
    print(f"     Variance: {stats['train_var']:.4f} (train), {stats['test_var']:.4f} (test)")
    print(f"     Class separability: Intra-class var {disc['mean_intra_class_var']:.4f}, Inter-class var {disc['inter_class_var']:.4f}")
    if disc['inter_class_var'] > disc['mean_intra_class_var']:
        print(f"     ✅ Good separability for {config}.")
    else:
        print(f"     ⚠️ Limited separability for {config}.")
print("5. Hue Histogram Features (Subsampled):")
print(f"   - Dimension {huehist_dim}, Training samples {train_huehist_features.shape[0]} (subsampled), Test samples {test_huehist_features.shape[0]} (subsampled)")
print(f"     Mean: {huehist_stats['train_mean']:.4f} (train), {huehist_stats['test_mean']:.4f} (test)")
print(f"     Variance: {huehist_stats['train_var']:.4f} (train), {huehist_stats['test_var']:.4f} (test)")
print(f"     Entropy: {huehist_stats['train_entropy']:.4f} (train), {huehist_stats['test_entropy']:.4f} (test)")
print(f"     Class separability: Avg KL Divergence {huehist_avg_kl:.4f}")
if huehist_avg_kl > 1.0:
    print(f"     ✅ Good separability for HueHist.")
else:
    print(f"     ⚠️ Limited separability for HueHist.")
print("6. Potential Issues:")
if invalid_images:
    print(f"   - {len(invalid_images)} images failed to load, indicating possible corruption.")
if invalid_hog_files:
    print(f"   - {len(invalid_hog_files)} HOG files failed to load, indicating possible corruption.")
if invalid_huehist_files:
    print(f"   - {len(invalid_huehist_files)} HueHist files failed to load, indicating possible corruption.")
if class_imbalance_ratio > 10:
    print("   - Class imbalance may bias model toward overrepresented classes.")
if train_height_cv > 20:
    print("   - High image size variability may require resizing to standardize input.")
if np.min(brightness_values) < 50:
    print("   - Low-brightness images may need enhancement for better feature detection.")
for config in hog_configs:
    if class_discriminability[config]['inter_class_var'] <= class_discriminability[config]['mean_intra_class_var']:
        print(f"   - {config} features may not effectively separate classes, consider alternative features or neural networks.")
if huehist_avg_kl <= 1.0:
    print(f"   - HueHist features may not effectively separate classes, consider combining with other features or neural networks.")

print("\nDataset analysis complete.")