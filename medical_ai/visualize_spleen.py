import os

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

CT_PATH = (
    "datasets/Task09_Spleen/imagesTr/spleen_19.nii.gz"
)

GROUND_TRUTH_PATH = (
    "datasets/Task09_Spleen/labelsTr/spleen_19.nii.gz"
)

PREDICTION_PATH = (
    "medical_data/volumes/spleen_prediction/"
    "spleen_19_prediction.nii.gz"
)

OUTPUT_DIR = (
    "medical_data/volumes/spleen_prediction"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "spleen_19_visual_comparison.png"
)


# ============================================================
# CHECK FILES
# ============================================================

for path in [
    CT_PATH,
    GROUND_TRUTH_PATH,
    PREDICTION_PATH,
]:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found: {path}"
        )


# ============================================================
# LOAD DATA
# ============================================================

print("Loading CT...")

ct_img = nib.load(CT_PATH)
ct = np.asarray(ct_img.dataobj)


print("Loading ground truth...")

gt_img = nib.load(GROUND_TRUTH_PATH)
ground_truth = np.asarray(
    gt_img.dataobj
)


print("Loading AI prediction...")

pred_img = nib.load(PREDICTION_PATH)
prediction = np.asarray(
    pred_img.dataobj
)


# ============================================================
# FIND THE BEST SLICE
# ============================================================

ground_truth_mask = (
    ground_truth == 1
)

prediction_mask = (
    prediction == 1
)


# Find slice containing the largest amount
# of ground-truth spleen tissue.

slice_counts = np.sum(
    ground_truth_mask,
    axis=(0, 1)
)

best_slice = int(
    np.argmax(slice_counts)
)


print(
    "\nBest visualization slice:",
    best_slice
)

print(
    "CT shape:",
    ct.shape
)

print(
    "Ground-truth shape:",
    ground_truth.shape
)

print(
    "Prediction shape:",
    prediction.shape
)


# ============================================================
# EXTRACT SLICE
# ============================================================

ct_slice = ct[:, :, best_slice]

gt_slice = ground_truth_mask[
    :, :, best_slice
]

pred_slice = prediction_mask[
    :, :, best_slice
]


# ============================================================
# CREATE VISUALIZATION
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 9)
)


# CT image
ax.imshow(
    np.rot90(ct_slice),
    cmap="gray"
)


# Ground-truth contour
ax.contour(
    np.rot90(gt_slice),
    levels=[0.5],
    linewidths=2,
    colors="lime",
)


# AI prediction contour
ax.contour(
    np.rot90(pred_slice),
    levels=[0.5],
    linewidths=2,
    colors="red",
)


ax.set_title(
    "Spleen CT Segmentation\n"
    "Green = Ground Truth | Red = AI Prediction"
)

ax.axis("off")


plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

plt.savefig(
    OUTPUT_PATH,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print(
    "\nVisualization saved to:"
)

print(
    OUTPUT_PATH
)