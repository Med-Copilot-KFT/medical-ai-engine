import os
import gc
import psutil

import torch
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to

from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    EnsureTyped,
)
from monai.inferers import sliding_window_inference


# ============================================================
# SAFETY / MEMORY SETTINGS
# ============================================================

# Keep CPU usage conservative.
torch.set_num_threads(2)

# IMPORTANT:
# We explicitly use CPU.
DEVICE = torch.device("cpu")

print("=" * 60)
print("Spleen CT Segmentation - Controlled CPU Inference")
print("=" * 60)


# ============================================================
# MEMORY MONITOR
# ============================================================

process = psutil.Process(os.getpid())


def show_memory(label):
    memory_gb = process.memory_info().rss / (1024 ** 3)
    print(f"[MEMORY] {label}: {memory_gb:.2f} GB")


show_memory("Before loading model")


# ============================================================
# PATHS
# ============================================================

IMAGE_PATH = (
    "datasets/Task09_Spleen/imagesTr/spleen_19.nii.gz"
)

GROUND_TRUTH_PATH = (
    "datasets/Task09_Spleen/labelsTr/spleen_19.nii.gz"
)

MODEL_PATH = (
    "models/spleen_ct_segmentation/models/model.pt"
)

OUTPUT_DIR = (
    "medical_data/volumes/spleen_prediction"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CHECK INPUT FILES
# ============================================================

print("\nChecking input files...")

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        f"CT image not found: {IMAGE_PATH}"
    )

if not os.path.exists(GROUND_TRUTH_PATH):
    raise FileNotFoundError(
        f"Ground-truth label not found: {GROUND_TRUTH_PATH}"
    )

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

print("CT image:       OK")
print("Ground truth:   OK")
print("Model weights:  OK")


# ============================================================
# BUILD THE SAME NETWORK AS THE PRETRAINED MODEL
# ============================================================

print("\nLoading model...")

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm="batch",
)


# Load pretrained weights onto CPU.
checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True,
)

model.load_state_dict(checkpoint)

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")

show_memory("After loading model")


# ============================================================
# PREPROCESSING
#
# This matches the preprocessing from the MONAI
# spleen_ct_segmentation bundle.
# ============================================================

preprocessing = Compose(
    [
        LoadImaged(
            keys="image"
        ),

        EnsureChannelFirstd(
            keys="image"
        ),

        Orientationd(
            keys="image",
            axcodes="RAS"
        ),

        Spacingd(
            keys="image",
            pixdim=(1.5, 1.5, 2.0),
            mode="bilinear"
        ),

        ScaleIntensityRanged(
            keys="image",
            a_min=-57,
            a_max=164,
            b_min=0,
            b_max=1,
            clip=True
        ),

        EnsureTyped(
            keys="image"
        ),
    ]
)


# ============================================================
# LOAD CT
# ============================================================

print("\nLoading CT scan...")

data = {
    "image": IMAGE_PATH
}

data = preprocessing(data)

image = data["image"]

print("CT loaded successfully.")

print(
    "Preprocessed tensor shape:",
    tuple(image.shape)
)

print(
    "Tensor dtype:",
    image.dtype
)

tensor_size_gb = (
    image.numel()
    * image.element_size()
    / (1024 ** 3)
)

print(
    f"Preprocessed image memory: "
    f"{tensor_size_gb:.3f} GB"
)

show_memory("After preprocessing")


# ============================================================
# ADD BATCH DIMENSION
# ============================================================

input_tensor = image.unsqueeze(0).to(DEVICE)

print(
    "\nInference input shape:",
    tuple(input_tensor.shape)
)

show_memory("Before inference")


# ============================================================
# CONTROLLED SLIDING-WINDOW INFERENCE
# ============================================================

print("\nStarting inference...")
print("Device: CPU")
print("Window: 96 x 96 x 96")
print("Windows processed simultaneously: 1")
print("PyTorch CPU threads: 2")
print()
print("Please do not close the terminal while this runs.")
print("Inference may take some time on CPU.")
print()


with torch.inference_mode():

    prediction = sliding_window_inference(
        inputs=input_tensor,

        # Same window size as the pretrained bundle.
        roi_size=(96, 96, 96),

        # IMPORTANT:
        # Only one 3D window at a time.
        sw_batch_size=1,

        predictor=model,

        overlap=0.5,

        device=DEVICE,
    )


show_memory("After inference")


# ============================================================
# CONVERT MODEL OUTPUT TO A 3D LABEL MAP
# ============================================================

print("\nProcessing model output...")

print(
    "Raw prediction shape:",
    tuple(prediction.shape)
)

# Raw shape:
#
# [batch, classes, X, Y, Z]
#
# Example:
# [1, 2, 272, 272, 126]
#
# Remove batch dimension.

prediction = prediction[0]

print(
    "After removing batch dimension:",
    tuple(prediction.shape)
)


# Select the class with the highest probability
# for every voxel.
#
# Result:
#
# [X, Y, Z]

prediction = torch.argmax(
    prediction,
    dim=0
)


# Move to CPU and convert to uint8.
prediction_array = (
    prediction
    .cpu()
    .numpy()
    .astype(np.uint8)
)

print(
    "3D segmentation shape:",
    prediction_array.shape
)

print(
    "Predicted classes:",
    np.unique(prediction_array)
)


# ============================================================
# SAVE PREDICTION IN PREPROCESSED SPACE
# ============================================================

print("\nCreating processed-space segmentation...")

# The preprocessing pipeline transformed the CT into
# RAS orientation and 1.5 x 1.5 x 2.0 mm spacing.
#
# Use that transformed image's affine.

processed_affine = (
    np.asarray(image.affine)
)

processed_prediction_img = nib.Nifti1Image(
    prediction_array,
    processed_affine
)

processed_prediction_path = os.path.join(
    OUTPUT_DIR,
    "spleen_19_prediction_processed.nii.gz"
)

nib.save(
    processed_prediction_img,
    processed_prediction_path
)

print(
    "Processed prediction saved to:"
)

print(
    processed_prediction_path
)


# ============================================================
# RESAMPLE PREDICTION BACK TO ORIGINAL CT GRID
# ============================================================

print(
    "\nResampling prediction back to "
    "original CT grid..."
)

# Load the original CT.
original_ct = nib.load(
    IMAGE_PATH
)

# Resample the segmentation onto the exact
# original CT shape + affine.
#
# order=0 is nearest-neighbor interpolation,
# which is important for segmentation labels.
original_space_prediction = resample_from_to(
    processed_prediction_img,
    original_ct,
    order=0
)

original_prediction_array = (
    np.asarray(
        original_space_prediction.dataobj
    )
    .round()
    .astype(np.uint8)
)


# ============================================================
# SAVE ORIGINAL-SPACE PREDICTION
# ============================================================

original_prediction_path = os.path.join(
    OUTPUT_DIR,
    "spleen_19_prediction.nii.gz"
)

nib.save(
    original_space_prediction,
    original_prediction_path
)

print(
    "Original-space prediction saved to:"
)

print(
    original_prediction_path
)

print(
    "Original CT shape:",
    original_ct.shape
)

print(
    "Prediction shape:",
    original_prediction_array.shape
)


# ============================================================
# CALCULATE PREDICTED SPLEEN VOXELS
# ============================================================

predicted_spleen_voxels = int(
    np.sum(
        original_prediction_array == 1
    )
)

print(
    "\nPredicted spleen voxels:",
    predicted_spleen_voxels
)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading ground-truth segmentation...")

ground_truth_img = nib.load(
    GROUND_TRUTH_PATH
)

ground_truth_array = (
    np.asarray(
        ground_truth_img.dataobj
    )
    .round()
    .astype(np.uint8)
)

print(
    "Ground-truth shape:",
    ground_truth_array.shape
)

print(
    "Ground-truth classes:",
    np.unique(ground_truth_array)
)


# ============================================================
# CHECK SHAPES
# ============================================================

if (
    original_prediction_array.shape
    != ground_truth_array.shape
):
    raise RuntimeError(
        "Prediction and ground-truth shapes "
        "do not match."
    )


# ============================================================
# CALCULATE DICE SCORE
# ============================================================

prediction_mask = (
    original_prediction_array == 1
)

ground_truth_mask = (
    ground_truth_array == 1
)

intersection = np.logical_and(
    prediction_mask,
    ground_truth_mask
).sum()

prediction_count = prediction_mask.sum()

ground_truth_count = ground_truth_mask.sum()


if (
    prediction_count + ground_truth_count
    == 0
):
    dice_score = 1.0

else:
    dice_score = (
        2.0 * intersection
        / (
            prediction_count
            + ground_truth_count
        )
    )


print("\n" + "=" * 60)
print("INFERENCE COMPLETE")
print("=" * 60)

print(
    f"Predicted spleen voxels: "
    f"{prediction_count}"
)

print(
    f"Ground-truth spleen voxels: "
    f"{ground_truth_count}"
)

print(
    f"Intersection voxels: "
    f"{intersection}"
)

print(
    f"Dice score: "
    f"{dice_score:.4f}"
)

print()
print(
    "Processed prediction:"
)

print(
    processed_prediction_path
)

print()
print(
    "Original-space prediction:"
)

print(
    original_prediction_path
)

show_memory("Final")


# ============================================================
# CLEANUP
# ============================================================

del prediction
del input_tensor
del image
del data
del model
del checkpoint

gc.collect()

print(
    "\nInference memory cleanup completed."
)