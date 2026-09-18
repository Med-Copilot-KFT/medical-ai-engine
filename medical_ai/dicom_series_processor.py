import os
import pydicom


def load_dicom_series(folder_path):
    """
    Load all DICOM slices from a folder and return
    them ordered by their slice position.
    """

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(
            f"DICOM folder not found: {folder_path}"
        )

    slices = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if not os.path.isfile(file_path):
            continue

        try:
            ds = pydicom.dcmread(
                file_path,
                force=True
            )

            # Make sure this file contains image data
            if hasattr(ds, "PixelData"):
                slices.append(ds)

        except Exception:
            # Ignore files that aren't valid DICOM files
            continue

    if not slices:
        raise ValueError(
            "No DICOM images found in the folder."
        )

    # Prefer ImagePositionPatient for spatial ordering
    if all(
        hasattr(ds, "ImagePositionPatient")
        for ds in slices
    ):
        slices.sort(
            key=lambda ds: float(
                ds.ImagePositionPatient[2]
            )
        )

    # Fallback to InstanceNumber
    elif all(
        hasattr(ds, "InstanceNumber")
        for ds in slices
    ):
        slices.sort(
            key=lambda ds: int(ds.InstanceNumber)
        )

    return slices


def get_series_info(slices):
    """
    Extract basic information about the DICOM series.
    """

    if not slices:
        raise ValueError("No DICOM slices provided.")

    first = slices[0]

    return {
        "number_of_slices": len(slices),
        "modality": str(
            getattr(first, "Modality", "Unknown")
        ),
        "body_part": str(
            getattr(first, "BodyPartExamined", "Unknown")
        ),
        "rows": int(
            getattr(first, "Rows", 0)
        ),
        "columns": int(
            getattr(first, "Columns", 0)
        ),
        "slice_thickness": float(
            getattr(first, "SliceThickness", 0)
        ),
    }