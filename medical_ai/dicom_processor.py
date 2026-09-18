import os
import pydicom


def read_dicom(dicom_path):
    """
    Read a DICOM file and extract basic study information.
    """

    if not os.path.exists(dicom_path):
        raise FileNotFoundError(f"DICOM file not found: {dicom_path}")

    ds = pydicom.dcmread(dicom_path)

    result = {
        "filename": os.path.basename(dicom_path),
        "modality": str(getattr(ds, "Modality", "Unknown")),
        "study_description": str(
            getattr(ds, "StudyDescription", "Unknown")
        ),
        "series_description": str(
            getattr(ds, "SeriesDescription", "Unknown")
        ),
        "body_part": str(
            getattr(ds, "BodyPartExamined", "Unknown")
        ),
        "rows": int(getattr(ds, "Rows", 0)),
        "columns": int(getattr(ds, "Columns", 0)),
        "has_pixel_data": hasattr(ds, "PixelData"),
    }

    return result