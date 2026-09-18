import os
from PIL import Image
import pytesseract


def extract_image_data(image_path):
    """
    Extract basic metadata and text from an image.
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Open image
    image = Image.open(image_path)

    # Basic metadata
    image_data = {
        "filename": os.path.basename(image_path),
        "format": image.format,
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
    }

    # OCR
    extracted_text = pytesseract.image_to_string(image)

    image_data["extracted_text"] = extracted_text.strip()

    return image_data