import os
import base64

from dotenv import load_dotenv
from google import genai


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)


def classify_medical_image(image_path):
    """
    Identify the likely medical imaging modality,
    anatomical region, and whether the image appears
    suitable for further analysis.
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    extension = os.path.splitext(image_path)[1].lower()

    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }

    mime_type = mime_types.get(extension, "image/png")

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=[
            {
                "type": "text",
                "text": """
You are performing preliminary medical-image study classification.

Analyze the image and return ONLY valid JSON.

Use this structure:

{
  "modality": "",
  "anatomical_region": "",
  "study_type": "",
  "image_type": "",
  "quality": "",
  "confidence": 0.0,
  "reasoning": "",
  "limitations": []
}

Instructions:

- modality examples: CT, MRI, X-ray, ultrasound,
  mammography, retinal imaging, pathology,
  endoscopy, photograph, unknown
- anatomical_region should identify the body region if reasonably
  visible.
- study_type should describe the likely examination.
- image_type should distinguish things such as a raw image,
  screenshot, 3D reconstruction, rendered image, etc.
- quality should be "adequate", "limited", or "poor".
- confidence must be between 0 and 1.
- reasoning should briefly explain the visual evidence.
- limitations should contain uncertainties.
- Do not diagnose a disease.
- Do not invent information that cannot be determined from the image.
""",
            },
            {
                "type": "image",
                "data": image_base64,
                "mime_type": mime_type,
            },
        ],
    )

    return interaction.output_text