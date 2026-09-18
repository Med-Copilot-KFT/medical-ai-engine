import os
import base64

from dotenv import load_dotenv
from google import genai


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")


client = genai.Client(api_key=api_key)


def analyze_image(image_path):
    """
    Analyze an image using Gemini Vision.
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
Analyze this image carefully.

Return:

1. Detailed description
2. Important objects
3. People visible
4. Text visible
5. Overall scene or context
6. Relationships between important objects
7. Actions or activities visible

Be factual and only describe what you can actually observe.
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