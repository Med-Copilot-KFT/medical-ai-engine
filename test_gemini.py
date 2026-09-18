import os
import base64

from dotenv import load_dotenv
from google import genai


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

image_path = "image_extraction/uploads/screenshot.png"

with open(image_path, "rb") as f:
    image_bytes = f.read()

image_base64 = base64.b64encode(image_bytes).decode("utf-8")

interaction = client.interactions.create(
    model="gemini-3.6-flash",
    input=[
        {
            "type": "text",
            "text": """
Analyze this image carefully.

Give me:

1. A detailed description of the image
2. Important objects visible
3. Any people visible
4. Any text visible
5. The overall scene or context
6. Relationships between important objects
7. Any actions or activities visible

Be factual and only describe what you can actually observe.
""",
        },
        {
            "type": "image",
            "data": image_base64,
            "mime_type": "image/png",
        },
    ],
)

print("\n===== GEMINI VISION RESULT =====\n")
print(interaction.output_text)