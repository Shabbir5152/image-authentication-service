import os
import json
import requests
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import uvicorn

# Load environment variables
load_dotenv()

# Configure Gemini API
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

genai.configure(api_key=API_KEY)

# Initialize FastAPI
app = FastAPI(title="Image Authentication Service")

class ImageRequest(BaseModel):
    url: str

def fetch_image(url: str):
    """
    Fetches an image from a URL and returns a PIL Image object.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image
    except Exception as e:
        print(f"Error fetching image: {e}")
        return None

@app.get("/")
def read_root():
    return {"message": "Image Authentication Service is running. Use POST /analyze to check images. Docs at /docs"}

@app.get("/analyze")
async def analyze_image_get(url: str):
    return await analyze_image_endpoint(ImageRequest(url=url))


@app.post("/analyze")
async def analyze_image_endpoint(request: ImageRequest):
    """
    Analyzes the image using Gemini to detect deepfakes or damage modification.
    """
    image_url = request.url
    print(f"Received request to analyze: {image_url}")

    image = fetch_image(image_url)
    if not image:
        raise HTTPException(status_code=400, detail="Failed to fetch image from URL.")

    # Use a vision-capable model
    # gemini-2.5-flash is used for speed and quota efficiency
    model = genai.GenerativeModel('gemini-2.5-flash')

    prompt = """
    Act as a Senior Forensic Digital Media Analyst. Your objective is to perform a multi-layered authentication of the provided image to detect GenAI (Generative AI), deepfakes, or advanced digital manipulation.

    Perform a step-by-step forensic "Stress Test" on the image using these specialized criteria:

    1. GLOBAL COHERENCE & PHYSICS:
    - Shadows & Reflections: Do reflections show the correct environment? Are shadows physically consistent with light sources (check for "detached" shadows)?
    - Perspective & Vanishing Points: Do the lines of the object and background converge at a logically consistent vanishing point?
    - Gravity & Depth: Do objects appear to "float" or intersect with surfaces in an impossible way (e.g., a phone merging with a hand)?

    2. TEXTURE & GENERATIVE ARTIFACTS:
    - High-Frequency Noise: Look for "Generative Sheen" (unnatural smoothness) or "Checkerboard Artifacts" common in GANs/Diffusion models.
    - Micro-Inconsistencies: Check for warped edges, "bleeding" colors, or textures that change density abruptly near a point of interest.
    - Cracks/Defects: Do cracks follow the physical stress properties of the material (glass vs. plastic), or do they look like a superimposed brush-stroke pattern?

    3. SEMANTIC & SYMBOLIC ACCURACY:
    - Text/Logos: Analyze brand logos or text. Are there subtle misspellings, warped characters, or "haloing" around the edges of the font?
    - Contextual Logic: Are background elements nonsensical (e.g., stairs leading nowhere, extra fingers, impossible clock faces)?

    4. EDGE & BLENDING ANALYSIS:
    - Transition Zones: Zoom into the boundary between the "damaged" area and the "original" area. Look for pixel-level discontinuities, blurring masks, or "double-edges" indicating a composite.

    STRICT OUTPUT REQUIREMENT:
    Return ONLY a valid JSON object with the following keys:
    {
    "is_authentic": boolean,
    "damage_status": "authentic_damage" | "ai_generated_damage" | "edited_damage" | "no_damage" | "fake_image_total",
    "confidence_score": float (0.00 to 1.00),
    "forensic_flags": [
        {
        "indicator": "string",
        "severity": "high" | "medium" | "low",
        "observation": "detailed technical description"
        }
    ],
    "reasoning_summary": "string",
    "verdict": "Final definitive statement"
    }
    """

    try:
        response = model.generate_content([prompt, image])
        text_response = response.text
        
        # Clean up code blocks if present
        if text_response.startswith("```json"):
            text_response = text_response[7:]
        if text_response.startswith("```"):
            text_response = text_response[3:]
        if text_response.endswith("```"):
            text_response = text_response[:-3]
            
        # Parse JSON to ensure it's valid
        try:
            json_response = json.loads(text_response)
            return json_response
        except json.JSONDecodeError:
            # Fallback if raw text isn't valid JSON, return as structure
            return {
                "is_authentic": False,
                "damage_status": "error_parsing_response",
                "confidence_score": 0.0,
                "reasoning": ["Failed to parse AI response as JSON."],
                "verdict": "Internal parsing error",
                "raw_response": text_response
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
