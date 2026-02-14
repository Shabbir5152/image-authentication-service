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
    Act as a forensic image analyst specialized in detecting digital manipulation and AI-generated content (Deepfakes/GenAI).
    
    Your task is to CRITICALLY analyze this image of a product to determine if the damage shown is REAL or FAKE/AI-GENERATED.
    
    Be extremely skeptical. Look for the following specific indicators of AI generation or editing:
    1. **Inconsistent Lighting/Shadows**: Do shadows match the light sources? Are they too soft or in the wrong direction?
    2. **Unnatural Textures**: Does the "damage" look like a flat texture pasted on? Is the screen crack too perfect or following a weird pattern?
    3. **Impossible Geometry**: Do lines connect logically? Are there warping or bending artifacts near the damage?
    4. **Text/Logo Artifacts**: If there is text or a logo, is it gibberish, misspelled, or blurry?
    5. **Surface Blending**: Does the damaged area blend naturally with the surrounding surface, or does it look like a separate layer?
    
    Analyze the image step-by-step.
    
    Return a VALID JSON response with:
    - "is_authentic": boolean (true ONLY if you are 100% sure it's real physical damage. If unsure or looks AI, false)
    - "damage_status": string ("authentic_damage", "ai_generated_damage", "edited_damage", "no_damage")
    - "confidence_score": float (0.0 to 1.0, where 1.0 is absolute certainty)
    - "reasoning": list of strings (specific observations that led to your conclusion)
    - "verdict": string (a short summary of why it's fake or real)
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
