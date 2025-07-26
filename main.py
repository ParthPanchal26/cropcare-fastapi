from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import uvicorn
import requests
from typing import Optional
import google.generativeai as genai
from dotenv import load_dotenv
import google.generativeai as genai
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing. Set it in environment variables.")

genai.configure(api_key=GEMINI_API_KEY)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
      allow_origins=[
        "http://localhost:5173",
        "https://parthpanchal26.github.io"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models = {
    "vit": {
        "name": "wambugu71/crop_leaf_diseases_vit",
        "processor": AutoImageProcessor.from_pretrained("wambugu71/crop_leaf_diseases_vit"),
        "model": AutoModelForImageClassification.from_pretrained("wambugu71/crop_leaf_diseases_vit")
    },
    "mobilenet": {
        "name": "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
        "processor": AutoImageProcessor.from_pretrained("linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"),
        "model": AutoModelForImageClassification.from_pretrained("linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification")
    }
}

def read_image(file) -> Image.Image:
    return Image.open(BytesIO(file)).convert("RGB")

def predict_with_model(image: Image.Image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        confidence = torch.softmax(logits, dim=1)[0][predicted_class_idx].item()
        label = model.config.id2label[predicted_class_idx]
        return label, confidence

def explain_plant_disease(predicted_label: str, user_text: str = None) -> str:    
    prompt = (f"""
        You are an expert in plant health and crop disease management.

        The detected plant disease is: '{predicted_label}'.
        The user said: '{user_text}'.

        Provide a detailed, well-structured explanation in **clear sections** using Markdown format:

        ### 1. Disease Name
        State the identified disease name clearly.

        ### 2. About the Disease
        Explain what this disease is, what causes it, and why it occurs.

        ### 3. Other Symptoms to Watch For
        List other common symptoms that can help identify this disease.

        ### 4. Practical Solutions
        Provide clear, actionable steps farmers can follow:
        - Chemical control methods (with examples of safe fungicides/pesticides)
        - Organic or natural remedies
        - Preventive measures for future

        ### 5. Answer to User Query
        If the user asked any specific question, answer it in a simple and clear manner.

        Keep the language **farmer-friendly**, avoid unnecessary jargon, and make it **practical and easy to implement** and always answer in english by default.
    """)

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error fetching explanation: {e}"


@app.post("/upload")
async def predict(file: UploadFile = File(...), text: str = Form(None)):
    try:
        image = read_image(await file.read())
        
        results = []
        for model_name, data in models.items():
            label, confidence = predict_with_model(image, data["processor"], data["model"])
            results.append({
                "model": model_name,
                "label": label,
                "confidence": round(confidence * 100, 2)
            })

        best = max(results, key=lambda x: x["confidence"])
        explanation = explain_plant_disease(best["label"], text)

        return {
            "status": "success",
            "user_text": text,
            "prediction": best,
            "all_predictions": results,
            "explanation": explanation
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)