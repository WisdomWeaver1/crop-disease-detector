import gradio as gr
import tensorflow as tf
import requests
import json
import os
import tempfile
from PIL import Image
import numpy as np

# ------------------- 1. CONFIGURE GITHUB SOURCE -------------------
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/WisdomWeaver1/crop-disease-detector/main/model/"
MODEL_URL = GITHUB_RAW_BASE + "crop_disease_model.h5"
CLASS_NAMES_URL = GITHUB_RAW_BASE + "class_names.json"
TREATMENTS_URL = GITHUB_RAW_BASE + "treatments.json"

# ------------------- 2. DOWNLOAD FILES (WITH CACHING) -------------------
def download_file(url, cache_name):
    """Download a file from a URL and save it locally to avoid re-downloading."""
    if not os.path.exists(cache_name):
        print(f"Downloading {cache_name}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(cache_name, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {cache_name}.")
    else:
        print(f"Using cached {cache_name}.")
    return cache_name

# Download class names and treatments (JSON files are small)
with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp_class:
    download_file(CLASS_NAMES_URL, tmp_class.name)
    with open(tmp_class.name, 'r') as f:
        CLASS_NAMES = json.load(f)

with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp_treat:
    download_file(TREATMENTS_URL, tmp_treat.name)
    with open(tmp_treat.name, 'r') as f:
        TREATMENTS = json.load(f)

# Download and load the Keras model (larger file)
model_path = download_file(MODEL_URL, "crop_disease_model.h5")
model = tf.keras.models.load_model(model_path)

# ------------------- 3. PREPROCESSING -------------------
# According to config.json, the model expects 96x96 images
IMG_SIZE = 96

def preprocess_image(img):
    """Resize image to 96x96 and normalize pixel values."""
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0  # Normalize to [0,1]
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ------------------- 4. PREDICTION FUNCTION -------------------
def predict(img, top_k=3):
    """Return top-k predictions with crop, disease, confidence, and treatment."""
    # Preprocess the image
    img_array = preprocess_image(img)
    
    # Get model predictions
    predictions = model.predict(img_array)[0]
    
    # Get top-k indices and probabilities
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    top_probs = predictions[top_indices]
    
    results = []
    for idx, prob in zip(top_indices, top_probs):
        class_name = CLASS_NAMES[idx]
        # Parse class name (format: "Crop___Disease")
        parts = class_name.split('___')
        crop = parts[0].replace('_', ' ')
        disease = parts[1].replace('_', ' ') if len(parts) > 1 else "healthy"
        
        # Get treatment (default if not found)
        treatment = TREATMENTS.get(class_name, "Consult a local agronomist.")
        
        results.append({
            'crop': crop,
            'disease': disease,
            'confidence': float(prob),
            'treatment': treatment
        })
    
    return results

# ------------------- 5. GRADIO INTERFACE -------------------
# Extract unique crops for the description
crop_list = sorted(set(c.split('___')[0].replace('_', ' ') for c in CLASS_NAMES))

def gradio_fn(img):
    if img is None:
        return 'Please upload a leaf photo.'
    
    res = predict(img, top_k=3)
    top = res[0]
    conf = top['confidence'] * 100
    warn = '\n⚠️ Low confidence — try a clearer, closer, well-lit photo.' if conf < 55 else ''
    
    out = (f"🌱 CROP     : {top['crop']}\n"
           f"🦠 DISEASE  : {top['disease']}\n"
           f"📊 CONFIDENCE: {conf:.1f}%{warn}\n\n"
           f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
           f"💊 TREATMENT:\n{top['treatment']}\n"
           f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\nOther possibilities:\n")
    
    for r in res[1:]:
        out += f"  • {r['crop']} — {r['disease']}: {r['confidence']*100:.1f}%\n"
    
    out += '\n⚠️ Always verify with a local agronomist before applying treatment.'
    return out

# Launch the app
gr.Interface(
    fn=gradio_fn,
    inputs=gr.Image(type='pil', label='Upload a leaf photo'),
    outputs=gr.Textbox(label='Diagnosis & Treatment', lines=16),
    title='🌿 Crop Disease Detector',
    description=f'Supported crops: {", ".join(crop_list)}',
    theme=gr.themes.Soft(),
    allow_flagging='never'
).launch(share=True)
