import os
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "models" / "plant_disease_model.keras"
IMG_SIZE = (224, 224)

st.set_page_config(page_title="AgriSense AI", page_icon="🌿", layout="wide")

DISEASE_INFO = {
    "Tomato___Early_blight": {
        "crop": "Tomato", "disease": "Early blight",
        "symptoms": "Possible dark leaf spots with concentric rings and yellowing around affected areas.",
        "actions": [
            "Remove badly affected leaves using clean tools and dispose of them safely.",
            "Avoid overhead watering; water near the soil and allow foliage to dry.",
            "Improve spacing and airflow, and avoid handling plants while foliage is wet.",
            "Confirm the diagnosis locally before using any plant-protection product."
        ],
        "risk": "Can spread faster in warm, humid conditions with prolonged leaf wetness."
    },
    "Tomato___Late_blight": {
        "crop": "Tomato", "disease": "Late blight",
        "symptoms": "Possible irregular water-soaked lesions that can turn brown; rapid spread may occur in favorable weather.",
        "actions": [
            "Isolate affected plants where practical and avoid moving wet plant material through the crop.",
            "Avoid overhead irrigation and inspect nearby plants for new symptoms.",
            "Seek prompt advice from a local agricultural extension officer because this disease can progress quickly.",
            "Use only locally approved products and follow their label and pre-harvest instructions."
        ],
        "risk": "Cool-to-mild, wet, humid weather can favor disease development."
    },
    "Tomato___healthy": {
        "crop": "Tomato", "disease": "Healthy (model prediction)",
        "symptoms": "The model did not detect one of its trained tomato disease classes.",
        "actions": [
            "Continue routine scouting, balanced irrigation, and good crop hygiene.",
            "Check both sides of leaves and inspect new growth regularly.",
            "A healthy prediction does not rule out pests, nutrient problems, or diseases absent from training data."
        ],
        "risk": "Continue monitoring; the model can miss unfamiliar or early-stage symptoms."
    },
    "Potato___Early_blight": {
        "crop": "Potato", "disease": "Early blight",
        "symptoms": "Possible brown leaf lesions, sometimes with target-like rings and surrounding yellow tissue.",
        "actions": [
            "Remove heavily affected foliage where appropriate and maintain field sanitation.",
            "Avoid prolonged leaf wetness and improve airflow.",
            "Review crop rotation and local extension recommendations.",
            "Confirm the diagnosis before treatment."
        ],
        "risk": "Warm conditions, plant stress, and leaf wetness can increase risk."
    },
    "Potato___Late_blight": {
        "crop": "Potato", "disease": "Late blight",
        "symptoms": "Possible rapidly enlarging dark lesions and, in humid weather, pale growth at lesion edges.",
        "actions": [
            "Inspect surrounding plants and seek prompt local agricultural advice.",
            "Avoid moving potentially infected plant material between fields.",
            "Reduce leaf wetness where possible and follow locally approved management guidance.",
            "Use only approved plant-protection products according to their label."
        ],
        "risk": "Cool, humid, wet weather can favor outbreaks."
    },
    "Potato___healthy": {
        "crop": "Potato", "disease": "Healthy (model prediction)",
        "symptoms": "The model did not detect one of its trained potato disease classes.",
        "actions": [
            "Continue regular scouting and maintain good crop hygiene.",
            "Monitor for changes in leaf color, spots, wilting, or pest damage.",
            "A healthy prediction does not exclude conditions the model has not been trained to recognize."
        ],
        "risk": "Continue monitoring and consult an expert if symptoms appear."
    },
    "Pepper,_bell___Bacterial_spot": {
        "crop": "Bell pepper", "disease": "Bacterial spot",
        "symptoms": "Possible small dark spots on leaves and fruit; symptoms can resemble other conditions.",
        "actions": [
            "Avoid working among wet plants and sanitize tools.",
            "Remove badly affected material if recommended by local guidance.",
            "Use clean seed and healthy transplants, and seek a local diagnosis.",
            "Do not apply treatments based on an AI prediction alone."
        ],
        "risk": "Warm, wet conditions can favor spread of bacterial diseases."
    },
    "Pepper,_bell___healthy": {
        "crop": "Bell pepper", "disease": "Healthy (model prediction)",
        "symptoms": "The model did not detect one of its trained bell pepper disease classes.",
        "actions": [
            "Continue routine scouting and balanced watering.",
            "Inspect leaf undersides and new growth regularly.",
            "Seek advice if symptoms appear despite this prediction."
        ],
        "risk": "A healthy prediction is limited to the classes represented during training."
    }
}

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return tf.keras.models.load_model(MODEL_PATH)

def get_weather(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "timezone": "auto", "forecast_days": 3
    }
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    return response.json()

def weather_summary(data):
    current = data.get("current", {})
    humidity = current.get("relative_humidity_2m")
    rain = current.get("precipitation")
    temp = current.get("temperature_2m")
    precip_probs = data.get("daily", {}).get("precipitation_probability_max", [])
    high_rain_forecast = max(precip_probs) if precip_probs else None
    notes = []
    if humidity is not None and humidity >= 80:
        notes.append("High relative humidity may favor some plant diseases, especially when leaves remain wet.")
    if rain is not None and rain > 0:
        notes.append("Precipitation is currently reported; avoid unnecessary leaf wetness and inspect crops after rain.")
    if high_rain_forecast is not None and high_rain_forecast >= 60:
        notes.append(f"Rain probability reaches {high_rain_forecast}% in the displayed forecast; plan field scouting and irrigation accordingly.")
    if not notes:
        notes.append("No simple high-humidity or high-rain flag was triggered. Weather alone cannot determine disease presence.")
    return temp, humidity, rain, high_rain_forecast, notes

st.title("🌿 AgriSense AI")
st.subheader("Weather-Aware Plant Doctor")
st.write("Upload a plant-leaf image to get a model prediction, practical crop-care guidance, and a weather-aware scouting summary.")

st.warning("Educational decision-support demo only. Image predictions can be wrong. Weather indicators are general heuristics, not validated disease forecasts. Confirm important diagnoses with a qualified agricultural expert or local extension service.")

model = load_model()
if model is None:
    st.error("The trained model file is not present yet.")
    st.markdown("""
    **Set up the model first:**
    1. Download the PlantVillage dataset from [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease) or another properly licensed source.
    2. Extract the dataset so folders are organized by class.
    3. Run `python train_model.py --data_dir path/to/PlantVillage --epochs 5`.
    4. The training script saves `models/plant_disease_model.keras` and `models/class_names.json`.
    5. Refresh this app.
    """)
    st.stop()

class_names_path = APP_DIR / "models" / "class_names.json"
import json
with open(class_names_path, "r", encoding="utf-8") as f:
    class_names = json.load(f)

left, right = st.columns([1, 1])
with left:
    uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded leaf image", use_container_width=True)
        if st.button("Analyze leaf", type="primary"):
            resized = image.resize(IMG_SIZE)
            array = np.asarray(resized, dtype=np.float32)  # Model applies its own Rescaling layer
            probabilities = model.predict(np.expand_dims(array, 0), verbose=0)[0]
            order = np.argsort(probabilities)[::-1]
            best_idx = int(order[0])
            predicted_class = class_names[best_idx]
            confidence = float(probabilities[best_idx])
            st.session_state["prediction"] = {
                "class": predicted_class, "confidence": confidence,
                "top": [(class_names[int(i)], float(probabilities[int(i)])) for i in order[:3]]
            }

with right:
    st.markdown("### Weather location")
    st.write("Enter coordinates for the farm or nearby town. Example coordinates are for a demonstration only.")
    lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=18.1067, step=0.01, format="%.4f")
    lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=83.3956, step=0.01, format="%.4f")
    if st.button("Get current weather and forecast"):
        try:
            weather = get_weather(lat, lon)
            st.session_state["weather"] = weather
            st.success("Weather loaded.")
        except requests.RequestException as exc:
            st.error(f"Could not load weather. Check your internet connection and coordinates. Details: {exc}")

if "prediction" in st.session_state:
    prediction = st.session_state["prediction"]
    class_key = prediction["class"]
    details = DISEASE_INFO.get(class_key, {
        "crop": class_key.split("___")[0].replace("_", " "),
        "disease": class_key.split("___")[-1].replace("_", " "),
        "symptoms": "No crop-specific guidance is configured for this model class.",
        "actions": ["Treat the result as a preliminary screening result.", "Consult a local agricultural expert before making treatment decisions."],
        "risk": "No class-specific weather interpretation is configured."
    })
    st.divider()
    st.header("Analysis results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted crop", details["crop"])
    c2.metric("Predicted condition", details["disease"])
    c3.metric("Model score", f"{prediction['confidence']:.1%}")
    if prediction["confidence"] < 0.60:
        st.warning("The model score is low. Treat this result as uncertain and seek another image or expert review.")
    st.write("**Possible visual symptoms:**", details["symptoms"])
    st.write("**Suggested next steps:**")
    for action in details["actions"]:
        st.write(f"- {action}")
    st.write("**General weather context:**", details["risk"])
    st.markdown("**Top 3 model outputs**")
    st.dataframe([{"Class": name, "Score": f"{score:.1%}"} for name, score in prediction["top"]], hide_index=True, use_container_width=True)
    st.caption("Model scores are not necessarily calibrated probabilities. A model can be confidently wrong on unfamiliar images.")

if "weather" in st.session_state:
    weather = st.session_state["weather"]
    temp, humidity, rain, rain_prob, notes = weather_summary(weather)
    st.divider()
    st.header("Weather-aware scouting")
    current = weather.get("current", {})
    w1, w2, w3, w4 = st.columns(4)
    w1.metric("Temperature", f"{temp} °C" if temp is not None else "N/A")
    w2.metric("Relative humidity", f"{humidity}%" if humidity is not None else "N/A")
    w3.metric("Current precipitation", f"{rain} mm" if rain is not None else "N/A")
    w4.metric("Max forecast rain chance", f"{rain_prob}%" if rain_prob is not None else "N/A")
    for note in notes:
        st.write(f"- {note}")
    daily = weather.get("daily", {})
    if daily.get("time"):
        st.markdown("**Three-day forecast**")
        rows = []
        for i, date in enumerate(daily["time"]):
            rows.append({
                "Date": date,
                "Min °C": daily.get("temperature_2m_min", [None]*len(daily["time"]))[i],
                "Max °C": daily.get("temperature_2m_max", [None]*len(daily["time"]))[i],
                "Rain chance %": daily.get("precipitation_probability_max", [None]*len(daily["time"]))[i]
            })
        st.dataframe(rows, hide_index=True, use_container_width=True)

st.divider()
st.caption("AgriSense AI • Prototype for education and portfolio demonstration • Weather supplied by Open-Meteo")
