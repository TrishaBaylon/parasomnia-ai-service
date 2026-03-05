import firebase_admin
from firebase_admin import credentials, db
import tensorflow as tf
import numpy as np
import joblib
import random
import time

# =====================================
# SETTINGS
# =====================================

TEST_MODE = False        # True = simulate data
TEST_SCENARIO = "LOW"    # NORMAL / LOW / HIGH (only used if TEST_MODE=True)

WINDOW_SIZE = 60

# =====================================
# LOAD MODEL AND SCALER
# =====================================

print("Loading LSTM model...")
model = tf.keras.models.load_model("parasomnia_lstm.keras")

print("Loading scaler...")
scaler = joblib.load("scaler.save")

# =====================================
# FIREBASE INIT
# =====================================

cred = credentials.Certificate("serviceAccountKey.json")

firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://sleepguard-8eb64-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

sensor_ref = db.reference("device1")
risk_ref = db.reference("active_user/risk_status")

print("Connected to Firebase.")

# =====================================
# POSTURE MAP
# =====================================

posture_map = {
    "Resting": 0,
    "Moving": 1,
    "Standing": 2
}

# =====================================
# WINDOW STORAGE
# =====================================

window = []

# =====================================
# RISK CALCULATION FUNCTION
# =====================================

def compute_risk_score(prediction, window):

    normal_prob = prediction[0]
    low_prob = prediction[1]
    high_prob = prediction[2]

    # Convert scaled values back to approximate scale
    bpm_values = [x[0] for x in window]
    posture_values = [x[2] for x in window]

    avg_bpm = np.mean(bpm_values)
    movement = np.mean(posture_values)

    # Physiological component
    physio_score = (avg_bpm * 40) + (movement * 60)

    # Model component
    model_score = (low_prob * 40) + (high_prob * 100)

    risk_score = round((physio_score * 0.4) + (model_score * 0.6))

    risk_score = max(0, min(100, risk_score))

    # Status from model
    class_index = np.argmax(prediction)

    if class_index == 0:
        status = "NORMAL"
    elif class_index == 1:
        status = "LOW RISK"
    else:
        status = "HIGH RISK"

    return risk_score, status, normal_prob, low_prob, high_prob

# =====================================
# TEST MODE DATA GENERATOR
# =====================================

def generate_test_sample():

    base_bpm = random.uniform(65, 85)

    if TEST_SCENARIO == "NORMAL":

        bpm = base_bpm + random.uniform(-5, 5)
        avgBPM = base_bpm + random.uniform(-3, 3)
        posture = random.choices([0,1], weights=[0.85,0.15])[0]

    elif TEST_SCENARIO == "LOW":

        if random.random() < 0.4:
            bpm = base_bpm + random.uniform(8, 15)
            posture = 1
        else:
            bpm = base_bpm + random.uniform(-3, 5)
            posture = random.choices([0,1], weights=[0.6,0.4])[0]

        avgBPM = base_bpm + random.uniform(3,10)

    else:

        if random.random() < 0.7:
            bpm = base_bpm + random.uniform(18, 30)
            posture = 1
        else:
            bpm = base_bpm + random.uniform(5,15)
            posture = random.choices([0,1], weights=[0.3,0.7])[0]

        avgBPM = base_bpm + random.uniform(10,20)

    return bpm, avgBPM, posture

# =====================================
# REALTIME LISTENER
# =====================================

def listener(event):

    global window

    data = event.data

    if not data:
        return

    bpm = data.get("bpm")
    avgBPM = data.get("avgBPM")
    posture = data.get("posture")

    if bpm is None or avgBPM is None or posture is None:
        return

    try:
        bpm = float(bpm)
        avgBPM = float(avgBPM)
    except:
        print("Invalid sensor data")
        return

    posture_encoded = posture_map.get(posture, 0)

    sample = np.array([[bpm, avgBPM, posture_encoded]])
    sample_scaled = scaler.transform(sample)

    window.append(sample_scaled[0])

    print(f"Collected {len(window)}/{WINDOW_SIZE}")

    if len(window) >= WINDOW_SIZE:

        input_data = np.array([window])
        prediction = model.predict(input_data, verbose=0)[0]

        risk_score, status, normal, low, high = compute_risk_score(prediction, window)

        risk_ref.set({
            "risk_score": risk_score,
            "risk_status": status,
            "normal_probability": round(normal*100),
            "low_probability": round(low*100),
            "high_probability": round(high*100)
        })

        print("\n==============================")
        print("Prediction Complete")
        print("Normal:", round(normal*100))
        print("Low:", round(low*100))
        print("High:", round(high*100))
        print("Risk Score:", risk_score)
        print("Status:", status)
        print("==============================\n")

        window.clear()

# =====================================
# TEST MODE LOOP
# =====================================

def run_test_mode():

    global window

    print("\nRunning TEST MODE:", TEST_SCENARIO)

    while True:

        bpm, avgBPM, posture = generate_test_sample()

        sample = np.array([[bpm, avgBPM, posture]])
        sample_scaled = scaler.transform(sample)

        window.append(sample_scaled[0])

        print(f"Simulated {len(window)}/60")

        if len(window) >= WINDOW_SIZE:

            input_data = np.array([window])
            prediction = model.predict(input_data, verbose=0)[0]

            risk_score, status, normal, low, high = compute_risk_score(prediction, window)

            risk_ref.set({
                "risk_score": risk_score,
                "risk_status": status,
                "normal_probability": round(normal*100),
                "low_probability": round(low*100),
                "high_probability": round(high*100)
            })

            print("\n==============================")
            print("Prediction Uploaded")
            print("Normal:", round(normal*100))
            print("Low:", round(low*100))
            print("High:", round(high*100))
            print("Risk Score:", risk_score)
            print("Status:", status)
            print("==============================\n")

            window.clear()

            time.sleep(5)

# =====================================
# START SYSTEM
# =====================================

print("System Ready")

if TEST_MODE:

    run_test_mode()

else:

    print("Listening for ESP32 data...")
    sensor_ref.listen(listener)