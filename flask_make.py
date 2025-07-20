from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import json

app = Flask(__name__)

df = pd.read_csv("database.csv")
driver_2025 = ['hamilton','alonso','hulkenberg','verstappen','sainz','ocon','stroll','leclerc','norris','russell','albon','tsunoda','gasly','piastri','lawson','bearman','colapinto','doohan']
circuit_2025 =['Australian Grand Prix','Chinese Grand Prix','Japanese Grand Prix','Bahrain Grand Prix','Saudi Arabian Grand Prix','Miami Grand Prix','Emilia Romagna Grand Prix','Monaco Grand Prix','Spanish Grand Prix','Canadian Grand Prix','Austrian Grand Prix','British Grand Prix','Belgian Grand Prix','Hungarian Grand Prix','Dutch Grand Prix','Italian Grand Prix','Azerbaijan Grand Prix','Singapore Grand Prix','United States Grand Prix','Mexico City Grand Prix','São Paulo Grand Prix','Las Vegas Grand Prix','Qatar Grand Prix','Abu Dhabi Grand Prix']

unique_drivers = sorted(df["driverRef"].dropna().unique())
unique_circuits = sorted(df["race_name"].dropna().unique())
pipeline_p = joblib.load(r"D:\summer 2_nd\Formula 1\f1_preprocessing_pipeline.pkl")
model = joblib.load(r"D:\summer 2_nd\Formula 1\model.pkl")

@app.route('/')
def hello():
    return render_template("index.html", drivers=unique_drivers, circuits=unique_circuits, circuits_2025=circuit_2025,drivers_2025=driver_2025)

@app.route("/predict-driver", methods=["POST"])
def predict_driver():
    def time_to_seconds(t):
        if not t:
            return None
        try:
            mins, secs = t.split(":")
            return int(mins) * 60 + float(secs)
        except:
            return None

    driver = request.form.get("driver")
    circuit = request.form.get("circuit")
    q1 = request.form.get("q1")
    q2 = request.form.get("q2")
    q3 = request.form.get("q3")
    grid_position = request.form.get("grid_position")
    use_previous = request.form.get("use_previous")  

    print("Use Previous:", use_previous)

    if use_previous==None:  
        print("processing")
        q1_sec = time_to_seconds(q1)
        q2_sec = time_to_seconds(q2)
        q3_sec = time_to_seconds(q3)

        print("Driver:", driver)
        print("Circuit:", circuit)
        print("Q1:", q1_sec)
        print("Q2:", q2_sec)
        print("Q3:", q3_sec)
        print("Grid Position:", grid_position)

        extracted_row = df[(df["driverRef"] == driver) & (df["race_name"] == circuit)]
        extracted_row["q1"] = q1_sec
        extracted_row["q2"] = q2_sec
        extracted_row["q3"] = q3_sec
        extracted_row["result_position"] = grid_position
        processed_row = pipeline_p.transform(extracted_row)
        podiumm = int(model.predict(processed_row)[0])
        confidence = float(max(model.predict_proba(processed_row)[0]))

      
    
        return jsonify({
            "driver": driver,
            "circuit": circuit,
            "grid": grid_position,
            "podium": podiumm,
            "confidence": confidence
        })

    else:
        extracted_row = df[(df["driverRef"] == driver) & (df["race_name"] == circuit)]
        processed_row = pipeline_p.transform(extracted_row)
        podiumm = int(model.predict(processed_row)[0])
        confidence = float(max(model.predict_proba(processed_row)[0]))
        return jsonify({
            "driver": driver,
            "circuit": circuit,
            "grid": grid_position,
            "podium": podiumm,
            "confidence": confidence
        })
@app.route("/full_race", methods=["POST"])
def full_race():
    def time_to_seconds(t):
        if not t:
            return None
        try:
            mins, secs = t.split(":")
            return int(mins) * 60 + float(secs)
        except:
            return None

    circuit = request.form.get("race_circuit")
    use_previous = request.form.get("use_previous_full")

    predictions = []

    for driver in driver_2025:
        q1 = request.form.get(f"q1_{driver}")
        q2 = request.form.get(f"q2_{driver}")
        q3 = request.form.get(f"q3_{driver}")
        grid = request.form.get(f"grid_{driver}")

        extracted_row = df[(df["driverRef"] == driver) & (df["race_name"] == circuit)]

        if extracted_row.empty:
            continue

        if use_previous is None:
            extracted_row["q1"] = time_to_seconds(q1)
            extracted_row["q2"] = time_to_seconds(q2)
            extracted_row["q3"] = time_to_seconds(q3)
            extracted_row["result_position"] = grid

        processed_row = pipeline_p.transform(extracted_row)
        podium = int(model.predict(processed_row)[0])
        confidence = float(max(model.predict_proba(processed_row)[0]))

        predictions.append({
            "driver": driver,
            "podium": podium,
            "confidence": confidence,
            "grid": grid
        })

    

    # 🖨️ Log predictions to console
    print("Full Race Predictions (Backend):")
    for p in predictions:
        print(f"Driver: {p['driver']}, Grid: {p['grid']}, Podium: {p['podium']}, Confidence: {p['confidence']:.4f}")

    return jsonify({"predictions": predictions})

    return "hello"

if __name__ == "__main__":
    app.run(debug=True)
