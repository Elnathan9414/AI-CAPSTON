import requests
from flask import Flask, jsonify, render_template, request

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
#API_URL = "http://15.222.64.78:8000/predict"   # your model API (remote server)
API_URL = "http://localhost:8000/predict"
API_FILE_FIELD = "file"                     # multipart field name expected by the API
API_TIMEOUT = 60                            # seconds

# Class names in the API's order (only used if the API returns a plain list of numbers)
CLASS_NAMES = [
    "normal",
    "polyps",
    "esophagitis",
    "ulcerative-colitis",
    "dyed-lifted-polyps",
    "dyed-resection-margins",
]

# Display names shown on the page
LABELS = {
    "normal": "Normal",
    "polyps": "Polyps",
    "esophagitis": "Esophagitis",
    "ulcerative-colitis": "Ulcerative colitis",
    "dyed-lifted-polyps": "Dyed lifted polyps",
    "dyed-resection-margins": "Dyed resection margins",
}

MAX_UPLOAD_MB = 10

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


# ---------------------------------------------------------------------------
# Adapter: turn the API response into what the web page expects.
# If your API returns a different JSON shape, this is the ONLY function
# you need to edit.
#
# Expected output:
#   {"prediction": str, "confidence": float,
#    "results": [{"label": str, "probability": float, "heatmap": str | None}, ...]}
# ---------------------------------------------------------------------------
def normalize(data: dict) -> dict:
    probs = data.get("probabilities") or data.get("probs") or data.get("scores")

    if isinstance(probs, dict):                       # {"Normal": 0.9, ...}
        items = [{"label": k, "probability": float(v)} for k, v in probs.items()]
    elif isinstance(probs, list) and probs and isinstance(probs[0], dict):
        # [{"label": "Normal", "probability": 0.9}, ...]
        items = [{"label": p["label"], "probability": float(p["probability"])} for p in probs]
    elif isinstance(probs, list):                     # [0.1, 0.9, ...]
        items = [{"label": n, "probability": float(p)} for n, p in zip(CLASS_NAMES, probs)]
    elif "predicted_class" in data:
        # The API only returns the top class and its confidence (no full distribution)
        confidence = float(str(data.get("confidence", 0)).strip("% "))
        items = [{"label": str(data["predicted_class"]), "probability": confidence}]
    else:
        raise ValueError(f"Unexpected response format. Keys received: {list(data.keys())}")

    # Some APIs return percentages (0-100) instead of probabilities (0-1)
    if sum(i["probability"] for i in items) > 1.5:
        for i in items:
            i["probability"] /= 100

    items.sort(key=lambda i: i["probability"], reverse=True)

    # Optional Grad-CAM: base64 PNG or data URL.
    #   "gradcams": {class_name: heatmap}  -> one heatmap per class
    #   "gradcam" / "heatmap"              -> heatmap of the predicted class only
    def as_data_url(value):
        if not isinstance(value, str) or not value:
            return None
        return value if value.startswith("data:") else "data:image/png;base64," + value

    per_class = data.get("gradcams") if isinstance(data.get("gradcams"), dict) else {}
    single = as_data_url(data.get("gradcam") or data.get("heatmap") or data.get("grad_cam"))
    for rank, item in enumerate(items):
        item["heatmap"] = as_data_url(per_class.get(item["label"])) or (single if rank == 0 else None)
        item["label"] = LABELS.get(item["label"], item["label"])

    return {
        "prediction": items[0]["label"],
        "confidence": items[0]["probability"],
        "results": items,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    uploaded = request.files.get("image")
    if uploaded is None:
        return jsonify(error="No image received."), 400

    try:
        api_response = requests.post(
            API_URL,
            files={API_FILE_FIELD: (uploaded.filename, uploaded.stream, uploaded.mimetype)},
            headers={"accept": "application/json"},
            timeout=API_TIMEOUT,
        )
    except requests.ConnectionError:
        return jsonify(error=f"API unreachable ({API_URL}). Is it running?"), 502
    except requests.Timeout:
        return jsonify(error="The API took too long to respond."), 504

    if not api_response.ok:
        return jsonify(error=f"API error (status code {api_response.status_code})."), 502

    try:
        return jsonify(normalize(api_response.json()))
    except Exception as exc:
        app.logger.error("Unexpected API response: %s | %s", exc, api_response.text[:500])
        return jsonify(error=str(exc)), 502


if __name__ == "__main__":
    app.run(debug=True)