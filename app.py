from flask import Flask, render_template, request, jsonify, send_file
import os
import subprocess
from main import train, generate

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    """Handle SVG dataset uploads."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename.endswith(".svg"):
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        return jsonify({"message": f"File {file.filename} uploaded successfully!"})
    else:
        return jsonify({"error": "Only SVG files are allowed"}), 400

@app.route("/train", methods=["POST"])
def start_training():
    """Start the training process."""
    try:
        epochs = int(request.form.get("epochs", 5))
        batch_size = int(request.form.get("batch_size", 8))
        lr = float(request.form.get("lr", 1e-4))

        train(data_dir=UPLOAD_FOLDER, output_dir=OUTPUT_FOLDER, epochs=epochs, batch_size=batch_size, lr=lr, device="cuda")
        return jsonify({"message": "Training completed successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/generate", methods=["POST"])
def generate_image():
    """Generate an image from a prompt."""
    try:
        prompt = request.form.get("prompt")
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        generate(prompt=prompt, output_dir=OUTPUT_FOLDER, device="cuda")
        return send_file(os.path.join(OUTPUT_FOLDER, "generated_image.png"), mimetype="image/png")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
