import os
from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
import io

# Assuming 'second.py' is in the same directory and contains EnhancedPDFImageExtractor
from second import EnhancedPDFImageExtractor

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize your extractor
# You can configure debug_mode and use_clip here
extractor = EnhancedPDFImageExtractor(debug_mode=False, use_clip=False)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/extract_image', methods=['POST'])
def extract_image():
    # --- Add these lines for debugging ---
    print("Received request.form keys:", list(request.form.keys()))
    print("Received request.files keys:", list(request.files.keys()))
    # ------------------------------------

    if 'pdf_file' not in request.files:
        return jsonify({"error": "No pdf_file part in the request"}), 400
    file = request.files['pdf_file']
    query = request.form.get('query')

    if not query:
        return jsonify({"error": "No query string provided"}), 400

    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "No selected file or file type not allowed (only .pdf)"}), 400

    if file:
        filename = secure_filename(file.filename)
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(pdf_path)

        # Use a temporary path for the output image
        output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], "extracted_image.png")

        extracted_image = extractor.find_image_for_query(pdf_path, query, output_image_path)

        if extracted_image:
            return send_file(output_image_path, mimetype='image/png')
        else:
            return jsonify({"error": "No suitable image found for the query"}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)