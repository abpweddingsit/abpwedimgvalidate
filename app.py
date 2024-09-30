from flask import Flask, request, jsonify
from main import get_result

app = Flask(__name__)

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.json
    image_url = data.get('image_url')

    if not image_url:
        return jsonify({"error": "Image URL is required"}), 400

    # Get results from main.py
    result = get_result(image_url)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0')