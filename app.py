from flask import Flask, render_template, request, jsonify
from response_model.call_language_model import query_model, get_random_msg
import base64
import os
import time
from response_model.query_gemini import get_gemini_response

app = Flask(__name__)

# Directory to save uploaded images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_message', methods=['GET'])
def get_message():
    # Get a random message as the trigger message
    random_msg = get_random_msg()
    # random_msg = 'Lets go running today!'
    gemini = True

    if gemini:
        response = get_gemini_response(emotion='angry',context = random_msg)
    else:
        # Use the language model to generate a response with neutral emotion
        response = query_model(emotion='angry', text_message=random_msg)

    return jsonify({
        "message": response,
        "original_message": random_msg,
        "emotion": "neutral"  # For now, we're using neutral emotion
    })


@app.route('/capture_photo', methods=['POST'])
def capture_photo():
    data = request.json
    if data and 'image' in data:
        # Remove the 'data:image/jpeg;base64,' prefix
        image_data = data['image'].split(',')[1]

        # Convert the base64 string to bytes
        image_bytes = base64.b64decode(image_data)

        # Save the image with timestamp to avoid overwrites
        timestamp = int(time.time())
        image_filename = os.path.join(UPLOAD_FOLDER, f'user_image_{timestamp}.jpg')
        with open(image_filename, 'wb') as f:
            f.write(image_bytes)

        return jsonify({"success": True, "filename": image_filename}), 200
    return jsonify({"success": False, "error": "No image data received"}), 400


if __name__ == "__main__":
    app.run(debug=True)