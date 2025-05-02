from flask import Flask, render_template, request, jsonify
from response_model.call_language_model import query_model, get_random_msg
import base64
import os
import time
from response_model.query_gemini import get_gemini_response
from JG_models.Webcam_thing.imagizer_modern import predict_emotion_from_image_path

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
    random_msg = get_random_msg() ### returns a string random message.

    return jsonify({
        "message": random_msg,
        "original_message": random_msg,
        "user_message": 'Analyzing your facial reaction to the message...'
    })

@app.route('/get_response',methods=['GET'])
def return_response(img,message): ### gets an image path from the front end and the random message displayed from /get_message and returns the predicted emotion, emotion probs, and the response
    emotion,emotion_class_probs = predict_emotion_from_image_path(img)

    gemini = True ### want this to be a user toggle

    if gemini:
        response = get_gemini_response(emotion=emotion,context = message) ### returns a string
    else:
        response = query_model(emotion='angry', text_message=message) ### returns a string

    return jsonify({
        "message": response,
        "original_message": message,
        "emotion": emotion,
        "emotion_class_probs": emotion_class_probs
    })


@app.route('/capture_photo', methods=['POST'])
def capture_photo():
    data = request.json
    if data and 'image' in data:
        image_data = data['image'].split(',')[1]

        image_bytes = base64.b64decode(image_data)

        timestamp = int(time.time())
        image_filename = os.path.join(UPLOAD_FOLDER, f'user_image_{timestamp}.jpg')
        with open(image_filename, 'wb') as f:
            f.write(image_bytes)

        return jsonify({"success": True, "filename": image_filename}), 200
    return jsonify({"success": False, "error": "No image data received"}), 400


if __name__ == "__main__":
    app.run(debug=True)