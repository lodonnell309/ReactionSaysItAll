from flask import Flask, render_template, request, jsonify, send_from_directory
import base64
import os
import time
import logging
import traceback
import sys
from JG_models.Webcam_thing.imagizer_modern import predict_emotion_from_image_path
from response_model.query_gemini import get_gemini_response
from response_model.call_language_model import query_model
import random

# Configure logging to show more detailed information
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Directory to save uploaded images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    logger.info(f"Created upload folder: {UPLOAD_FOLDER}")


@app.route('/')
def index():
    logger.info("Serving index page")
    return render_template('index.html')


@app.route('/get_message', methods=['GET'])
def get_message():
    logger.info("Received request for random message")
    try:
        # Import here to avoid issues if the module isn't available
        try:
            from response_model.call_language_model import get_random_msg
            # random_msg = get_random_msg()  # returns a string random message
            MESSAGES = [
                "Big news! Call me when you can.",
                "You’re not going to believe what just happened.",
                "I have a surprise for you 🎁",
                "Can we talk about your last presentation?",
                "Someone just mentioned you in a meeting…",
                "Check your inbox right now!",
                "Guess who’s back on the team?",
                "You crushed it today 👏",
                "Look what I found from college 😅",
                "Did you mean to send that email?",
                "I just recommended you for something big.",
                "We need to talk. Not bad, I promise!",
                "Look at this old photo I just found 😆",
                "I heard what you said—well done 👏",
                "Your LinkedIn post is blowing up!",
                "You’ve been nominated for something 🎉",
                "Someone just asked for your resume.",
                "Your project is going live today!",
                "This reminded me of you…",
                "You won’t believe who just joined our company!"
            ]

            # 2) Pick one at random whenever you need it
            random_msg = random.choice(MESSAGES)
            logger.info(f"Generated random message: {random_msg}")
        except Exception as e:
            logger.error(f"Error importing or calling get_random_msg: {str(e)}")
            logger.error(traceback.format_exc())
            random_msg = "How are you feeling today?"  # Fallback message

        return jsonify({
            "message": random_msg,
            "original_message": random_msg,
            "user_message": 'Analyzing your facial reaction to the message...'
        })
    except Exception as e:
        logger.error(f"Unexpected error in get_message: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "message": "How are you feeling today?",
            "original_message": "How are you feeling today?",
            "error": str(e)
        })


@app.route('/get_response', methods=['POST'])
def get_response():
    logger.info("Received request for response analysis")
    data = request.json or {}
    message = data.get('message') or data.get('original_message', '')
    # … your existing validation …

    # Grab the toggle
    use_gemini = bool(data.get('use_gemini', False))
    logger.info(f"Use Gemini? {use_gemini}")

    # emotion prediction … same as before …

    # Choose which LLM call

    try:
        data = request.json
        logger.debug(f"Request data: {data}")

        if not data:
            logger.error("No JSON data received")
            return jsonify({"error": "No data received"}), 400

        if 'image_path' not in data:
            logger.error("Missing image_path in request")
            return jsonify({"error": "Missing image_path"}), 400

        if 'message' not in data:
            logger.error("Missing message in request")
            return jsonify({"error": "Missing message"}), 400

        img_path = data['image_path']
        message = data['message']
        print('--------MESSAGE------------')
        print(message)
        print('---------------------------')

        logger.info(f"Processing image at path: {img_path}")
        logger.info(f"Message to respond to: {message}")

        # Verify file exists
        if not os.path.exists(img_path):
            logger.error(f"Image file not found: {img_path}")
            return jsonify({"error": f"Image file not found: {img_path}"}), 404

        # Check file size (to make sure it's not empty)
        file_size = os.path.getsize(img_path)
        logger.info(f"Image file size: {file_size} bytes")
        if file_size == 0:
            logger.error("Image file is empty")
            return jsonify({"error": "Image file is empty"}), 400

        try:
            # Predict emotion from the captured image
            logger.info("Importing emotion prediction module...")

            logger.info(f"Predicting emotion from image: {img_path}")
            emotion, emotion_class_probs = predict_emotion_from_image_path(img_path)
            logger.info(f"Predicted emotion: {emotion}")
            logger.debug(f"Emotion probabilities: {emotion_class_probs}")

            # Generate response based on emotion
            logger.info("Importing Gemini response module...")


            logger.info(f"Generating response with Gemini for emotion: {emotion}")
            if use_gemini:
                print('Here is the message:',message)
                response = get_gemini_response(emotion=emotion, context=message)
            else:
                response = query_model(emotion=emotion, text_message=message)
            logger.info(f"Generated response: {response}")

            # Convert emotion_class_probs to a format suitable for JSON
            formatted_probs = {}
            for emotion_name, prob in emotion_class_probs.items():
                formatted_probs[emotion_name] = float(prob)  # Ensure it's a Python float

            return jsonify({
                "message": response,
                "original_message": message,
                "emotion": emotion,
                "emotion_class_probs": formatted_probs
            })
        except ImportError as e:
            logger.error(f"Import error: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                "message": f"Missing required module: {str(e)}",
                "original_message": message,
                "emotion": "neutral",
                "emotion_class_probs": {"neutral": 1.0}
            }), 500
    except Exception as e:
        logger.error(f"Error in response generation: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "message": f"Sorry, there was an error analyzing your response: {str(e)}",
            "original_message": message if 'message' in locals() else "Unknown",
            "emotion": "error",
            "emotion_class_probs": {"error": 1.0}
        }), 500


@app.route('/capture_photo', methods=['POST'])
def capture_photo():
    logger.info("Received request to capture photo")
    try:
        data = request.json
        if not data:
            logger.error("No JSON data in request")
            return jsonify({"success": False, "error": "No data received"}), 400

        if 'image' not in data:
            logger.error("No image data in request")
            return jsonify({"success": False, "error": "No image data received"}), 400

        logger.info("Image data received, processing...")
        image_data = data['image']

        # Handle both prefixed and non-prefixed base64 data
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        try:
            image_bytes = base64.b64decode(image_data)
            logger.info(f"Successfully decoded base64 data, size: {len(image_bytes)} bytes")

            if len(image_bytes) < 100:
                logger.error(f"Image data too small: {len(image_bytes)} bytes")
                return jsonify({"success": False, "error": "Image data too small"}), 400
        except Exception as e:
            logger.error(f"Error decoding base64 data: {str(e)}")
            return jsonify({"success": False, "error": f"Invalid image data: {str(e)}"}), 400

        timestamp = int(time.time())
        image_filename = os.path.join(UPLOAD_FOLDER, f'user_image_{timestamp}.jpg')
        logger.info(f"Saving image to: {image_filename}")

        with open(image_filename, 'wb') as f:
            f.write(image_bytes)

        file_size = os.path.getsize(image_filename)
        logger.info(f"Image saved successfully, size: {file_size} bytes")

        if file_size == 0:
            logger.error("Saved image file is empty")
            os.remove(image_filename)  # Clean up empty file
            return jsonify({"success": False, "error": "Saved image is empty"}), 400

        original_message = data.get('original_message', 'How are you feeling today?')

        return jsonify({
            "success": True,
            "filename": image_filename,
            "timestamp": timestamp,
            "size": file_size
        }), 200
    except Exception as e:
        logger.error(f"Error capturing photo: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": f"Error processing image: {str(e)}"
        }), 500


# Serve static files from uploads directory
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    logger.info("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0',port = 5001)  # Added host to make it accessible from other devices