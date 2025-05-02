import cv2
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
from models.cnn_model import EmotionCNN 

IMAGE_PATH = 'webcam_image.png'
MODEL_PATH = 'best_model.pth'
INPUT_SIZE = (48, 48)
CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def crop_center_square(img):
    h, w = img.shape[:2]
    min_dim = min(h, w)
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2
    return img[top:top + min_dim, left:left + min_dim]

def capture_and_process_image(output_path=IMAGE_PATH, size=INPUT_SIZE):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Taking picture... Press 's' to snap.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Webcam - Press s to Snap', frame)
        key = cv2.waitKey(1)
        if key == ord('s'):
            break

    cap.release()
    cv2.destroyAllWindows()

    #downsizing here
    cropped = crop_center_square(frame)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    pil_img = Image.fromarray(sharpened)
    resized = pil_img.resize(size, resample=Image.LANCZOS)
    resized.save(output_path)
    print(f"Saved processed image to {output_path}")

def load_image_as_tensor(image_path=IMAGE_PATH):
    transform = transforms.Compose([
        transforms.ToTensor(),  # (1, 48, 48)
        transforms.Normalize((0.5,), (0.5,))
    ])
    img = Image.open(image_path).convert('L')
    return transform(img).unsqueeze(0)  # (1, 1, 48, 48)

#actually using the model
def predict_emotion(model, image_tensor, class_names=None):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1).squeeze()  # shape: (7,)
        prob_list = probs.cpu().numpy()


        class_probs = list(zip(class_names, prob_list))
        class_probs.sort(key=lambda x: x[1], reverse=True)

        # print("\n Emotion Probabilities:")
        # print("{:<12} | {:<10}".format("Emotion", "Probability"))
        # print("-" * 26)
        # for cls, prob in class_probs:
        #     print("{:<12} | {:.4f}".format(cls, prob))
        predicted_class = class_probs[0][0]
        class_probs = dict(class_probs)
        return predicted_class,class_probs  # Return top prediction if needed


def main():

    capture_and_process_image()

    img_tensor = load_image_as_tensor()

    model = EmotionCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))

    prediction = predict_emotion(model, img_tensor, CLASS_NAMES)
    print("\n Predicted emotion:", prediction)

## lod func
def predict_emotion_from_image_path(image_path, model_path='epoch_884.pth', input_size=(48,48), class_names= ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']):
    # Applying same process as Jeremy
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    cropped = crop_center_square(img)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)

    pil_img = Image.fromarray(sharpened)
    resized = pil_img.resize(input_size, resample=Image.LANCZOS)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img_tensor = transform(resized).unsqueeze(0)  # shape: (1, 1, 48, 48)

    # Load model and predict
    model = EmotionCNN()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return predict_emotion(model, img_tensor, class_names)


if __name__ == '__main__':
    print(predict_emotion_from_image_path(IMAGE_PATH))