import cv2
from PIL import Image
import numpy as np

def crop_center_square(img):
    h, w = img.shape[:2]
    min_dim = min(h, w)
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2
    return img[top:top + min_dim, left:left + min_dim]

def capture_and_process_image(output_path='webcam_image.png', size=(48, 48)):
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

    # Crop to square
    cropped = crop_center_square(frame)

    # Convert to grayscale
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # Sharpen the image
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)

    # Use PIL for high-quality downsampling
    pil_img = Image.fromarray(sharpened)
    resized = pil_img.resize(size, resample=Image.LANCZOS)

    # Save the result
    resized.save(output_path)
    print(f"Saved processed image to {output_path}")

    return resized

# Run it
if __name__ == '__main__':
    capture_and_process_image()
