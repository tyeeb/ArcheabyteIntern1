import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        try:
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                raise IOError(f"cv2.imread failed to load image at {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
            images.append(img)
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
    return images

def preprocess_images(images, size=(128, 128)):
    processed_images = []
    for img in images:
        try:
            # Resize image
            img = cv2.resize(img, size)
            
            # Normalize pixel values to be between 0 and 1
            img = img / 255.0
            
            processed_images.append(img)
        except Exception as e:
            print(f"Error processing image: {e}")
    
    return np.array(processed_images)

def detect_and_highlight_scratches(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian Blur to smooth the image and reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Use Canny edge detector to find edges in the image
    edges = cv2.Canny(blurred_image, 100, 200)
    
    # Convert edges to RGB to overlay with the original image
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    # Highlight edges (potential scratches) in red
    highlighted_image = np.where(edges_colored == [255, 255, 255], [255, 0, 0], image)
    
    return highlighted_image

# Load and preprocess images from specific folders
with_scratches = load_images_from_folder('with_scratches')
without_scratches = load_images_from_folder('without_scratches')

# Optional: Display a few images to verify loading
# Display the first image with scratches
if with_scratches:
    plt.imshow(with_scratches[0])
    plt.title('First Loaded Image With Scratches')
    plt.show()

# Highlight scratches on the first image with scratches (as an example)
if with_scratches:
    highlighted_image = detect_and_highlight_scratches(with_scratches[0])
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(without_scratches[0])
    axes[0].set_title('Original Image')
    axes[1].imshow(highlighted_image)
    axes[1].set_title('Scratches Highlighted')
    plt.show()
else:
    print("No images with scratches were loaded.")
