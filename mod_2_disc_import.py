import cv2
import numpy as np
import matplotlib.pyplot as plt

# Path to the image
image_path = "/Volumes/Seagate/CSC515/Mod2/shutterstock227361781--125.jpg"  

# Load the image using OpenCV
img = cv2.imread(image_path)

# Check if the image was loaded successfully
if img is None:
    print("Error: Could not load image. Check the file path and try again.")
else:
    # Convert the image to grayscale for processing
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Dimensions of the image
    rows, cols = gray_img.shape

    # 1. Translation Matrix (Shift by 50 pixels in both x and y directions)
    M_translation = np.float32([[1, 0, 50], [0, 1, 50]])
    translated_img = cv2.warpAffine(gray_img, M_translation, (cols, rows))

    # 2. Rotation Matrix (Rotate by 30 degrees around the center)
    M_rotation = cv2.getRotationMatrix2D((cols / 2, rows / 2), 30, 1)
    rotated_img = cv2.warpAffine(gray_img, M_rotation, (cols, rows))

    # 3. Scaling (Scale down to 80% of the original size)
    scaled_img = cv2.resize(gray_img, None, fx=0.8, fy=0.8)

    # 4. Shearing Matrix (Shear the image)
    shear_factor = 0.5
    M_shearing = np.float32([[1, shear_factor, 0], [shear_factor, 1, 0]])
    sheared_img = cv2.warpAffine(gray_img, M_shearing, (int(cols * 1.5), int(rows * 1.5)))

    # Display the original and transformed images
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))

    axs[0].imshow(gray_img, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(translated_img, cmap='gray')
    axs[1].set_title('Translated Image')
    axs[1].axis('off')

    axs[2].imshow(rotated_img, cmap='gray')
    axs[2].set_title('Rotated Image')
    axs[2].axis('off')

    axs[3].imshow(scaled_img, cmap='gray')
    axs[3].set_title('Scaled Image')
    axs[3].axis('off')

    axs[4].imshow(sheared_img, cmap='gray')
    axs[4].set_title('Sheared Image')
    axs[4].axis('off')

    plt.show()
