import os
import cv2
import numpy as np

def add_salt_and_pepper_noise(image, salt_vs_pepper=0.5, amount=0.004):
    noisy = np.copy(image)
    num_salt = np.ceil(amount * image.size * salt_vs_pepper)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[coords[0], coords[1]] = 1

    num_pepper = np.ceil(amount * image.size * (1.0 - salt_vs_pepper))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[coords[0], coords[1]] = 0
    return noisy

def apply_blur(image, blur_amount=3):
    return cv2.GaussianBlur(image, (blur_amount, blur_amount), 0)

def enhance_images(dataset_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i in range(1, 176):  # 175个人
        for j in range(1, 8):  # 每个人7张照片
            img_path = os.path.join(dataset_path, f'{i:03d}_{j:02d}.bmp')
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not load image {img_path}")
                continue

            # Save the original image
            original_img_path = os.path.join(output_path, f'{i:03d}_{j:02d}_original.bmp')
            cv2.imwrite(original_img_path, img)

            # Apply salt and pepper noise
            img_noisy = add_salt_and_pepper_noise(img)

            # Apply blur
            img_blurred = apply_blur(img)

            # Save the enhanced images
            noisy_img_path = os.path.join(output_path, f'{i:03d}_{j:02d}_noisy.bmp')
            blurred_img_path = os.path.join(output_path, f'{i:03d}_{j:02d}_blurred.bmp')
            cv2.imwrite(noisy_img_path, img_noisy)
            cv2.imwrite(blurred_img_path, img_blurred)

# Set the paths
dataset_path = 'feret_k175_s7_w80_h80'  # Update this to your dataset path
output_path = 'feret_improved'  # Update this to your desired output path

# Enhance images
enhance_images(dataset_path, output_path)