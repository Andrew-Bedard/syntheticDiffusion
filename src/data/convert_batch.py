"""
Convert a batch of images to cifar-10 format
"""
import os
from PIL import Image
from torchvision import transforms

def process_images(input_directory, output_directory):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to 32x32
        transforms.ToTensor()  # Convert to tensor
    ])

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        img_path = os.path.join(input_directory, filename)
        image = Image.open(img_path)  # Load your image
        transformed_image = transform(image)  # Apply the transform without normalization
        transformed_image_pil = transforms.ToPILImage()(transformed_image)  # Convert the tensor back to a PIL image
        transformed_image_pil.save(os.path.join(output_directory, filename), format="PNG")


input_directory = 'C:\\Users\\mongo\\Desktop\\cat1'
output_directory = 'D:\\Projects\\syntheticDiffusion\\data\\synthetic_cats\\'
process_images(input_directory, output_directory)
