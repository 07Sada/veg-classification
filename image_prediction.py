import onnxruntime
import numpy as np
from PIL import Image
from torchvision import transforms
import glob

# Define the class labels
CLASS_LABELS = ['Cucumber', 'Capsicum', 'Papaya', 'Tomato', 'Cabbage', 'Pumpkin', 'Bitter_Gourd',
                'Radish', 'Broccoli', 'Cauliflower', 'Bean', 'Carrot', 'Bottle_Gourd', 'Potato', 'Brinjal']

# Finding the model path from the artifacts directory
model_paths = glob.glob('artifacts/*')
model_path = model_paths[0] if model_paths else None
if not model_path:
    print("No model found in the 'artifacts' directory.")
    exit()

# Initialize the ONNX runtime session
session = onnxruntime.InferenceSession(model_path)

class ImageClassification:
    def __init__(self, img_path):
        self.img_path = img_path

    def image_transformation(self):
        # Open the image
        image = Image.open(self.img_path)

        # Define the transformations
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

        # Apply the transformations to the image
        transformed_image = transform(image)

        return transformed_image

    def infer_and_display_image(self):
        # Perform inference
        input_image = self.image_transformation()
        input_tensor = np.expand_dims(input_image, axis=0)

        # Run inference
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        output = session.run([output_name], {input_name: input_tensor})

        # Postprocess the output
        output = output[0]
        predicted_class_index = np.argmax(output)
        predicted_class = CLASS_LABELS[predicted_class_index]

        return predicted_class
