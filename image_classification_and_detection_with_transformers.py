# -*- coding: utf-8 -*-

# Setup :

# From the transformers package, import ViTFeatureExtractor and ViTForImageClassification
from transformers import ViTFeatureExtractor, ViTForImageClassification

# From the PIL package, import Image and Markdown
from PIL import Image

# import requests
import requests

# import torch
import torch

# import matplotlib
import matplotlib.pyplot as plt

# Task 1: Image Classification - Loading Vision Transformer :

# Load the feature extractor for the vision transformer
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# Load the pre-trained weights from vision transformer
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Task 2: Image Classification - Generate features from an Image :

image = plt.imread('laptop.jpeg')

# Display the image
plt.imshow(image)

# Extract features from the image using the feature extractor
inputs = feature_extractor(images=image, return_tensors="pt")

# Task 3: Image Classification - Make Predictions :

# Extracted pixel values from the image
pixel_values = inputs["pixel_values"]

# Make predictions using the ViT model
outputs = model(pixel_values)

# Get the logits (raw scores) for different classes
logits = outputs.logits

# Determine the number of classes
logits.shape

# Find the index of the predicted class with the highest probability
pretrained_class_idx = logits.argmax(-1).item()

# Display the index of the class
pretrained_class_idx

# Extract the class name using the model's configuration
predicted_class = model.config.id2label[pretrained_class_idx]

# Display the predicted class name
predicted_class

# Task 4: Zero-shot Image Classification - Loading Models :

# Load a pre-trained model
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

# Specify the checkpoint name or identifier for the pre-trained model you want to use
checkpoint = "openai/clip-vit-large-patch14"

# Initialize the pre-trained model for zero-shot image classification
model = AutoModelForZeroShotImageClassification.from_pretrained(checkpoint)

# Initializes the processor associated with the same pre-trained model
processor = AutoProcessor.from_pretrained(checkpoint)

# Task 5: Zero-shot Image Classification - Prepare the Inputs :

# URL of the image you want to classify
url = "https://unsplash.com/photos/xBRQfR2bqNI/download?ixid=MnwxMjA3fDB8MXxhbGx8fHx8fHx8fHwxNjc4Mzg4ODEx&force=true&w=640"

# Open the image from the URL using the requests library and PIL
image = Image.open(requests.get(url, stream=True).raw)

# Display Image
image

# List of candidate labels for classification
candidate_labels = ["tree", "car", "bike", "cat"]

# Prepare inputs for the zero-shot image classification model
inputs = processor(images=image, text=candidate_labels, return_tensors="pt", padding=True)

# Task 6: Zero-shot Image Classification - Generate Predictions :

# Import Pytorch
import torch

# Perform inference with the model
with torch.no_grad():
    outputs = model(**inputs)

# Extract logits and calculate probabilities
logits = outputs.logits_per_image[0]
probs = logits.softmax(dim=-1).numpy()

# Convert probabilities to scores as a list
scores = probs.tolist()

# Create a list of results, sorted by scores in descending order
result = [
    {"score": score, "lable": candidate_label}
    for score, candidate_label in sorted(zip(probs, candidate_labels), key = lambda x: -x[0])
]

# Display result
result

# Task 7: Zero-shot Object Detection - Loading Model :

# Import classes from the transformers library to work with a pre-trained model
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Specify the checkpoint name or identifier for the pre-trained model you want to use
checkpoint = "google/owlvit-base-patch32"

# Initialize the pre-trained model for zero-shot object detection
model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint)

# Initializes the processor associated with the same pre-trained model
processor = AutoProcessor.from_pretrained(checkpoint)

# Task 8: Zero-shot Object Detection - prepare the inputs for the model :

# URL of the image you want to analyze
url = "https://unsplash.com/photos/oj0zeY2Ltk4/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8MTR8fHBpY25pY3xlbnwwfHx8fDE2Nzc0OTE1NDk&force=true&w=640"

# Open the image from the URL using the requests library and PIL
im = Image.open(requests.get(url, stream=True).raw)

# Display Image
im

# List of textual queries describing objects
text_queries = ["hat", "book", "sunglasses", "camera"]

# Prepare inputs for zero-shot object detection
inputs = processor(text=text_queries, images=im, return_tensors="pt")

# Task 9: Zero-shot Object Detection - Visualize the Results :

# From PIL import the ImageDraw function
from PIL import ImageDraw

# Perform inference with the model
with torch.no_grad():
    outputs = model(**inputs)
    target_sizes = torch.tensor([im.size[::-1]])
    results = processor.post_process_object_detection(outputs, threshold=0.1, target_sizes=target_sizes)[0]

# Create a drawing object for the image
draw = ImageDraw.Draw(im)

# Extract detection results (scores, labels, and bounding boxes)
scores = results["scores"].tolist()
labels = results["labels"].tolist()
boxes = results["boxes"].tolist()

# Iterate over detected objects and draw bounding boxes and labels
for score, label, box in zip(scores, labels, boxes):
    xmin, ymin, xmax, ymax = box
    draw.rectangle((xmin, ymin, xmax, ymax), outline="green", width=1)
    draw.text((xmin, ymin), f"{text_queries[label]}: {round(score, 2)}", fill="purple")

# Display the image with bounding boxes and labels
im