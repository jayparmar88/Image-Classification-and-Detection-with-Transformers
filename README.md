# Visual Transformers (ViT) for Image Understanding :

## Project Overview :
Welcome to the Visual Transformers for Image Understanding project! This project explores the capabilities of `Visual Transformers (ViT)` in tackling various image understanding tasks. By leveraging ViT models, we aim to showcase their adaptability and effectiveness in tasks such as simple image classification, zero-shot image classification, and zero-shot object detection.

## Purpose of the Project :
The primary purpose of this project is to demonstrate how Visual Transformers, a type of transformer architecture originally designed for natural language processing, can be applied to image-related tasks. Traditional convolutional neural networks (CNNs) have been the go-to choice for image-related tasks, but ViTs offer a new perspective by treating images as sequences of patches. This approach allows ViTs to excel in tasks beyond classification, including zero-shot learning and object detection.

## Usefulness and Applications :
### 1. Simple Image Classification
- **Purpose:** Train a ViT model to classify images into predefined categories.
- **Usefulness:** This task serves as a foundational example of using ViTs for image understanding. It is applicable in scenarios where traditional image classification is required.

### 2. Zero-Shot Image Classification
- **Purpose:** Explore the ViT model's ability to classify images into categories it has never seen during training.
- **Usefulness:** Demonstrates the model's capability to generalize to new classes, making it valuable in real-world situations where new categories may emerge.

### 3. Zero-Shot Object Detection
- **Purpose:** Localize and identify objects within images, even if the model has never encountered those objects during training.
- **Usefulness:** Showcases the ViT model's potential for object detection in novel contexts, especially useful when dealing with rare or infrequent objects.

## Project Structure and Tasks :
1. **Image Classification - Loading Vision Transformer :**
   - Initialize the feature extractor and pre-trained ViT model for image classification.
   - Example: "vit-base-patch16-224"

2. **Image Classification - Generate Features from an Image :**
   - Extract features from an image using the pre-trained ViT model.

3. **Image Classification - Make Predictions :**
   - Utilize the ViT model to make predictions for the given image.

4. **Zero-shot Image Classification - Loading Models :**
   - Load pre-trained models designed for zero-shot image classification.
   - Example: "openai/clip-vit-large-patch14"

5. **Zero-shot Image Classification - Prepare the Inputs :**
   - Prepare inputs for zero-shot image classification using textual descriptions and an image.

6. **Zero-shot Image Classification - Generate Predictions :**
   - Generate predictions for zero-shot image classification based on the pre-trained model and inputs.

7. **Zero-shot Object Detection - Loading Model :**
   - Load pre-trained models for zero-shot object detection.
   - Example: "google/owlvit-base-patch32"

8. **Zero-shot Object Detection - Prepare the Inputs for the Model :**
   - Prepare inputs for zero-shot object detection using textual queries and an image.

9. **Zero-shot Object Detection - Visualize the Results :**
   - Perform zero-shot object detection on the image and visualize the results with bounding boxes and labels.

## Requirements :
- Python 3.7+
- [Transformers](https://github.com/huggingface/transformers)
- [PIL](https://pillow.readthedocs.io/en/stable/)
- [Requests](https://docs.python-requests.org/en/latest/)
- [torch](https://pytorch.org/getting-started/locally/)

## License :
This project is licensed under the [MIT License](LICENSE).

## Credit and Acknowledgment :
- The project uses models and components from the [Hugging Face Transformers](https://github.com/huggingface/transformers) library.
- DataCamp - https://www.datacamp.com/code-along/image-classification-hugging-face
