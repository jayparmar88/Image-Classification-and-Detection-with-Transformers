# Image-Classification-and-Detection-with-Transformers :

# Visual Transformers (ViT) Project

## Overview
Welcome to the Visual Transformers (ViT) project! This project explores the capabilities of ViT models for various image understanding tasks. It showcases the versatility of ViT models through examples of simple image classification, zero-shot image classification, and zero-shot object detection.

## Project Structure
The project is organized into tasks, each demonstrating a specific aspect of using ViT models. Below is an overview of the project structure:

1. **Task 1: Image Classification - Loading Vision Transformer**
   - Initializes the feature extractor and a pre-trained ViT model for image classification.

2. **Task 2: Image Classification - Generate Features from an Image**
   - Extracts features from an image using the pre-trained ViT model.

3. **Task 3: Image Classification - Make Predictions**
   - Makes predictions for an image using the pre-trained ViT model.

4. **Task 4: Zero-shot Image Classification - Loading Models**
   - Loads pre-trained models for zero-shot image classification.

5. **Task 5: Zero-shot Image Classification - Prepare the Inputs**
   - Prepares inputs for zero-shot image classification using a processor.

6. **Task 6: Zero-shot Image Classification - Generate Predictions**
   - Generates predictions for zero-shot image classification.

7. **Task 7: Zero-shot Object Detection - Loading Model**
   - Loads pre-trained models for zero-shot object detection.

8. **Task 8: Zero-shot Object Detection - Prepare the Inputs for the Model**
   - Prepares inputs for zero-shot object detection using an image and textual queries.

9. **Task 9: Zero-shot Object Detection - Visualize the Results**
   - Performs zero-shot object detection and visualizes the results by adding bounding boxes and labels.

## Usage
Each task is documented in its respective section within the project. To explore a specific task, navigate to the corresponding section and follow the instructions provided.

## Requirements
Ensure you have the necessary dependencies installed before running the code. The required packages include transformers, PIL, requests, torch, and matplotlib. You can install them using the following command:

```bash
pip install transformers pillow requests torch matplotlib
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [Hugging Face Transformers](https://github.com/huggingface/transformers): Used for accessing pre-trained ViT models.
