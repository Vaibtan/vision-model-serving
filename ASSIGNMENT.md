# Assignment Requirements: Vision Model Serving Pipeline

## Overview & Broad Tasks

### 1. Understand the Provided Models and Inference Pipeline
Familiarize yourself with the architecture of each model, the expected input and output formats, preprocessing steps, postprocessing logic, and any dependencies required for inference. The objective is to understand the existing inference pipeline before integrating it into an API service.

### 2. Build a Django Inference Service
Develop a Django application that loads the provided models and exposes REST API endpoints for inference. The service should accept the appropriate inputs, perform all necessary preprocessing, run inference using the loaded models, and return the predictions in a well-structured JSON response.

### 3. Design the Application for Efficient Serving
Ensure that the models are loaded only once during application startup and reused for subsequent requests. Organize the project with a clean and modular structure, separating API routes, model loading, inference logic, configuration, and utility functions. The application should be robust, easy to extend, and suitable for production deployment. If you are able to convert the model into acceptable TensorRT format for deployment, it will be further appreciated.

### 4. Containerize and Validate the Deployment
Package the application using Docker so that it can be deployed with minimal setup. Provide clear instructions for reproducing the service locally, along with example API requests and responses. Verify that the endpoints function correctly and that the service can handle inference requests reliably.

---

## Purpose & Assessment Focus

Through this assessment, we aim to evaluate your ability to integrate machine learning models into a production-ready backend, design clean and maintainable software, and deploy inference services using modern engineering practices. The focus of the assignment is not on developing new machine learning models, but on building a reliable and well-engineered deployment pipeline.

In case you have any queries, feel free to reach out to us at any time.
Best of luck with the assignment!

---

## Reference Resources

- **Paper**: [MICCAI 2024 Paper (1311)](https://papers.miccai.org/miccai-2024/paper/1311_paper.pdf)
- **MMBCD Code**: [adsbansal/MMBCD GitHub Repository](https://github.com/adsbansal/MMBCD)
- **Focal-Net DINO Repo**: [FocalNet/FocalNet-DINO GitHub Repository](https://github.com/FocalNet/FocalNet-DINO)
- **Focal-Net DINO Weights and Config**: [Google Drive Folder](https://drive.google.com/drive/folders/1rYFfGH97fLRnybRlP1p11Viku2Ofti9v)

---

## Important Points to Note

1. **DICOM Testing Data**: Please download a publicly available DICOM of a Mammogram to test the pipeline.
2. **Model Loading / Unloading**: There are 2 models. Please ensure the model inferencing logic uses one model at a time, so it must load and unload the models.
