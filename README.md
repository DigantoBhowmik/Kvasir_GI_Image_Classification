# Kvasir GI Image Classification with EfficientViT and PyTorch Lightning

This project focuses on building an efficient deep learning model for the classification of GI images into different categories, enabling automated and accurate disease identification.

## Dataset
The Kvasir dataset comprises annotated medical images of the gastrointestinal (GI) tract, with each class showing various Glandmarks, pathological findings, and endoscopic procedures. The dataset includes hundreds of images for each class, making it suitable for tasks such as machine learning, deep learning, and transfer learning. It covers anatomical landmarks like the Z-line, pylorus, and cecum, as well as pathological findings such as esophagitis, polyps, and ulcerative colitis. Additionally, there are images related to lesion removal procedures, like "dyed and lifted polyp" and "dyed resection margins." The images vary in resolution, ranging from 720x576 to 1920x1072 pixels, and are organized into folders corresponding to their content. Some images contain a green picture-in-picture overlay showing the endoscope's position and configuration, which can aid in interpretation but should be handled carefully when detecting endoscopic findings.

## Data Transformation and Augmentation
### Data Augmentation for Training
- To enhance model robustness, a series of data augmentations are applied during training:
  - Random Horizontal Flip: Images are randomly flipped horizontally.
  - Random Vertical Flip: Images are randomly flipped vertically.
  - Random Rotation (up to 15 degrees): Images undergo random rotation.
  - Resize to a specified image size (e.g., 224x224 pixels).
  - Conversion to PyTorch tensor.
  - Normalization using ImageNet statistics.

### Data Transformation for Validation and Testing
- For validation and testing, a simpler transformation pipeline is used:
  - Resize to a specified image size.
  - Conversion to PyTorch tensor.
  - Normalization using ImageNet statistics.

### Dataset Preparation
The training dataset includes a split of 90% training and 10% validation data. The dataset is organized into training, validation, and testing sets. Test transform use on test and validation dataset.

## Model Training using PyTorch Lightning
### Model Initialization
- The selected model architecture is 'efficientvit_m3,' loaded with pre-trained weights.
- A custom PyTorch Lightning module is defined for training and evaluation.

### Training Configuration
- Training is conducted for a specified number of epochs (e.g., 20) using a 16-bit precision.
- Model checkpoints are saved during training.

### Training and Validation Metrics
- During training, various metrics are logged:
  - Accuracy
  - Top-3 Accuracy
  - Top-5 Accuracy
  - F1 Score
  - Loss
- Metrics are tracked for both the training and validation datasets.

## Result:
- Train Accuracy: 0.9688
- Train Top-3 Accuracy: 1.0000
- Train Top-5 Accuracy: 1.0000
- Train Loss: 0.1443
- Train F1 Score: 0.9773
- Validation Accuracy: 0.9016
- Validation Top-3 Accuracy: 1.0000
- Validation Top-5 Accuracy: 1.0000
- Validation Loss: 0.2466
- Validation F1 Score: 0.8930
- Test Accuracy: 0.9049
- Test Top-3 Accuracy: 0.9994
- Test Top-5 Accuracy: 1.0000
- Test Loss: 0.2621
- Test F1 Score: 0.5420


## Image Feature Extraction

The code then loops through a directory containing images, loading each image one by one. For each image:

1. The image is preprocessed using the same transformation used during training. This typically includes resizing and normalization.

2. A forward pass is made through the EfficientViT model. This results in a set of features for the image.

3. The features are flattened into a one-dimensional tensor.

4. A linear layer is added to further reduce the dimensionality of the features to 2600 dimensions.

5. The features are converted to a NumPy array.

6. The class label (inferred from the subdirectory of the image) is appended to the end of the feature array.

7. The image filename is used as a unique identifier, and the feature array is stored in a dictionary with the filename as the key.

## Feature Transformation with LDA

1. Feature Extraction: Features are extracted from images using a pre-trained model and flattened.

2. Data Preparation: Extracted features are converted into a DataFrame for further processing.

3. Label Encoding: Class labels are encoded for classification.

4. Standardization: Feature values are standardized to have zero mean and unit variance.

5. Linear Discriminant Analysis (LDA): LDA is applied to reduce feature dimensionality to 7 components.

## Classification
For image classification, we employ several machine learning classifiers on two types of features: EfficientViTM3 features and concatenated model features. Here are the results for each classifier:

### Classification for EfficientViTM3 features

| Model                       | Accuracy   |
|-----------------------------|------------|
| Random Forest               | 0.88       |
| LightGBM                    | 0.88       |
| Support Vector Machine (SVM)| 0.875      |
| k-Nearest Neighbors (KNN)   | 0.86       |
| Adaboost                    | 0.745      |

### Classification for Concatenated model features

| Model                       | Accuracy   |
|-----------------------------|------------|
| Random Forest               | 0.99625    |
| Support Vector Machine (SVM)| 0.99625    |
| k-Nearest Neighbors (KNN)   | 0.99625    |
| LightGBM                    | 0.99625    |
| Adaboost                    | 0.9925     |

## Conclusion
In this task, I demonstrated the process of utilizing a pre-trained EfficientViT model for image feature extraction, dimensionality reduction with LDA, and image classification using various machine learning classifiers. The results showed that the concatenated model features significantly outperformed the EfficientViTM3 features across all classifiers, indicating the effectiveness of the feature transformation process.

The use of deep learning models for feature extraction followed by machine learning classifiers allows for robust image classification in various applications, including object recognition, content-based image retrieval, and more. The choice of classifiers should be based on the specific requirements and characteristics of the dataset, and hyperparameter tuning can further improve classification performance. Overall, this approach provides a valuable framework for leveraging pre-trained models and machine learning techniques for image analysis tasks.
