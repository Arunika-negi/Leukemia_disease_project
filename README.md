# Leukemia_disease_project
This project focuses on building a deep learning–based system for automatic detection of Leukemia from microscopic blood smear images. Early and accurate diagnosis of leukemia is critical for effective treatment, and this project demonstrates how modern AI techniques can support medical analysis with high reliability.

# Dataset
The dataset contains 8009 microscopic blood smear images categorized into:
*Leukemia (infected)*
*Normal (healthy)*

🔗 **Download Full Dataset (Google Drive):** 
https://drive.google.com/drive/folders/1_Aywcveu4rJgT2zeLpYb_LwtYG1ltdgJ?usp=sharing

# Models
keras folder : https://drive.google.com/drive/folders/1Uc-xN3HXJol_jOHTXMKV2tzw2XyS5a4C?usp=sharing
h5 file : https://drive.google.com/file/d/1yQTj_Rxppv6F5QCoQxxC5Zd37oblvjuU/view?usp=sharing

# Project Structure
📁 leukemia-detection/
│
├── dataset/                 # Images (train/test)
├── model.keras              # Saved trained model
├── leukemia_detection.ipynb # Training + evaluation notebook
├── README.md                # Project documentation
└── requirements.txt         # Dependencies

# Technologies Used
Python
TensorFlow / Keras
NumPy
Matplotlib
OpenCV
Scikit-learn

# Model Architecture (Summary)
*This notebook builds a Convolutional Neural Network (CNN) consisting of:*
Convolution layers
MaxPooling layers
Flatten layer
Dense fully connected layers
Softmax/Sigmoid output layer (depending on class count)

*The network is trained using:*
Loss: categorical_crossentropy / binary_crossentropy
Optimizer: Adam
Metrics: Accuracy
