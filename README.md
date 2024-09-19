# Face Emotion Recognition Project 
# This project serves as the graduation project for my internship at the National Telecommunication Institute (NTI).
# Project 1 Description
This project aims to perform real-time emotion recognition focuses on building a Convolutional Neural Network (CNN) to recognize human emotions from grayscale images. The goal is to classify images into one of seven emotion categories: angry, disgust, fear, happy, neutral, sad, and surprise. The project uses a modified dataset of images and employs various deep learning techniques to achieve accurate emotion classification.

## Dataset
The dataset consists of RGB images labeled with one of seven emotion categories: angry, disgust, fear, happy, neutral, sad, and surprise. The dataset is divided into training and validation sets, with images preprocessed to a target size of 48x48 pixels.

## Model Architecture
The CNN model, built using the Keras library, comprises four convolutional layers with filters of sizes 64, 128, and 512, each followed by batch normalization, ReLU activation, max pooling, and dropout. This is succeeded by three fully connected layers with units 256, 512, and 512, incorporating batch normalization, ReLU activation, and dropout. The output layer is a dense layer with 7 units and softmax activation to classify the emotions.

## Training
The model is compiled with the Adam optimizer, using a learning rate of 0.0001 and categorical cross-entropy loss. The model is trained for 30 epochs with early stopping and learning rate reduction on plateau callbacks to prevent overfitting.

## Evaluation
Loss and Accuracy Plots: The training and validation loss and accuracy are plotted to visualize the model's performance over epochs.
Confusion Matrix: A confusion matrix is plotted using seaborn to evaluate the model's classification performance on the validation set.
Classification Report: A classification report is generated to provide precision, recall, and F1-score for each emotion category.
Model Saving and Inference
The trained model is saved to an HDF5 file (model.h5) for later use. A custom function emotion_analysis is implemented to visualize the predicted emotion probabilities for a given input image.

## Results
The model achieves good performance on the validation set, accurately classifying emotions from grayscale images. The confusion matrix and classification report provide detailed insights into the model's strengths and weaknesses across different emotion categories.

## Conclusion
This project demonstrates the application of deep learning techniques to emotion recognition from images. The CNN model shows promising results and can be further improved with more data and advanced techniques.

Feel free to explore, modify, and extend this project to suit your needs. Happy coding!

# Project 2 Description 
This project aims to enhance emotion recognition from images using a combination of deep learning and machine learning techniques. By leveraging a pre-trained Xception model for feature extraction and a genetic algorithm for feature selection, we build a Random Forest classifier to accurately classify emotions.

## Dataset
The dataset consists of RGB images labeled with one of seven emotion categories: angry, disgust, fear, happy, neutral, sad, and surprise. The dataset is divided into training and validation sets, with images preprocessed to a target size of 48x48 pixels.

## Techniques Used
#### Feature Extraction
A pre-trained Xception model is used to extract features from the images. The Xception model is loaded with weights trained on ImageNet and modified to output feature vectors.

#### Feature Selection with Genetic Algorithm
A genetic algorithm is implemented to select the most relevant features for classification. It starts by generating a population of binary feature masks. Each mask is evaluated based on the classification accuracy of a Random Forest model using the selected features. The best individuals are selected for crossover and mutation to create offspring, with the top performers retained for the next generation. This process repeats for several generations to evolve an optimal set of features.

## Classification
A Random Forest classifier is trained on the selected features to classify the emotions. The model's performance is evaluated using accuracy, a confusion matrix, and a classification report.

## Results
The genetic algorithm successfully identifies the most relevant features, and the Random Forest classifier achieves a high accuracy on the validation set. The confusion matrix and classification report provide detailed insights into the model's performance across different emotion categories.

## Conclusion
This project demonstrates a novel approach to emotion recognition by combining deep learning feature extraction with a genetic algorithm for feature selection and a Random Forest classifier for emotion classification. The results indicate that this hybrid approach can effectively identify and classify emotions from images.

Feel free to explore, modify, and extend this project to suit your needs. Happy coding!






