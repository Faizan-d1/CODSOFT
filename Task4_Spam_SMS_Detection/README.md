# Task 3 - SMS Spam Detection (Machine Learning)

## Description
This project builds a Machine Learning model that classifies SMS messages as **Spam** or **Ham (Legitimate)** using Natural Language Processing (NLP) techniques.  
The system uses **TF-IDF vectorization** to convert text into numerical features and applies classification algorithms to detect spam messages.

## Features
- Loads and preprocesses SMS message dataset  
- Converts categorical labels (ham/spam) into numeric format  
- Cleans text data by removing stop words and converting to lowercase  
- Converts text into numerical features using **TF-IDF**  
- Splits dataset into training and testing sets using stratified sampling  
- Trains and evaluates multiple models:
  - Multinomial Naive Bayes  
  - Logistic Regression  
- Evaluates model performance using accuracy, precision, recall, F1-score, and confusion matrix  
- Visualizes:
  - Spam vs Ham distribution  
  - Confusion matrix for Naive Bayes  
  - Confusion matrix for Logistic Regression  

## How to Run
1. Open terminal or VS Code  
2. Navigate to the project folder  
3. Open the Jupyter Notebook file:  
   SMS_Spam_Detection.ipynb  
4. Run all cells step by step to:  
   - Preprocess the data  
   - Train the models  
   - Evaluate performance  
   - View visualizations  

## Author
Faizan  
CodSoft Machine Learning Internship
