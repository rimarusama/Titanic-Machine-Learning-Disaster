# Titanic Machine Learning Disaster

This is a Python script that predicts the survival of passengers on the Titanic using a Random Forest Classifier. The script reads training and test data from CSV files, performs data preprocessing, trains the model, and generates a submission file with predictions.

## Getting Started
To get started with this project, follow these steps:

### Prerequisites
You need to have Python 3 installed, along with the following libraries:

numpy
pandas
scikit-learn (for RandomForestClassifier)

You can install these libraries using pip:
pip install numpy pandas scikit-learn

### Installation
1. Clone this GitHub repository:
   git clone https://github.com/your-username/Titanic-Machine-Learning-Disaster.git

   cd Titanic-Machine-Learning-Disaster

3. Download the Titanic dataset files train.csv and test.csv and place them in the root directory of the project.

### Usage
To run the script and generate predictions, execute the following command:

python titanicPrediction.py

The script will read the training and test data, train the Random Forest Classifier, make predictions, and save the results in a file called submission.csv in the same directory.

### Model Configuration
You can adjust the model's hyperparameters by modifying the following lines in the script:

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

Feel free to experiment with different values to improve prediction accuracy.

### Acknowledgments
The Titanic dataset is available from Kaggle.

This project is by the Kaggle competition Titanic: Machine Learning from Disaster.
