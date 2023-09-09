import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

#Reading the files
train_df = pd.read_csv('train.csv')# Train file
test_df = pd.read_csv('test.csv')# Test file

#Percentage of women survived
women = train_df.loc[train_df.Sex == 'female']['Survived']
rate_women = sum(women)/len(women)

#Percentage of men survived
men = train_df.loc[train_df.Sex == 'male']['Survived']
rate_men = sum(men)/len(men)

#Training
y = train_df['Survived']

#Features
features = ['Pclass', 'Sex', 'SibSp', 'Parch']
X = pd.get_dummies(train_df[features])
X_test = pd.get_dummies(test_df[features])

#Trained Model
model = RandomForestClassifier(n_estimators = 100, max_depth=5, random_state=1)
model.fit(X,y)
prediction = model.predict(X_test)

#Getting the Dataframe
output = pd.DataFrame({'PassengerID': test_df.PassengerId, 'Survived':prediction})

#generating the Submission file
output.to_csv('submission.csv', index=False)
print('done')
