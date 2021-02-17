#IMPORTING LIBRARIES
import pandas as pd
from pandas import DataFrame

#READING DATASET IN TO DATAFRAME
dFrame1 = pd.read_csv("train.csv")
print(dFrame1) 

#TRIMMING DATASET OF RELEVANT COLUMNS FOR ANALYSIS
X = dFrame1.iloc[:, [2,4]].values
Y = dFrame1.iloc[:, 1].values
print(X)
print(Y)


#CONVERTING CATEGORICAL DATA IN TO NUMERIC COLUMNS
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],
                                     remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

#SPLITTING DATASET IN TO TEST AND TRAIN  SET RESPECTIVELY
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.469, random_state=0)

#IMPORTING CLASSIFIER
from sklearn.ensemble import RandomForestClassifier
classifier  = RandomForestClassifier(n_estimators=20, criterion='entropy', random_state=0)
classifier.fit(X_train, Y_train)

#PREDICTING OUTCOME
Y_pred = classifier.predict(X_test)

#MEASURING MODEL PERFORMANCE
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

Accuracy = 219 + 107
print (Accuracy)

Accuracy_Rate = (219 + 107) / 418
print (Accuracy_Rate)

Error = 45 + 47
print (Error)

Error_Rate = (45 + 47) / 418
print (Error_Rate)








