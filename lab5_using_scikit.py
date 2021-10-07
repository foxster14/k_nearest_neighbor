import pandas as pd
from scipy.sparse import data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

path = '/Users/sarahfox/OneDrive - Dunwoody College of Technology/Junior Year/Data Science/Lab 5/' #path where all files are stored
filename = 'fruits_classification_edited.csv'
data_frame = pd.read_csv(path + filename)

## Assigning attributes and class variables
x = data_frame.iloc[:, :-1].values ## this means all rows & all the columns except the last column
y = data_frame.iloc[:,4].values

## Performing train/test split (randomly)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.40)

## Normalizing the dataset
scalar = StandardScaler()
scalar.fit(x) # gives us the average & standard deviation of each x-variable
# The y-column is a class with categorical data so we don't need to normalize it

x_train = scalar.transform(x_train) ## This applies the Z-score
x_test = scalar.transform(x_test)

## Fitting the model using KNN
classifier = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
classifier.fit(x_train, y_train) # using training vectors

y_pred = classifier.predict(x_test) # for each row of the test data set, i want to predict the Y
# this will be a column vector

print(y_test, y_pred)
print(accuracy_score(y_test, y_pred))