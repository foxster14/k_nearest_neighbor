# Randomly split the train/test by 60/40
# then normalize the data
# for each test through, find the euclidean distance from each training rows & each data point
# Choose row by row for test data to see the distances from the test data's x and y-values
# arrange that vector in ascending order (don't reset index)
# add a column to the vector for distance 
# for each row, iterate through columns for smallest K number of values
    # K 1-10 is how many neighbors we are picking
# Make list out of the smallest K values (these are the K-nearest neighbors) 
    # Create a dictionary off the fruit names 
    # Convert the K-nearest neighbor distances to their fruit name using dictionary based off trainData index
# Find most occurences (AKA mode) of fruit_name        
# Add a column called y_predicted to new dataframe
# next find accuracy, number of correct y-value predictions in predicted data set which can be found
# by comparing predicted y-values to test-values 
# the complications happen when there are equidistant values 

import pandas as pd
from sklearn import linear_model
import sklearn.metrics as sk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import scipy.stats as stats
from statistics import multimode, mode
import random
import matplotlib.pyplot as plt

path = '/Users/sarahfox/OneDrive - Dunwoody College of Technology/Junior Year/Data Science/Lab 5/' #path where all files are stored
filename = 'fruits_classification.csv'
data_frame = pd.read_csv(path + filename)

## Using sci-kit to validate
x = data_frame[['mass','width','height','color_score']]
y = data_frame['fruit_name']

## normalize before splitting
normalized_x = x.apply(stats.zscore)

## Randomly split the train/test data 60/40
x_train, x_test, y_train, y_test = train_test_split(normalized_x, y, train_size = 0.6, random_state = 0, shuffle=True)

#### ------- Calculate the Euclidean Distance Between Two Vectors ------- #### 
def euclidean_distance(x1, x2, y1, y2, z1, z2, k1, k2):
	distance = 0.0
	distance += np.sqrt(((x1 - x2)**2) + ((y1 - y2)**2) + ((z1 - z2)**2) + ((k1 - k2)**2))
	return distance

distance_df = pd.DataFrame()

## train data is the columns (header) of the new vector
## test data is the rows (index)
for index, row in x_test.iterrows():
    for index2, row2 in x_train.iterrows():
        distance = euclidean_distance(row['mass'], row2['mass'], row['width'], row2['width'], row['height'], row2['height'], row['color_score'], row2['color_score'])
        distance_df.loc[index, index2] = distance

## Dataframe to store accuracy counts vs k-iterations
k_value_vs_accuracy = pd.DataFrame()
k_value_vs_accuracy['Accuracy'] = ''
k_value_vs_accuracy['K Value'] = 0

#### ----- Iterate through K 1-10 ----- ####
for k in range(1,11):

    #### -------- Sort Distance Values by Test & Find K-Nearest Neighbor ------ ####
    test_list = distance_df.index
    d = data_frame['fruit_name'].to_dict()
    neighbors_df = pd.DataFrame(index= distance_df.index, columns=np.arange(k))

    for i in range(len(distance_df)):
        test = distance_df.iloc[:, np.argsort(distance_df.loc[test_list[i]])]
        test = test.iloc[:,:k]
        neighbors_df.loc[test_list[i]] = test.loc[test_list[i]].index

    #### ------- Find the Predicted Value -------- ####
    fruit_locations = neighbors_df.replace(d)
    fruit_locations['predicted_y'] = ''

    #### ------- Locate occurences of each fruit ---------- ####
    lemonLoc = fruit_locations.isin(['lemon']).sum(axis=1)
    orangeLoc = fruit_locations.isin(['orange']).sum(axis=1)
    appleLoc = fruit_locations.isin(['apple']).sum(axis=1)
    mandarinLoc = fruit_locations.isin(['mandarin']).sum(axis=1)
    fruitCount_df = pd.DataFrame(index=distance_df.index, columns=['lemon','orange','apple','mandarin'])
    fruitCount_df['lemon'] = lemonLoc
    fruitCount_df['orange'] = orangeLoc
    fruitCount_df['apple'] = appleLoc
    fruitCount_df['mandarin'] = mandarinLoc

    ## Need list of the index values to locate rows in the dataframes inside of for loop
    testDataIndexValue = fruitCount_df.index
    ## Declare Empty Dataframe to Store Predicted-Y, Actual-Y and K-Value 
    comparison_df = pd.DataFrame(index=distance_df.index, columns=['Predicted Y','Actual Y'])

    #### -------- Find Predicted Y-Values & Actual Y-Values From Test Data -------- ####
    def accuracy_metric(actual, predicted):
        correct = 0
        for i in actual.index:
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

    fruitCount_df['modes'] = ''

    for i in range(len(fruit_locations)):
        ## Locate the mode(s) of each fruitname in each row of testData
        modes = multimode(fruit_locations.iloc[i,:])
        ## Store the list of modes in case they need to be accessed
        fruitCount_df.at[testDataIndexValue[i],'modes'] = modes
        ## Bring over the actual y-values from the testData so I can compare accuracy
        comparison_df.at[testDataIndexValue[i],'Actual Y'] = y_test[testDataIndexValue[i]]
        ## Locate the ties and randomly choose which fruit will be used as prediction
        if len(modes) == 1:
            comparison_df.at[testDataIndexValue[i],'Predicted Y'] = modes[0]
        else:
            randomlySelected = random.choice(modes)
            comparison_df.at[testDataIndexValue[i],'Predicted Y'] = randomlySelected
        #comparison_df.at[testDataIndexValue[i],'Accuracy'] = accuracy_metric(fruit_locations, comparison_df.at[testDataIndexValue[i],'Actual Y'], comparison_df.at[testDataIndexValue[i],'Predicted Y'])

    actual = comparison_df['Actual Y']
    predicted = comparison_df['Predicted Y']
    accuracy = accuracy_metric(actual, predicted)

    k_value_vs_accuracy.at[k, 'Accuracy'] = accuracy
    k_value_vs_accuracy.at[k,'K Value'] = k


#### ------ Plot Accurracy & K-Value on Scatter Plot Graph ------ ####
x = k_value_vs_accuracy.iloc[:,0]
y = k_value_vs_accuracy.iloc[:,1]
plt.scatter(x,y)
## Add an x-axis label
plt.ylabel("K-Value")
## Add a y-axis label
plt.xlabel("Accuracy (%)")
## Add a title
plt.title("Accuracy vs K-Value")
plt.show()

        

