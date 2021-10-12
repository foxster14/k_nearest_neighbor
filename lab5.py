# Randomly split the train/test by 60/40
# then normalize the data
# for each test through, find the euclidean distance from each training rows & each data point
# arrange that vector in ascending order (don't reset index)
# if k=1 then classify it as apple
# add a column to the vector for distance 
# for each row, iterate through columns for smallest x-values ####
# out of the smallest values (make list), find most occurences of fruit_label        
# Add a column called y_predicted to the test dataframe
# find the most occured value 
# find the frequency of each unique value, and choose the top frequency value

import pandas as pd
from sklearn import linear_model
import sklearn.metrics as sk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import scipy.stats as stats
from statistics import multimode, mode

path = '/Users/sarahfox/OneDrive - Dunwoody College of Technology/Junior Year/Data Science/Lab 5/' #path where all files are stored
filename = 'fruits_classification.csv'
data_frame = pd.read_csv(path + filename)

## Using sci-kit to validate
x = data_frame[['mass','width','height','color_score']]
y = data_frame['fruit_label']

## normalize before splitting
normalized_x = x.apply(stats.zscore)

## Split the data 60/40
## randomly select the training & test data, choose row by row for test data to see the distances from the test data
## x-value, and y-value 
x_train, x_test, y_test, y_train = train_test_split(normalized_x, y, train_size = 0.6, random_state = 0, shuffle=True)

# calculate the Euclidean distance between two vectors
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
    
k = 4

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

testDataIndexValue = fruitCount_df.index

for i in range(len(fruitCount_df)):
    temp = fruitCount_df.iloc[:, np.argsort(fruitCount_df.loc[testDataIndexValue[i]])]
    temp = temp.iloc[:,:k]
    sortedFruitCount = fruitCount_df.sort_values(by=testDataIndexValue[i], axis=1, ascending=False)
    #fruitCount_df[testDataIndexValue[i]] = sortedFruitCount
    #neighbors_df.loc[test_list[i]] = temp.loc[test_list[i]].index


for i in range(len(fruitCount_df)):
    #print(fruitCount_df.iloc[i,:])
    temp = fruitCount_df.iloc[:, np.argsort(fruitCount_df.loc[testDataIndexValue[i]])]
    temp = temp.iloc[:,:k]
    sortedFruitCount = fruitCount_df.sort_values(by=testDataIndexValue[i], axis=1, ascending=False)
    #print(sortedFruitCount.iloc[i,:])
    maxValueIndex = sortedFruitCount.max(axis=1)
    #if maxValueIndex[testDataIndexValue[i]] > (len(fruit_locations.columns)/2):
        #fruit_locations.at[testDataIndexValue[i], 'predicted_y'] = fruit_locations[testDataIndexValue[i]]

#maxValueIndexObj = fruitCount_df.idxmax(axis=1)


#### -------- Declare Empty Dataframe to Store Predicted-Y, Actual-Y and K-Value -------- ####
comparison_df = pd.DataFrame(index=distance_df.index, columns=['Predicted Y','Actual Y', 'K-Value'])
maxValueIndex = fruitCount_df.max(axis=1)

for i in fruit_locations.index:
    if lemonLoc[i] == maxValueIndex[i]:
        comparison_df.at[i,'Predicted Y'] = 'lemon'
    elif orangeLoc[i]== maxValueIndex[i]:
        comparison_df.at[i,'Predicted Y'] = 'orange'
    elif appleLoc[i] == maxValueIndex[i]:
        comparison_df.at[i,'Predicted Y'] = 'apple'
    elif mandarinLoc[i] == maxValueIndex[i]:
        comparison_df.at[i,'Predicted Y'] = 'mandarin' 
    #elif mandarinLoc[i] > (len(fruit_locations.columns)/2):
#if lemonLoc >= 2:
#print(fruit_locations)
print(comparison_df)

counter = 0

for row in fruit_locations.index:
    #print(fruit_locations.at[row].values == 'lemon')
    #print(fruit_locations.iloc[counter,:])
    counter += 1

for i in range(len(fruit_locations)):
    prediction = fruit_locations.iloc[i,:]
    #for fruit in prediction:
        #fruitCount = fruitCount + fruit
## make an array of actual fruit & predicted fruit
    


#for row in range(len(distance_df)):
    #sorted_values.append(distance_df.iloc[row, np.argsort(distance_df.iloc[row,:])].head(k))
    #print(distance_df.sort_values([distance_df.columns], ascending=False))
    #print(distance_df.sort_values([i], ascending=False))
    #for i2 in distance_df.columns
#print(sorted_values.name)
        
## create a dictionary off the fruit names 
## create a dictionary out of the y-values 
## K 1-10 is how many neighbors we are picking
## then pick the one that is most common

## next find accuracy, number of correct y-value predictions in predicted data set which can be found
## by comparing predicted y-values to test-values 
## the complications happen when there are equidistant values 
