# split the train test
# then normalize the data
# for each test through, find the euclidean distance from each training rows & each data point
# arrange that vector in ascending order (don't reset index)
# if k=1 then classify it as apple
# add a column to the vector for distance 

import pandas as pd
from sklearn import linear_model
import sklearn.metrics as sk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import scipy.stats as stats

path = '/Users/sarahfox/OneDrive - Dunwoody College of Technology/Junior Year/Data Science/Lab 5/' #path where all files are stored
filename = 'fruits_classification.csv'
data_frame = pd.read_csv(path + filename)

## Using sci-kit to validate
x = data_frame[['mass','width','height','color_score']]
y = data_frame['fruit_label']

## normalize before splitting
x = x.apply(stats.zscore)

## Split the data 60/40
## randomly select the training & test data, choose row by row for test data to see the distances from the test data
## x-value, and y-value 
train, test = train_test_split(data_frame, train_size = 0.6, random_state = 0, shuffle=True)

# calculate the Euclidean distance between two vectors
def euclidean_distance(x1, x2, y1, y2, z1, z2, k1, k2):
	distance = 0.0
	distance += np.sqrt(((x1 - x2)**2) + ((y1 - y2)**2) + ((z1 - z2)**2) + ((k1 - k2)**2))
	return distance

distance_df = pd.DataFrame()
distance_df['distance'] = 0.0
distance_df['fruit_label'] = 0.0
distance_df['fruit_name'] = ''

#print(x_test.at[0,'width'].reset_index())
for index, row in test.iterrows():
    for index2, row2 in train.iterrows():
        distance = euclidean_distance(row['mass'], row2['mass'], row['width'], row2['width'], row['height'], row2['height'], row['color_score'], row2['color_score'])
        distance_df.loc[index, index2] = distance
print(distance_df)
    #distance = ((test.at[i,'mass'] - train.at[i, 'mass'])**2) + ((test.at[i,'width'] - train.at[i,'width'])**2) + ((test.at[i,'height'] - train.at[i,'height'])**2) + ((test.at[i,'color_score'] - train.at[i,'color_score'])**2)
    #distance = np.sqrt(distance)
    #distance_df.at[i,'distance'] = distance
    #distance_df.at[i,'fruit_label'] = train.at[i,'fruit_label']

## find the most occured value 
## find the frequency of each unique value, and choose the top frequency value
distance_df = distance_df.sort_values(by=18, ascending=True)
print(distance_df)


## K 1-10 is how many neighbors we are picking
## then pick the one that is most common

## next find accuracy, number of correct y-value predictions in predicted data set which can be found
## by comparing predicted y-values to test-values 
## the complications happen when there are equidistant values 
