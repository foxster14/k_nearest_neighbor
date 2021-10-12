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

for i in range(len(fruitCount_df)):
    #print(fruitCount_df.iloc[i,:])
    temp = fruitCount_df.iloc[:, np.argsort(fruitCount_df.loc[testDataIndexValue[i]])]
    temp = temp.iloc[:,:k]
    sortedFruitCount = fruitCount_df.sort_values(by=testDataIndexValue[i], axis=1, ascending=False)
    #print(sortedFruitCount.iloc[i,:])
    maxValueIndex = sortedFruitCount.max(axis=1)
    #if maxValueIndex[testDataIndexValue[i]] > (len(fruit_locations.columns)/2):
        #fruit_locations.at[testDataIndexValue[i], 'predicted_y'] = fruit_locations[testDataIndexValue[i]]