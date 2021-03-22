import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from sklearn.linear_model import LinearRegression

def RMSE(predictions, labels):
    """Function that takes in predictions and real values and computes RMSE."""
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return np.sqrt(sum(differences)/len(differences))

def main(configs):
    combined = pd.read_csv(configs['combined'])
    with open(configs['testSet'], 'rb') as filehandle:
        # read the data as binary data stream
        userItem = pickle.load(filehandle)
    UIR = combined[['user_id','recipe_id','rating']].values.tolist()
    allRecipes = combined.recipe_id.unique().tolist()
    uniqV = {}

    #Loops through user item pair and creates dictionaries for each user and all games they have played
    for x in UIR:
        if x[0] not in uniqV.keys():
            uniqV[x[0]] = [x[1]]
        else:
            uniqV[x[0]].append(x[1])
    
    recipeCount = defaultdict(int)
    totalTried = 0

    for x in UIR:
        recipeCount[x[1]] += 1
        totalTried += 1

    mostPopular = [(recipeCount[x], x) for x in recipeCount]
    mostPopular.sort()
    mostPopular.reverse()
    
    return1 = set()
    return2 = set()
    return3 = set()
    return4 = set()
    return5 = set()
    return6 = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        return1.add(i)
        if count > totalTried*.7: break

    count = 0
    for ic, i in mostPopular:
        count += ic
        return2.add(i)
        if count > totalTried*.16: break

    count = 0
    for ic, i in mostPopular:
        count += ic
        return3.add(i)
        if count > totalTried*.03: break

    count = 0
    for ic, i in mostPopular:
        count += ic
        return4.add(i)
        if count > totalTried*.02: break

    count = 0
    for ic, i in mostPopular:
        count += ic
        return5.add(i)
        if count > totalTried*.01: break

    count = 0
    for ic, i in mostPopular:
        count += ic
        return6.add(i)
        if count > totalTried*.05: break
            
    #The true values for the validation set
    trueTried = [d[2] for d in UIR]

    #The predicted value based on a 50th percentile threshold
    tryPred = []
    #for x in UIR:
    #    if x[1] in return5:
    #        tryPred.append(1)
    #    elif x[1] in return4:
    #        tryPred.append(2)
    #    elif x[1] in return3:
    #        tryPred.append(3)
    #    elif x[1] in return2:
    #        tryPred.append(4)
    #    elif x[1] in return1:
    #        tryPred.append(5)
    #    else:
    #        tryPred.append(0)

    for x in userItem:
        if x[1] in return5:
            tryPred.append(1)
        elif x[1] in return4:
            tryPred.append(2)
        elif x[1] in return3:
            tryPred.append(3)
        elif x[1] in return2:
            tryPred.append(4)
        elif x[1] in return1:
            tryPred.append(5)
        else:
            tryPred.append(0)
    
    ifTried = []
    for x in userItem:
        if x[1] in uniqV[x[0]]:
            ifTried.append(1)
        else:
            ifTried.append(0)
    
    for x in range(len(userItem)):
        print("Recommended Dish and Rating: " + str(userItem[x][1]) + ", " + str(tryPred[x]))
    
if __name__ == '__main__':
    main(sys.argv)