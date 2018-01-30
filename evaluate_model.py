import numpy as np

def custom_categorical_evaluation(model,X,Y):
    newy = map(lambda x: np.where(x==max(x))[0][0], Y)
    newx = map(lambda x: np.where(x==max(x))[0][0], model.predict(X))
    incorrect_count = sum([1 for i in range(len(newy)) if newx[i]!=newy[i]])
    # return sum([1 for i in range(len(newy)) if newx[i]!=newy[i]]) / float(len(Y))
    return (incorrect_count,len(Y), incorrect_count/float(len(Y)))

def custom_categorical_evaluation_best_only(model,X,Y):
    filtered_x, filtered_y = calc_best_only(model,X,Y)
    filtered_x = np.array(filtered_x)
    filtered_y = np.array(filtered_y)
    newx = map(lambda x: np.where(x==max(x))[0][0], model.predict(filtered_x))
    newy = map(lambda x: np.where(x==max(x))[0][0], filtered_y)
    
    # newx=filtered_x
    # newy=filtered_y
    
    incorrect_count = sum([1 for i in range(len(newy)) if newx[i]!=newy[i]])
    return (incorrect_count,len(newy), incorrect_count/float(len(newy)))

def calc_best_only(model,X,Y):
    newy = map(lambda x: np.where(x==max(x))[0][0], Y)
    predictions = model.predict(X)

    filtered_x=[]
    filtered_y=[]
    for i in range(len(predictions)):
        if(max(predictions[i]) >= 0.7):
            # filtered_x.append( np.where( predictions[i] == max(predictions[i]) )[0][0])
            # filtered_y.append(newy[i])
            filtered_x.append(X[i])
            filtered_y.append(Y[i])
    return filtered_x, filtered_y

def count_unique_mistakes_categorical(model,X,Y,hr_values):
    predictions = model.predict(X)
    new_predictions = map(lambda x: hr_values[np.where(x==max(x))[0][0]], predictions)
    newy = map(lambda x: hr_values[np.where(x==max(x))[0][0]], Y)
    z = zip(new_predictions, newy)
    return [[x, z.count(x)] for x in set(z) if(x[0] != x[1])]
    # return new_predictions, newy