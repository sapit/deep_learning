import numpy as np
import os
import matplotlib.pyplot as plt
from utils import *
import read_dataset as rd
from keras.models import load_model
from natsort import natsorted, ns

def custom_categorical_evaluation(model,X,Y):
    newy = list(map(lambda x: np.where(x==max(x))[0][0], Y))
    newx = list(map(lambda x: np.where(x==max(x))[0][0], model.predict(X)))
    incorrect_count = sum([1 for i in range(len(newy)) if newx[i]!=newy[i]])
    # return sum([1 for i in range(len(newy)) if newx[i]!=newy[i]]) / float(len(Y))
    return (incorrect_count,len(Y), incorrect_count/float(len(Y)))

def custom_categorical_evaluation_best_only(model,X,Y):
    filtered_x, filtered_y = calc_best_only(model,X,Y)
    if(len(filtered_x) == 0):
        return None
    filtered_x = np.array(filtered_x)
    filtered_y = np.array(filtered_y)
    newx = list(map(lambda x: np.where(x==max(x))[0][0], model.predict(filtered_x)))
    newy = list(map(lambda x: np.where(x==max(x))[0][0], filtered_y))
    
    # newx=filtered_x
    # newy=filtered_y
    
    incorrect_count = sum([1 for i in range(len(newy)) if newx[i]!=newy[i]])
    return (incorrect_count,len(newy), incorrect_count/float(len(newy)))

def calc_best_only(model,X,Y):
    newy = list(map(lambda x: np.where(x==max(x))[0][0], Y))
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
    new_predictions = list(map(lambda x: hr_values[np.where(x==max(x))[0][0]], predictions))
    newy = list(map(lambda x: hr_values[np.where(x==max(x))[0][0]], Y))
    z = zip(new_predictions, newy)
    return [[x, z.count(x)] for x in set(z) if(x[0] != x[1])]
    # return new_predictions, newy

def try_folder(folder, model,X,Y):
    if folder[-1]!='/':
        folder += '/'
    files = natsorted(os.listdir(folder))
    for f in files:
        print(f)
        model.load_weights(folder+f)
        print("Filename: %s"%(f))
        print("Evaluation: %s" % (str(custom_categorical_evaluation(model,X,Y))))
        print("Evaluation(best only): %s\n" % (str(custom_categorical_evaluation_best_only(model,X,Y))))

def plot_error_folder(folder, model,X,Y):
    if folder[-1]!='/':
        folder += '/'
    files = natsorted(os.listdir(folder))

    array_to_plot = []
    array_to_plot2 = []
    array_to_plot3 = []
    array_to_plot3_2 = []
    epoch=0

    model= load_model(folder + files[0])
    
    for f in files:
        epoch+=1
        print(f)
        model.load_weights(folder+f)
        print("Filename: %s"%(f))
        res2 = custom_categorical_evaluation(model,X,Y)
        print("Evaluation: %s" % (str(res2)))
        res = custom_categorical_evaluation_best_only(model,X,Y)
        # if(res == None or float(res[1])<100):
        if(res == None):
            continue
        (mistaken, selected, error) = res
        print("Evaluation(best only): %s\n" % (str((mistaken, selected, error))))

        # (mistaken, selected, error) = custom_categorical_evaluation_best_only(model,X,Y)
        plot_x = epoch
        plot_y = (1 - selected/len(X))*(1-error)
        array_to_plot.append( (plot_x, plot_y) )

        plot_x = epoch
        plot_y = (1 - (2*selected)/len(X))*(1-error)
        array_to_plot2.append( (plot_x, plot_y) )

        plot_x = epoch
        plot_y = 1 - error
        plot_z = selected/len(X)
        array_to_plot3.append((plot_x, plot_y, plot_z))
        
        plot_x_2 = epoch
        plot_y_2 = 1 - res2[2]
        array_to_plot3_2.append( (plot_x_2, plot_y_2) )
 
    # x,y = zip(*array_to_plot)
    # print(x)
    # print(y)
    # plt.plot(x,y, 'rx')

    # x,y = zip(*array_to_plot2)
    # print(x)
    # print(y)
    # plt.plot(x,y, 'bx')

    # plt.show()

    x,y=map(np.array, zip(*array_to_plot3_2))
    plt.scatter(x,y,)
    plt.show()
    
    x,y,z = map(np.array, zip(*array_to_plot3))
    plt.scatter(x,y, s=((z)**2)*6000)
    plt.show()

    return(array_to_plot, array_to_plot2, array_to_plot3)

def plot_error_folder_with_smile(folder, model,words):
    X,Y = eval_with_smile(words)
    return plot_error_folder(folder, model, X,Y)

def eval_with_smile(words):
    onehot = {
        "happy": [1,0,0,0],
        "sad": [0,1,0,0],
        "angry":[0,0,1,0]
    }
    df = rd.readSmileDatasetDf()
    X=[]
    Y=[]
    for i,row in df.iterrows():
        pass
        if(row["emotion"] in onehot.keys()):
            vector = vectoriseSentence(processMessage(process_tweet(row["tweet"]),[]),words)
            X.append(vector)
            Y.append(onehot[row["emotion"]])
    return np.array(X),np.array(Y)