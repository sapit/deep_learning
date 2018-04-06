from semEvalCat import *
from keras.models import load_model
from utils import *
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
np.random.seed(1337)
mymodel = load_model("data/emotion-detection-weights-improvement-100-0.1098.hdf5")

anger = json.load(open("evaluation_dataset2/anger_data.json", 'r'))
joy = json.load(open("evaluation_dataset2/joy_data.json", 'r'))
fear = json.load(open("evaluation_dataset2/fear_data.json", 'r'))
sadness = json.load(open("evaluation_dataset2/sadness_data.json", 'r'))

combined = json.load(open("evaluation_dataset2/combined.json", 'r'))

e_to_idx = {emotions[i]:i for i in range(len(emotions))}

def plot_conf_matrix(conf_arr, filename='confusion_matrix.png', alphabet=None):

    # conf_arr = [[33,2,0,0,0,0,0,0,0,1,3], 
    #         [3,31,0,0,0,0,0,0,0,0,0], 
    #         [0,4,41,0,0,0,0,0,0,0,1], 
    #         [0,1,0,30,0,6,0,0,0,0,1], 
    #         [0,0,0,0,38,10,0,0,0,0,0], 
    #         [0,0,0,3,1,39,0,0,0,0,4], 
    #         [0,2,2,0,4,1,31,0,0,0,2],
    #         [0,1,0,0,0,0,0,36,0,2,0], 
    #         [0,0,0,0,0,0,1,5,37,5,1], 
    #         [3,0,0,0,0,0,0,0,0,39,0], 
    #         [0,0,0,0,0,0,0,0,0,0,38]]

    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a) if a!=0 else 0)
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
                    interpolation='nearest')

    width, height = conf_arr.shape

    for x in range(width):
        for y in range(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center', 
                        color='white', 
                        # path_effects=[path_effects.withSimplePatchShadow()])
                        path_effects=[path_effects.Stroke(linewidth=1, foreground='black'), path_effects.Normal()])
                        

    cb = fig.colorbar(res)
    if alphabet is None:
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])
    plt.savefig(filename, format='png', bbox_inches='tight')

def compute_author_conf_matrix(author='computer', inorout='output'):
    conf_matrix = [ [0 for _ in combined] for _ in combined ]
    if(author == 'both'):
        author_emotion = [(i[inorout], e) for e in combined for a in combined[e] for i in combined[e][a] ]
    else:
        author_emotion = [(i[inorout], e) for e in combined for i in combined[e][author]]
    predicted_author_emotion = predictions_from_raw([i[0] for i in author_emotion] , mymodel, words)
    predicted_author_emotion = [(author_emotion[i][0], vector_to_emotion(predicted_author_emotion[i])) for i in range(len(predicted_author_emotion))]
    
    for i in range(len(predicted_author_emotion)):
        r = e_to_idx[predicted_author_emotion[i][1]]
        c = e_to_idx[author_emotion[i][1]]
        conf_matrix[r][c]+=1
    for i in conf_matrix:
        print (i)
    return(conf_matrix)

if __name__ == "__main__":
    folder = "evaluation_dataset2/"

    filename = "input_conf_matrix.png"
    plot_conf_matrix( np.array(compute_author_conf_matrix(inorout='input', author='both')), filename=folder+filename, alphabet=emotions)

    # filename = "human_conf_matrix.png"
    # plot_conf_matrix( np.array(compute_author_conf_matrix(author='human')), filename=folder+filename, alphabet=emotions)

    # filename = "computer_conf_matrix.png"
    # plot_conf_matrix( np.array(compute_author_conf_matrix(author='computer')), filename=folder+filename, alphabet=emotions)
