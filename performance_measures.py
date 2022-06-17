import numpy as np
import matplotlib.pyplot as plt
import itertools

## Confusion matrix

def create_confusion_matrix(Y_true, Y_pred ):
    n_classes = len(np.unique(Y_true))
    def foreachelem(i, j):
        arr = np.zeros(shape=(n_classes, n_classes))
        for i in range(n_classes):
            maski = Y_true == i
            for j in range(n_classes):
                maskj = Y_pred[maski] == j
                arr[i][j] = np.count_nonzero(maskj)
        return arr
    return np.fromfunction(foreachelem, shape=(n_classes, n_classes)).astype(int)


def confusion_matrix(Y_true, Y_pred,labels):
    cm = create_confusion_matrix(Y_true, Y_pred)
    n_classes = cm.shape[0]

    # Plot the figure and make it pretty
    fig, ax = plt.subplots(figsize=(10, 10))
    # colors will represent how 'correct' a class is, darker == better
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    # Label the axes
    ax.set(title="Confusion Matrix",
           xlabel="Predicted label",
           ylabel="True label",
           # create enough axis slots for each class
           xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=labels,  # axes will labeled with class names
           yticklabels=labels)

    # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

  # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]}",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black",
                 size=15)


## Rappel
def rappel(Y_true, Y_pred , verbose = False):
    # Rappel de classe = VP/(FN+VP)
    cm = create_confusion_matrix(Y_true, Y_pred)
    n_classes = cm.shape[0]
    rappels = np.array([])
    for i in range(n_classes):
        vp = cm[i,i]
        fn =  cm[i][cm[i]>0].sum() - vp
        if verbose : print(f"Rappel de la classe {i} = {float(vp)/(fn+vp)} , vp = {vp} , fn = {fn}")
        rappels = np.append(rappels,float(vp)/(fn+vp))
    print(f"Rappel systeme : {rappels.sum()/n_classes}")


# Precision
def precision(Y_true, Y_pred, verbose = False):
    # Precision de classe = VP/(VP+FP)
    cm = create_confusion_matrix(Y_true, Y_pred)
    n_classes = cm.shape[0]
    precisions = np.array([])
    for i in range(n_classes):
        vp = cm[i,i]
        fp =  cm[:,i][cm[:,i]>0].sum() - vp
        if verbose : print(f"Precision de la classe {i} = {float(vp)/(fp+vp)} , vp = {vp} , fp = {fp}")
        precisions = np.append(precisions,float(vp)/(fp+vp))
    print(f"Precision systeme : {precisions.sum()/n_classes}")

## False positif rate

def taux_de_fp(Y_true, Y_pred):
    # Taux de FP de classe = FP/(VP+FP)
    cm = create_confusion_matrix(Y_true, Y_pred)
    n_classes = cm.shape[0]

    for i in range(n_classes):
        vp = cm[i,i]
        fp =  cm[:,i][cm[:,i]>0].sum() - vp
        print(f"Taux de FP de la classe {i} = {float(fp)/(fp+vp)} , vp = {vp} , fp = {fp}")

## specificity
def specificite(Y_true, Y_pred):
    # Specificite de classe = VP/(VP+FP)
    cm = create_confusion_matrix(Y_true, Y_pred)
    n_classes = cm.shape[0]

    for i in range(n_classes):
        vp = cm[i,i]
        fp =  cm[:,i][cm[:,i]>0].sum() - vp
        print(f"Specificite de la classe {i} = {float(vp)/(fp+vp)} , vp = {vp} , fp = {fp}")

## ROC
def genXY(cm):
    VN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    VP = cm[1, 1]
    
    # Calculates tpr and fpr
    vpr =  VP/(VP + FN) # sensitivity - true positive rate
    fpr = 1 - VN/(VN+FP) # 1-specificity - false positive rate

    return fpr,vpr

def roc(Y_true, Y_proba_pred , graphResolition = 10): # for the binary
    x = []
    y = []

    for i in range(0,graphResolition) :
        thresh = i/graphResolition
        Y_Pred = Y_proba_pred>thresh
        xx,yy = genXY(create_confusion_matrix(Y_true,Y_Pred))
        x.append(xx)
        y.append(yy)
    
    plt.xlabel("Taux fp")
    plt.ylabel("Sens")
    plt.plot(x,y)
