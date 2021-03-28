import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

#used in src/model/baseline_model.py, src/model/model.py, and src/analysis/compare.py
def visualize(name, test=False):
    
    if test:
        vis_path = 'test/'
    else:
        vis_path = 'data/visualizations/'
    
    plt.savefig(vis_path + name)
    
    return None

#used in src/model/baseline_model.py, src/model/model.py
def visualize_loss(model):
    
    plt.figure()
    plt.plot(model.history['loss'],label='Loss')
    plt.plot(model.history['val_loss'],label='Val. loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    
    return None
    
#used in src/model/baseline_model.py, src/model/model.py
def visualize_roc(fpr_cnn, tpr_cnn, fpr_dnn=None, tpr_dnn=None, fpr_gnn=None, tpr_gnn=None):
    sns.set_theme()
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle('ROC Curves by Class of Particle', fontsize=20)
    
    labels = ['QCD_b', 'QCD_bb', 'QCD_c', 'QCD_cc', 'QCD_other', 'H_bb']
    for i in range(len(labels)):
        ax = fig.add_subplot(2, 3, i + 1) #create a new set of axis at location i in the figure
    
        #ax.plot(tpr_dnn[i], fpr_dnn[i], lw=2.5, label="Dense, AUC = {:.1f}%".format(auc(fpr_dnn[i],tpr_dnn[i])*100))
        sns.lineplot(ax=ax, x=fpr_dnn[i], y=tpr_dnn[i], label="Dense, AUC = {:.1f}%".format(auc(fpr_dnn[i],tpr_dnn[i])*100))
        #ax.plot(tpr_cnn[i], fpr_cnn[i], lw=2.5, label="Conv1D, AUC = {:.1f}%".format(auc(fpr_cnn[i],tpr_cnn[i])*100))
        sns.lineplot(ax=ax, x=fpr_cnn[i], y=tpr_cnn[i], label="Conv1D, AUC = {:.1f}%".format(auc(fpr_cnn[i],tpr_cnn[i])*100))
        #ax.plot(tpr_gnn[i], fpr_gnn[i], lw=2.5, label="Graph, AUC = {:.1f}%".format(auc(fpr_gnn[i],tpr_gnn[i])*100))
        sns.lineplot(ax=ax, x=fpr_gnn[i], y=tpr_gnn[i], label="Graph, AUC = {:.1f}%".format(auc(fpr_gnn[i],tpr_gnn[i])*100))
                     
        
        ax.set_title(str(labels[i]))
        ax.set_ylabel(r'True positive rate')
        ax.set_yscale('log')
        ax.set_xlabel(r'False positive rate')
        ax.set_xscale('log')
        ax.set_ylim(0.001,1)
        ax.set_xlim(0.001,1)
        ax.grid(True)
        ax.legend(loc='lower right')
    
    plt.show()
    
    return None

#used in src/analysis/compare.py
def visualize_hist(data, weight_1, weight_2, bin_vars, x_label,y_label):
    
    plt.figure()
    plt.hist(data,weights=weight_1,bins=bin_vars,density=True,alpha=0.7,label='QCD')
    plt.hist(data,weights=weight_2,bins=bin_vars,density=True,alpha=0.7,label='H(bb)')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    
    return None

#used in src/analysis/compare.py
def visualize_roc_compare(fpr, tpr):
    
    plt.figure()
    plt.plot(fpr, tpr, lw=2.5, label="AUC = {:.1f}%".format(auc(fpr,tpr)*100))
    plt.xlabel(r'False positive rate')
    plt.ylabel(r'True positive rate')
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.plot([0, 1], [0, 1], lw=2.5, label='Random, AUC = 50.0%')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.show()
    
    return None