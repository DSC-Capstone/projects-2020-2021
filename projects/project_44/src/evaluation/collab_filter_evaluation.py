from lightfm.evaluation import auc_score

def auc_eval(model, interactions, num_threads):
    auc = auc_score(model, interactions, num_threads=num_threads).mean()
    print(auc)