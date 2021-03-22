import numpy as np
import matplotlib.pyplot as plt

def offset(X, y, offset, f):
    if offset >= 0:
        return f(X.shift(offset)[offset:], y[offset:])
    return f(X.shift(offset)[:offset], y[:offset])

def showLoss(X,y,lower,upper,fs):
    spearmanPearsonLosses = []
    spearmanLosses = []
    pearsonLosses = []
    for i in range(lower, upper):
        print(f"Offset: {i}. \n\tSpearman/Pearson Loss Score: \
            {round(offset(X, y, i, fs[0]), 2)}\n\tPearson Loss Score: \
            {round(offset(X, y, i, fs[1]), 2)}\n\tSpearman Loss Score: \
            {round(offset(X, y, i, fs[2]), 2)}")
        spearmanPearsonLosses.append(offset(X, y, i, fs[0]))
        pearsonLosses.append(offset(X,y,i,fs[1]))
        spearmanLosses.append(offset(X,y,i, fs[2]))
    SPindex = np.argmin(spearmanPearsonLosses)
    Pindex = np.argmin(pearsonLosses)
    Sindex = np.argmin(spearmanLosses)
    return {"sp": SPindex + lower, "p": Pindex + lower, "s": Sindex + lower}

def makePlot(spOffset, pOffset, sOffset, fileName, df):
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.plot(df["Date"], df["cases_reported"], linewidth=5, label="Case Reported")
    plt.plot(df["Date"], df["cases_specimen"].shift(spOffset), linestyle="--", linewidth=3, label="Highest Spearman/Pearson Combo")
    plt.plot(df["Date"], df["cases_specimen"].shift(sOffset), linestyle=":", linewidth=3, label="Highest Spearman")
    plt.plot(df["Date"], df["cases_specimen"].shift(pOffset), linestyle=":", linewidth=3, label="Highest Pearson")

    plt.legend()
    plt.savefig(f"figures/{fileName}")
