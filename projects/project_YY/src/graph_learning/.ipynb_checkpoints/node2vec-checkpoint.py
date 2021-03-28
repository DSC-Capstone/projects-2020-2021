import os
import time
from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk

def node2vec_walk(G, params):
    """Performs biased random walks using StellarGraph to generate corpus used in node2vec and writes corpus to a txt file 
    
    :param G : StellarGraph graph 
    Nodes consist of apps, api calls, packages, and invoke methods
    
    :param params : dict
    dict["key"] where dict is global parameter dictionary and key returns node2vec parameter sub-dictionary
    """
    start_walks = time.time()
    print("Starting Random Walks")
    
    rw = BiasedRandomWalk(G)
    fp=os.path.join(params["save_dir"],params["filename"])
    os.makedirs(params["save_dir"], exist_ok=True)

    walks = rw.run(nodes=list(G.nodes(node_type="app_nodes")),  # root nodes
        length=params["length"],  # maximum length of a random walk
        n=params["n"],  # number of random walks per root node
        p=params["p"], # Defines prob, 1/p, of returning to source node
        q=params["q"], # Defines prob, 1/q, for moving away from source node
    )
    print("--- Done Walking in " + str(int(time.time() - start_walks)) + " Seconds ---")
    print()
    print("Number of random walks: {}".format(len(walks)))
    
    # save walks to file
    with open(fp, 'w') as f:
        for walk in walks:
            for node in walk:
                f.write(str(node) + ' ')
            f.write('\n')
    f.close()
    
    if params["verbose"]:
        print("Saved %s to %s" %(params["filename"], params["save_dir"]))
    
    return
    