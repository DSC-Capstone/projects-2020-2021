import os
import time
from stellargraph.data import UniformRandomMetaPathWalk

def metapath2vec_walk(G, params):
    """Performs uniform random metapath walks using StellarGraph to generate corpus used in metapath2vec and writes corpus to a txt file 
    
    :param G : StellarGraph graph 
    Nodes consist of apps, api calls, packages, and invoke methods
    
    :param params : dict
    dict["key"] where dict is global parameter dictionary and key returns metapath2vec parameter sub-dictionary
    """
    fp=os.path.join(params["save_dir"],params["filename"])
    os.makedirs(params["save_dir"], exist_ok=True)
    # Create the random walker
    rw = UniformRandomMetaPathWalk(G,
        length=params["walk_length"],  # maximum length of a random walk
        n=params["n"],  # number of random walks per root node
        metapaths=params["metapaths"]  # the metapaths)
                                  )
    print("Starting MetaPath Walks")
    
    start_walks = time.time()
    walks = rw.run(
        nodes=list(G.nodes_of_type("app_nodes")),  # root nodes (app_nodes)
        length=params["walk_length"],  # maximum length of a random walk
        n=params["n"],  # number of random walks per root node
        metapaths=params["metapaths"]  # the metapaths
    )
    
    print("--- Done Walking in " + str(int(time.time() - start_walks)) + " Seconds ---")
    print()
    print("Number of metapath walks: {}".format(len(walks)))

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
