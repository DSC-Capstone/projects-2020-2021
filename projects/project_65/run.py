import torch
import sys
import os.path as osp
import numpy as np
import yaml
from tqdm import tqdm
from torch_geometric.data import DataLoader
from src.LRP import LRP
from src.util import model_io,load_from,write_to,plot_static
from src.model import GraphDataset,InteractionNetwork,main
from src.sanity_check import make_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__=="__main__":
    # get the targets
    targets=sys.argv[1:]
   
    # load feature definitions
    with open('./data/definitions.yml') as file:
        definitions = yaml.load(file, Loader=yaml.FullLoader)
    
    features = definitions['features']
    spectators = definitions['spectators']
    labels = definitions['labels']

    nfeatures = definitions['nfeatures']
    nspectators = definitions['nspectators']
    nlabels = definitions['nlabels']
    ntracks = definitions['ntracks']
    
    if len(targets)==0:
        targets.append("all")

    if "test" in targets:                     # run targets on dev data
        file_names=["./test/test.root"]
        if len(targets)==1:
            targets.append("all")
    else:                                     # run targets on actual data
        file_names=["/teams/DSC180A_FA20_A00/b06particlephysics/test/ntuple_merged_0.root"]
    
    # start a model
    model=InteractionNetwork().to(device)
    

    # run targets related to actual usage of the project with trained model
    if not (("sanity-check" in targets) or ("sc" in targets)):  
        if "all" in targets:
            targets+=["explain","plot"]
        if "test" in targets:
            root="./test"
        else:
            root="./data"
        graph_dataset = GraphDataset(root, features, labels, spectators, n_events=10000, n_events_merge=1000, 
                                    file_names=file_names)
        
        batch=graph_dataset[0]
        batch_size=1
        batch_loader=DataLoader(batch,batch_size = batch_size)

        if "explain" in targets:    
            state_dict=torch.load("./data/model/IN.pth",map_location=device)
            model=model_io(model,state_dict,dict())

            t=tqdm(enumerate(batch_loader),total=len(batch)//batch_size)
            explainer=LRP(model)
            results=[]

            if "QCD" in targets:   # relevance w.r.t. QCD
                signal=torch.tensor([1,0],dtype=torch.float32).to(device)
                if "test" in targets:
                    save_to="./data/test_relevance_QCD.pt"
                else:
                    save_to="./data/file_0_relevance_QCD.pt"
            else:                  # default: relevance w.r.t. Hbb
                signal=torch.tensor([0,1],dtype=torch.float32).to(device)
                if "test" in targets:
                    save_to="./data/test_relevance.pt"
                else:
                    save_to="./data/file_0_relevance.pt"

            for i,data in t:
                data=data.to(device)
                to_explain={"A":dict(),"inputs":dict(x=data.x,
                                                    edge_index=data.edge_index,
                                                    batch=data.batch),"y":data.y,"R":dict()}
                
                model.set_dest(to_explain["A"])
                
                results.append(explainer.explain(to_explain,save=False,return_result=True,
                signal=signal))
                
            torch.save(results,save_to)
        
        if "plot" in targets:       # plot precomputed relevance scores
            if "test" in targets:
                path="./data/test_relevance.pt"
                plot_path="./test_relevance.png"
            else:
                path="./data/file_0_relevance.pt"
                plot_path="./file_0_jet_0.png"

            if osp.isfile(path):
                R=torch.load(path)
                plot_static(R,0,features,plot_path)
            else:
                print("relevance score not computed yet, need to run `explain` first")


    else: # run targets related to sanity check of the explanation method
        if len(targets)==1:
            targets.append("all")
        if "all" in targets:
            targets+=["data","train","explain","plot"]
        if "data" in targets:        # generate new data for sanity check purpose
            # declare variables
            nfeatures=48
            ntracks=10
            nsamples=2000+500
            x_idx=0
            y_idx=3
            save_to="./data/{}_sythesized.pt"
            make_data(nfeatures,ntracks,nsamples,x_idx,y_idx,save_to)

        if "train" in targets:       # train on generated train data
            train_data=torch.load("./data/{}_sythesized.pt".format("train"))
            test_data=torch.load("./data/{}_sythesized.pt".format("test"))

            main(train_data,test_data,"./data/model/IN_sythesized.pth","./data/IN_sythesized_roc.png")

        if "explain" in targets:     # explain the prediction on generated test data
            state_dict=torch.load("./data/model/IN_sythesized.pth",map_location=device)
            model=model_io(model,state_dict,dict())
            explainer=LRP(model)
            
            batch=torch.load("./data/test_sythesized.pt")
            g=batch[1]
            g.batch=torch.tensor(np.zeros(g.x.shape[0]).astype("int64"))
            
            results=[]
            signal=torch.tensor([0,1],dtype=torch.float32).to(device)
            save_to="./data/test_sythesized_relevance.pt"

            data=g.to(device)
            to_explain={"A":dict(),"inputs":dict(x=data.x,
                                                edge_index=data.edge_index,
                                                batch=data.batch),"y":data.y,"R":dict()}
                
            model.set_dest(to_explain["A"])
                
            results.append(explainer.explain(to_explain,save=False,return_result=True,
                                            signal=signal))

            torch.save(results,save_to)

        if "plot" in targets:        # plot the precomptued relevance score of generated test data
            if osp.isfile("./data/test_sythesized_relevance.pt"):
                R=torch.load("./data/test_sythesized_relevance.pt")
                plot_static(R,0,features,"./test_sythesized_jet_0.png")
            else:
                print("relevance score not computed yet, need to run `explain` first")