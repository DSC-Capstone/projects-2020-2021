# Parts of this code is borrowed from the original SHNE paper: https://github.com/chuxuzhang/WSDM2019_SHNE
import argparse
import json
import os


def read_args(**params):
    print("ARGS START")
    # with open("config/params.json", "r") as read_file:
    #     args = json.load(read_file)["shne-params"]
    args=params["shne-params"]
    
#     src=os.path.split(args["datapath"])[0]
    src=args["datapath"]
    fn=args["node_counts_filename"]
    print("src",src)
    print("fn", fn)
    fp=os.path.join(src,fn)
    print("fp", fp)
    with open(fp,"r") as read_file:
        vals = json.load(read_file)
        
    m2v_path=os.path.join(params["metapath2vec-params"]["save_dir"], params["metapath2vec-params"]["filename"])
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default=args["datapath"],
                    help='path to store data')
    parser.add_argument('--model_path', type=str, default=args["model_path"],
                    help='path to save model')

    #Automate this?
    parser.add_argument('--A_n', type = int, default = vals["api_call_nodes"],
                    help = 'number of api nodes')
    parser.add_argument('--P_n', type = int, default = vals["app_nodes"],
                    help = 'number of app nodes')
    parser.add_argument('--V_n', type = int, default = vals["package_nodes"],
                    help = 'number of package nodes')
    parser.add_argument('--B_n', type = int, default = vals["block_nodes"],
                    help = 'number of block nodes')
    
    parser.add_argument('--embed_d', type = int, default = params["word2vec-params"]["size"],
                    help = 'embedding dimension')
    parser.add_argument('--lr', type = int, default = args["learning_rate"],
                    help = 'learning rate')
    parser.add_argument('--mini_batch_s', type = int, default = args["mini_batch_size"],
                    help = 'mini batch_size')
    parser.add_argument('--batch_s', type = int, default = args["batch_size"],
                    help = 'batch_size')
    parser.add_argument('--train_iter_max', type = int, default = args["max_training_itter"],
                    help = 'max number of training iteration')
    parser.add_argument('--c_len', type = int, default = args["max_len_semantic"],
                    help = 'max len of semantic content')
    parser.add_argument('--save_model_freq', type = float, default = args["itter_save_freq"],
                    help = 'number of iterations to save model')
    parser.add_argument("--train", type=int, default= args["train"],
                    help = 'train/test label')
    parser.add_argument("--random_seed", type=int, default=args["random_seed"],
                    help = 'fixed random seed')
    parser.add_argument("--walk_l", type=int, default=args["walk_l"],
                    help = 'length of random walk')
    parser.add_argument("--win_s", type=int, default=args["window_s"],
                    help = 'window size for graph context')
    parser.add_argument("--cuda", type=int, default = args["cuda"],
                    help = 'GPU running label')
    parser.add_argument("--embeddings_filename", type=str, default=params["word2vec-params"]["embeddings_filename"],
                    help = 'Word Embeddings Filename')
    parser.add_argument("--content_filename", type=str, default=params["word2vec-params"]["content_filename"],
                    help = 'Content .pkl Filename') 
    parser.add_argument("--model_filename", type=str, default=args["model_filename"],
                    help = 'Content .pkl Filename')
    parser.add_argument("--m2v_walk", type=str, default=m2v_path, help="Metapath2vec walk file path")
    

    args, unknown = parser.parse_known_args()

    print("------arguments/parameters-------")
    for k, v in vars(args).items():
        print(k + ': ' + str(v))
    print("---------------------------------")

    return args