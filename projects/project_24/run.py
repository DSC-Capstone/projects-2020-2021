import sys
import subprocess
import src.etl.get_anames as gn
import src.etl.get_atexts as gt
import src.etl.get_ibc as gibc
import src.etl.get_revision_xmls as grx
import src.models.get_gns_scores as gns
import src.models.partyembed_ibc as pei
import src.models.partyembed_current_pages as pecp
import src.models.partyembed_revisions as per
import src.models.gns_histories as gnsh

args = sys.argv[1:]
fname = ""

if "scrape_anames" in args:
    fname = gn.scrape()
    
if "retrieve_anames" in args:
    if fname == "":
        artnames = gn.retrieve()
    else:
        artnames = gn.retrieve(fname)
        
if "ibc" in args:
    with open('config/get_ibc_params.json') as fh:
        data_cfg = json.load(fh)
        
    gibc.sample_ibc(**data_cfg)

if "interpret_ibc" in args:
    subprocess.call('git clone https://github.com/lrheault/partyembed.git', shell = True)
    with open('config/interpret_ibc_params.json') as fh:
        data_cfg = json.load(fh)
    
    pei.interpret_ibc(**data_cfg)

if "revision_xmls" in args:
    grx.main()

if "partyembed" in args:
    pecp.main()

if "partyembed_time" in args:
    per.main()
    
if "all" in args:
    subprocess.call('git clone https://github.com/lrheault/partyembed.git', shell = True)
    # fname = gn.scrape()
    # gt.scrape_atexts()
    nametxt_dict = gt.retrieve_atexts()
    namestat_dict = gns.get_stat_dict(nametxt_dict)
#     print(namestat_dict)


    # get IBC data
    with open('config/get_ibc_params.json') as fh:
        data_cfg = json.load(fh)
    gibc.sample_ibc(**data_cfg)
    
    # run on IBC
    print("Running model on IBC Data...")
    with open('config/interpret_ibc_params.json') as fh:
        data_cfg = json.load(fh)
    pei.interpret_ibc(**data_cfg)
    print("Finished IBC, output in test/out/means.csv")
    
    # run on current articles
    print("Running partyembed model on Current Page articles")
    pecp.main()
    
    #run on revision histories
    print("Running partyembed model on Revision Histories")
    per.main()


if "test" in args:
    subprocess.call('git clone https://github.com/lrheault/partyembed.git', shell = True)
    gn.scrape_anames()
    gibc.sample_ibc("False")
    print("Running model on test data...")
    pei.interpret_ibc(temp_directory="test/temp/", out_directory = 'test/out/', agg_func='mean',ibc_path='data/full_ibc/ibcData.pkl',test=True)
    print("Finished, output in test/out/means.csv")
    
    grx.test()
    
    #partyembed
    print("Partyembed: testing current page articles")
    pecp.test()
    print("Complete")
    print("Partyembed: testing revision histories")
    per.test()
    print("Complete")
    
    #G&S
    print("G&S: testing current page articles")
    nametxt_dict = gt.retrieve_atexts(test=True)
    namestat_dict = gns.get_stat_dict(nametxt_dict, test=True)
    print("Complete")
    print("G&S: testing revision histories")
    gnsh.get_all_history_stats(test = True)
    print("Complete")
    



