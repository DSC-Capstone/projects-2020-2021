import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d.axes3d import Axes3D
import networkx as nx
from torch_geometric.utils import to_networkx
from bokeh.io import save
from bokeh.models import BasicTicker, ColorBar, ColumnDataSource,LinearColorMapper, PrintfTickFormatter
from bokeh.plotting import figure
from bokeh.transform import transform

sns.set(style="white")

def plot_static(R,ix,features,save_to="jet_0.png"):
    r=(R[ix]["node"]).clone()
    r[torch.isnan(r)]=0

    val=r.detach().cpu().numpy()
    # val=val[sort_pt_idx]
    df=pd.DataFrame(val,columns=features).T

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(8,8))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    s=sns.heatmap(df, cmap=cmap, center=0,yticklabels=1,
                square=True, linewidths=0.01, cbar_kws={"shrink": .5})

    s.tick_params(labelsize=7)

    
    if R[ix]['label'][:,1]>0:
        title_str="node feature relevance of a signal data sample\n\n"
    else:
        title_str="node feature relevance of a background data sample\n\n"

    plt.title(title_str+"prediction:{}".format(R[ix]["pred"].detach().cpu().numpy().round(4)))
    plt.savefig(save_to)

def plot_interactive(R,ix,raw_input,features,save_to="jet_0.html"):
    r=(R[ix]["node"]).clone()
    r[torch.isnan(r)]=0

    val=r.detach().cpu().numpy()
    data=pd.DataFrame(val,columns=features)
    data.columns.name="feature"
    data.index.name="particle"
    
    df = pd.DataFrame(data.stack(), columns=['relevance']).reset_index()

    raw=raw_input[ix].x
    sort_idx=torch.argsort(raw[:,0])
    raw=raw[sort_idx]
    df["raw data"]=raw.reshape(-1,1).clone().detach().numpy()
    
    df["particle"]=df["particle"].astype(str)
    df.drop_duplicates(['particle','feature'],inplace=True)
    data = data.T.reset_index().drop_duplicates(subset='feature').set_index('feature').T
    source = ColumnDataSource(df)
    
    colors=[]
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    for i in range(cmap.N):
        rgba = cmap(i)
        colors.append(matplotlib.colors.rgb2hex(rgba))

    scale=max(np.abs(df.relevance.min()),np.abs(df.relevance.max()))
    mapper = LinearColorMapper(palette=colors, low=-scale, high=scale)

    if R[ix]['label'][:,1]>0:
        title_str="Node Relevance Heatmap for Higgs boson Signal"
    else:
        title_str="Node Relevance Heatmap for background"

    subtitle_str="prediction:{}".format(R[ix]["pred"].detach().cpu().numpy().round(4)[0])


    p = figure(x_range=[str(i) for i in data.index],
               y_range=list(reversed(data.columns)),
               tools=["hover","save"])
    
    p.add_layout(Title(text=subtitle_str, text_font_style="italic"), 'above')
    p.add_layout(Title(text=title_str, text_font_size="12pt"), 'above')

    p.rect(x="particle", y="feature", width=1, height=1, source=source,
           line_color='white', fill_color=transform('relevance', mapper))

    p.hover.tooltips = [
        ("particle", "@particle"),
        ("feature", "@feature"),
        ("relevance score", "@relevance"),
        ("input data","@{raw data}")
    ]

    color_bar = ColorBar(color_mapper=mapper,
                         ticker=BasicTicker(desired_num_ticks=10),
                         location=(0,0),
                         formatter=PrintfTickFormatter(format="%d"))

    p.add_layout(color_bar,'right')
    save(p,save_to)

def network_plot_3D(G, angle,label, edge_alpha,title_str,zlabel="track_pt",threshold=0.5,save_to="jet_0_edge3d.png"):
    pos = nx.get_node_attributes(G, 'pos')
    node_shade=nx.get_node_attributes(G,"node_shade")

    # 3D network plot
    with plt.style.context(('ggplot')):
        
        fig = plt.figure(figsize=(10,7))
        ax = Axes3D(fig)
        
        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        for key, value in pos.items():
            xi = value[0]
            yi = value[1]
            zi = value[2]
            
            # Scatter plot
            ax.scatter(xi, yi, zi, c='k',s=25*node_shade[key], edgecolors='orange', alpha=0.5)
        

        for i,j in enumerate(G.edges()):

            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))

            alpha=edge_alpha[i]
            if alpha<threshold:
                ax.plot(x, y, z, c='#546163', alpha=0.03)
            else:
                ax.plot(x, y, z, c='r', alpha=alpha,linewidth=2)
    
    # Set the initial view
    ax.view_init(30, angle)

    plt.xlabel("track_etarel")
    plt.ylabel("track_phirel")
    ax.set_zlabel(zlabel)

    
    plt.title(title_str)

    plt.savefig(save_to)


def plot_edge3d(R,ix,raw_input,features,x,y,z):
    r_node=R['node'].clone()
    r_edge=R['edge'].clone()
    r_node[torch.isnan(r_node)]=0
    r_edge[torch.isnan(r_edge)]=0

    edge_shade=torch.norm(r_edge,dim=1)
    edge_alpha=(edge_shade-edge_shade.min())/edge_shade.max()
    edge_alpha=edge_alpha.detach().cpu().numpy()

    node_shade=np.linalg.norm(r_node.detach().cpu().numpy(),axis=1)
    
    x_idx=features.index(x)
    y_idx=features.index(y)
    z_idx=features.index(z)
    
    raw_input.edge_alpha=edge_alpha
    raw_input.node_shade=node_shade

    pos=np.array(list(zip(raw_input.x[:,x_idx].detach().numpy(),
                          raw_input.x[:,y_idx].detach().numpy(),
                          raw_input.x[:,z_idx].detach().numpy())))
    raw_input.pos=pos
    
    G = to_networkx(raw_input, node_attrs=["pos","node_shade"])

    if R[ix]['label'][:,1]>0:
        title_str="Edge and Node Relevance in 3d Space for Higgs boson Signal\n\n"
    else:
        title_str="Edge and Node Relevance in 3d Space for background\n\n"
    title_str+="prediction:{}".format(R[ix]["pred"].detach().cpu().numpy().round(4)[0])
    
    network_plot_3D(G,45,raw_input.y.detach(),edge_alpha,title_str,zlabel=z)


def plot(R,ix,raw_input,features,save_to=f"jet_0"):
    plot_static(R,ix,features,save_to+".png")
    plot_interactive(R,ix,raw_input,features,save_to+".html")
    plot_edge3d(R,ix,raw_input,features,save_to+"_edge3d.png")

