from collections import defaultdict
import geopandas
import networkx as nx
class ManHoleGraph:
    # Assuming the network folder in the same path
    def __init__(self):
        self.graph = None
        self.df_pipe = geopandas.read_file('data/network/Sewer_Pipe.shp')
        self.df_manhole = geopandas.read_file('data/network/Sewer_Manhole.shp')
        self.manhole_to_coords_map = {elem['UCSD_ID']:(elem['geometry'].x,elem['geometry'].y) for _,elem in self.df_manhole[["UCSD_ID","geometry"]].iterrows()}
        self.coords_to_manhole_map = MirrorMap(self.manhole_to_coords_map).mirror
        self.edges = None
        self.graph_without_all_downstream = dict() # might not include all downstream but simplify graph
    def getSewerEdge(self):
        sewer_edges = defaultdict(set)
        for index, row in self.df_pipe.iterrows():
            for idx in range(1,len(row['geometry'].coords)):
                previous_coord, curr_coord = row['geometry'].coords[idx-1],row['geometry'].coords[idx]
                sewer_edges[previous_coord].add(curr_coord)
                graph_key = self.coords_to_manhole_map.get(previous_coord,previous_coord)
                graph_val = self.coords_to_manhole_map.get(curr_coord,curr_coord)
                self.graph_without_all_downstream[graph_key] =  graph_val
        self.edges = sewer_edges
    def getDownstream(self,seg_loc,visited,res):
        if seg_loc in visited: return
        visited.add(seg_loc)
        if seg_loc in self.coords_to_manhole_map: res.add(self.coords_to_manhole_map[seg_loc])
        for seg in self.edges[seg_loc]:
            self.getDownstream(seg,visited,res)
    def buildGraph(self):
        self.getSewerEdge()
        visited = set()
        graph = dict()
        for manhole_id in self.df_manhole['UCSD_ID']:
            if manhole_id not in visited:
                visited.add(manhole_id)
                component_sewer = set()
                self.getDownstream(self.manhole_to_coords_map[manhole_id],set(),component_sewer)
                graph[manhole_id] = component_sewer
        self.graph = graph
    def toNetworkGraph(self):
        return nx.DiGraph(self.graph)
    def toDF(self):
        G = self.toNetworkGraph()
        return nx.to_pandas_adjacency(G)
    def exportCSV(self):
        df = self.toDF()
        df.to_csv('data/downstream_graph.csv',index=False)
        print("graph csv has been saved to the data folder")
class MirrorMap:
    # take in a map and invert key and value
    def __init__(self,orginal):
        self.mirror = {value:key for key,value in orginal.items()}






