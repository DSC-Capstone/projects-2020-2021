import geopandas
from collections import defaultdict
import networkx as nx
class MirrorMap:
    # take in a map and invert key and value
    def __init__(self,original,one_to_one_set=False):
        try:
            self.mirror = {value:key for key,value in original.items()}
        except TypeError:
            if not one_to_one_set:
                self.mirror = defaultdict(set)
                for key,neighbors in original.items():
                    for n in neighbors:
                        self.mirror[n].add(key)
            else:
                self.mirror = dict()
                for key,neighbors in original.items():
                    for n in neighbors:
                        self.mirror[n] = key
class TraceGraph:
    # Assuming the network folder in the same path
    def __init__(self):
        self.graph = None
        self.manhole_graph = None
        self.trace_graph = None
        self.df_pipe = geopandas.read_file('../data/network/Sewer_Pipe.shp')
        self.df_manhole = geopandas.read_file('../data/network/Sewer_Manhole.shp')
        self.df_buildings= geopandas.read_file('../data/network/Sewer_Buildings.shp')
        self.manhole_to_coords_map = {elem['UCSD_ID']:(elem['geometry'].x,elem['geometry'].y) for _,elem in self.df_manhole[["UCSD_ID","geometry"]].iterrows()}
        self.coords_to_manhole_map = MirrorMap(self.manhole_to_coords_map).mirror
        self.build_to_coords_map = defaultdict(set)
        for _,elem in self.df_buildings[["CAANtext","geometry"]].iterrows():
            if elem['CAANtext']:self.build_to_coords_map[elem['CAANtext']].add((elem['geometry'].x,elem['geometry'].y))
        self.coords_to_build_map = MirrorMap(self.build_to_coords_map,True).mirror
        self.edges = None
        self.mirror_edges = None
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
        self.mirror_edges = MirrorMap(self.edges).mirror
    def getFlow(self,seg_loc,visited,res,origin,mode="downstream",only_mode=None):
        if seg_loc in visited: return
        visited.add(seg_loc)
        if (seg_loc in self.coords_to_manhole_map) and (self.coords_to_manhole_map[seg_loc] != origin) and ((not only_mode) or only_mode == "manhole"): 
            res.add(self.coords_to_manhole_map[seg_loc])
        if (seg_loc in self.coords_to_build_map) and (self.coords_to_build_map[seg_loc] != origin)  and ((not only_mode) or only_mode == "build"): 
            res.add(self.coords_to_build_map[seg_loc])
        if mode == "downstream":
            for seg in self.edges[seg_loc]:
                self.getFlow(seg,visited,res,origin,mode,only_mode)
        else:
            for seg in self.mirror_edges[seg_loc]:
                self.getFlow(seg,visited,res,origin,mode,only_mode)
    def buildGraph(self):
        self.getSewerEdge()
        visited = set()
        graph = dict()
        # Build downward manhole graph
        for manhole_id in self.df_manhole['UCSD_ID']:
            if manhole_id not in visited:
                visited.add(manhole_id)
                component_sewer = set()
                self.getFlow(self.manhole_to_coords_map[manhole_id],set(),component_sewer,manhole_id,"downstream","manhole")
                graph[manhole_id] = component_sewer
        self.manhole_graph = graph
        # Build upward tracing graph
        visited = set()
        graph = dict()
        for manhole_id in self.df_manhole['UCSD_ID']:
            if manhole_id not in visited:
                visited.add(manhole_id)
                component_sewer = set()
                self.getFlow(self.manhole_to_coords_map[manhole_id],set(),component_sewer,manhole_id,"upstream","build")
                graph[manhole_id] = component_sewer
        self.trace_graph = graph
        # Build the main graph
        visited = set()
        graph = defaultdict(set)
        for key,value in self.mirror_edges.items():
            temp_key = key
            if (key in self.coords_to_build_map):temp_key = self.coords_to_build_map[key]
            if (key in self.coords_to_manhole_map):temp_key = self.coords_to_manhole_map[key]
            for val in value:
                temp_val = val
                if (val in self.coords_to_build_map):temp_val = self.coords_to_build_map[val]
                if (val in self.coords_to_manhole_map):temp_val = self.coords_to_manhole_map[val]
                graph[temp_key].add(temp_val)
        self.graph = graph
    def toNetworkGraph(self,g_type="manhole"):
        if g_type == "manhole":
            return nx.DiGraph(self.manhole_graph)
        if g_type == "trace":
            return nx.DiGraph(self.trace_graph)
        return nx.DiGraph(self.graph)
    def toDF(self):
        G = self.toNetworkGraph()
        return nx.to_pandas_adjacency(G)
    def exportCSV(self):
        df = self.toDF()
        df.to_csv('data/downstream_graph.csv',index=False)
        print("graph csv has been saved to the data folder")
