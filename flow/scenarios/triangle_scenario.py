"""Contains the triangle scenario class"""

from flow.scenarios import MergeScenario
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from numpy import pi, sin, cos

INFLOW_EDGE_LEN = 200  # length of the inflow edges (needed for resets)
VEHICLE_LENGTH = 5

additional_net_params = {
    # length of the merge edge
    "merge_length": 100,
    # length of the highway leading to the merge
    "pre_merge_length": 200,
    # length of the highway past the merge
    "post_merge_length": 100,
    # number of lanes in the merge
    "merge_lanes": 2,
    # number of lanes in the highway
    "highway_lanes": 5,
    # max speed limit of the network
    "speed_limit": 30,
}

# we choose to make the main highway slightly longer
additional_net_params["pre_merge_length"] = 150

class TriangleMergeScenario(MergeScenario):
    
    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):
        """Initialize a merge scenario."""
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))
    
        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)

    
    def specify_nodes(self, net_params):
        angle = (2*pi)/3
        smaller_angle = pi / 3
        merge = net_params.additional_params["merge_length"]
        premerge = net_params.additional_params["pre_merge_length"]
        postmerge = net_params.additional_params["post_merge_length"]
        
        nodes = [
                 {
                 "id": "inflow_highway",
                 "x": -INFLOW_EDGE_LEN,
                 "y": 0
                 },
                 {
                 "id": "left",
                 "y": 0,
                 "x": 0
                 },
                 {
                 "id": "center",
                 "y": 0,
                 "x": premerge,
                 "radius": 10
                 },
                 {
                 "id": "right",
                 "y": 0,
                 "x": premerge + postmerge
                 },
                 {
                 "id": "inflow_merge",
                 "x": premerge - (merge + INFLOW_EDGE_LEN) * cos(angle),
                 "y": -(merge + INFLOW_EDGE_LEN) * sin(angle)
                 },
                 {
                 "id": "bottom",
                 "x": premerge - merge * cos(angle),
                 "y": -merge * sin(angle)
                 },{
                 "id": "center_2",
                 "y": 0,
                 "x": -INFLOW_EDGE_LEN-1,
                 "radius": 10
                 },{
                 "id": "inflow_merge_2",
                 "x": -INFLOW_EDGE_LEN-1 - (merge + INFLOW_EDGE_LEN) * cos(smaller_angle),
                 "y": -(merge + INFLOW_EDGE_LEN) * sin(smaller_angle)
                 },
                 {
                 "id": "bottom_2",
                 "x": -INFLOW_EDGE_LEN-1 - merge * cos(smaller_angle),
                 "y": -merge * sin(smaller_angle)
                 },{
                 "id": "inflow_highway_2",
                 "x": -3*INFLOW_EDGE_LEN,
                 "y": 0
                 },
                 {
                 "id": "left_2",
                 "y": 0,
                 "x": -INFLOW_EDGE_LEN-100
                 }
                 ]
                 
        return nodes
    
    def specify_edges(self, net_params):
        """See parent class."""
        merge = net_params.additional_params["merge_length"]
        premerge = net_params.additional_params["pre_merge_length"]
        postmerge = net_params.additional_params["post_merge_length"]
        
        edges = [{
                 "id": "inflow_highway",
                 "type": "highwayType",
                 "from": "inflow_highway",
                 "to": "left",
                 "length": premerge
                 }, {
                 "id": "left",
                 "type": "highwayType",
                 "from": "left",
                 "to": "center",
                 "length": premerge
                 }, {
                 "id": "inflow_merge",
                 "type": "mergeType",
                 "from": "bottom",
                 "to": "inflow_merge",
                 "length": INFLOW_EDGE_LEN
                 }, {
                 "id": "bottom",
                 "type": "mergeType",
                 "from": "center",
                 "to": "bottom",
                 "length": merge
                 }, {
                 "id": "center",
                 "type": "highwayType",
                 "from": "center",
                 "to": "right",
                 "length": postmerge
                 },
                 {
                 "id": "bottom_2",
                 "type": "mergeType",
                 "from": "bottom_2",
                 "to": "inflow_highway",
                 "length": merge
                 },{
                 "id": "inflow_merge_2",
                 "type": "mergeType",
                 "from": "inflow_merge_2",
                 "to": "bottom_2",
                 "length": INFLOW_EDGE_LEN
                 }, {
                 "id": "inflow_highway_2",
                 "type": "highwayType",
                 "from": "inflow_highway_2",
                 "to": "left_2",
                 "length": INFLOW_EDGE_LEN
                 }, {
                 "id": "left_2",
                 "type": "highwayType",
                 "from": "left_2",
                 "to": "inflow_highway",
                 "length": premerge
                 }]
                 
        return edges
    
    def specify_edge_starts(self):
        """See parent class."""
        premerge = self.net_params.additional_params["pre_merge_length"]
        postmerge = self.net_params.additional_params["post_merge_length"]
        
        edgestarts = [("inflow_highway", 0), ("left", INFLOW_EDGE_LEN + 0.1),
                      ("center", INFLOW_EDGE_LEN + premerge + 22.6),
                      ("inflow_merge",
                       INFLOW_EDGE_LEN + premerge + postmerge + 22.6),
                      ("bottom",
                       2 * INFLOW_EDGE_LEN + premerge + postmerge + 22.7),
                      #  ("center_2", -1),
                      ("inflow_merge_2", -1),
                      ("bottom_2", -2),
                      ("inflow_highway_2", -3),
                      ("left_2", -INFLOW_EDGE_LEN + 0.1)]
                      
        return edgestarts
    
    def specify_internal_edge_starts(self):
        """See parent class."""
        premerge = self.net_params.additional_params["pre_merge_length"]
        postmerge = self.net_params.additional_params["post_merge_length"]
        
        internal_edgestarts = [
                               (":left", INFLOW_EDGE_LEN), (":center",
                                                            INFLOW_EDGE_LEN + premerge + 0.1),
                               (":bottom", 2 * INFLOW_EDGE_LEN + premerge + postmerge + 22.6)
                               ]
                               
        return internal_edgestarts
    
    def specify_routes(self, net_params):
        """See parent class."""
        rts = {
            "inflow_highway": ["inflow_highway", "left"],
            "left": [(["left", "bottom"], .3), (["left", "center"], .7)],
            "center": ["center"],
            "inflow_merge": ["inflow_merge"],
            "bottom": ["bottom", "inflow_merge"],
            "inflow_merge_2": ["inflow_merge_2", "bottom_2"],
            "bottom_2": ["bottom_2", "inflow_highway"],
            "inflow_highway_2": ["inflow_highway_2", "left_2"],
            "left_2": ["left_2", "inflow_highway"]
        }
        
        return rts

