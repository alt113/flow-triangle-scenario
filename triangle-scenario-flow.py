from flow.scenarios import MergeScenario
from flow.controllers import ContinuousRouter, IDMController
from flow.core.params import SumoCarFollowingParams, SumoLaneChangeParams
from flow.core.params import VehicleParams
from flow.controllers import IDMController, RLController
from flow.core.params import SumoCarFollowingParams
from flow.core.params import InFlows
from flow.core.params import NetParams
from flow.scenarios.triangle_scenario import TriangleMergeScenario
from flow.core.params import SumoParams, EnvParams, InitialConfig
from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.core.experiment import Experiment
import sys
import json
import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments
from ray.tune.registry import register_env
from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from flow.scenarios.triangle_scenario import additional_net_params

"""
    Usage: python triangle-scenario-flow.py <mode> <path/to/csv-files>
    ----------------------------------------------------------------------
    Note: if mode == train then no need for path to csv file
"""

def triangle_scenario_example(highway_inflow,
                              middle_length,
                              emission_dir,
                              render_=False):
    # create an empty vehicles object
    vehicles = VehicleParams()

    # add some vehicles to this object of type "human"
    vehicles.add(
                 veh_id = "human",
                 acceleration_controller=(IDMController, {}),
                 routing_controller = (ContinuousRouter, {}),
                 car_following_params=SumoCarFollowingParams(
                                                             speed_mode="obey_safe_speed",
                                                             ),
                 lane_change_params=SumoLaneChangeParams(
                                                         lane_change_mode= "strategic",
                                                         ), num_vehicles = 0)

    inflow = InFlows()

    inflow.add(veh_type="human",
               edge="inflow_highway_2",
               vehs_per_hour=highway_inflow,
               departSpeed=10,
               departLane="random")

    inflow.add(veh_type="human",
               edge="inflow_merge_2",
               vehs_per_hour=500,
               departSpeed=10,
               departLane="random")

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
    additional_net_params["pre_merge_length"] = middle_length

    net_params = NetParams(inflows=inflow,  # our inflows
                           no_internal_links=False,
                           additional_params=additional_net_params)

    sumo_params = SumoParams(render=render_,
                             sim_step=0.2,
                             emission_path=emission_dir)

# '/Users/apple/Desktop/Berkeley/Repo/Flow/triange-data/'

    env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

    initial_config = InitialConfig(spacing="random", perturbation=1)

    scenario = TriangleMergeScenario(
                                     name="custom-triangle-merge-example",
                                     vehicles=vehicles,
                                     net_params=net_params,
                                     initial_config=initial_config,
                                     inflow_edge_len = middle_length)

    env = AccelEnv(env_params, sumo_params, scenario)

    return Experiment(env)

# TODO Do I add an emission variable here?
def stabilizing_triangle(highway_inflow):
    # experiment number
    # - 0: 10% RL penetration,  5 max controllable vehicles
    # - 1: 25% RL penetration, 13 max controllable vehicles
    # - 2: 33% RL penetration, 17 max controllable vehicles
    EXP_NUM = 0
    
    # time horizon of a single rollout
    HORIZON = 600
    # number of rollouts per training iteration
    N_ROLLOUTS = 20
    # number of parallel workers
    N_CPUS = 2
    
    # inflow rate at the highway
    FLOW_RATE = highway_inflow
    # percent of autonomous vehicles
    RL_PENETRATION = [0.1, 0.25, 0.33][EXP_NUM]
    # num_rl term (see ADDITIONAL_ENV_PARAMs)
    NUM_RL = [5, 13, 17][EXP_NUM]
    # We consider a highway network with an upstream merging lane producing
    # shockwaves

    # RL vehicles constitute 5% of the total number of vehicles
    vehicles = VehicleParams()
    vehicles.add(
                 veh_id = "human",
                 acceleration_controller=(IDMController, {}),
                 routing_controller = (ContinuousRouter, {}),
                 car_following_params=SumoCarFollowingParams(
                                                             speed_mode="obey_safe_speed",
                                                             ),
                 lane_change_params=SumoLaneChangeParams(
                                                         lane_change_mode= "strategic",
                                                         ), num_vehicles = 0)
    vehicles.add(
                 veh_id="rl",
                 acceleration_controller=(RLController, {}),
                 car_following_params=SumoCarFollowingParams(
                                                             speed_mode="obey_safe_speed",
                                                             ),
                 num_vehicles=0)

    # Vehicles are introduced from both sides of merge, with RL vehicles entering
    # from the highway portion as well
    inflow = InFlows()
    inflow.add(
               veh_type="human",
               edge="inflow_highway_2",
               vehs_per_hour=(1 - RL_PENETRATION) * FLOW_RATE,
               departLane="random",
               departSpeed=10)
    inflow.add(
               veh_type="rl",
               edge="inflow_highway_2",
               vehs_per_hour=RL_PENETRATION * FLOW_RATE,
               departLane="random",
               departSpeed=10)
    inflow.add(
               veh_type="human",
               edge="inflow_merge_2",
               vehs_per_hour=500,
               departLane="random",
               departSpeed=7.5)

    flow_params = dict(
                       # name of the experiment
                       exp_tag="stabilizing_triangle_merge",
                       
                       # name of the flow environment the experiment is running on
                       env_name="AccelEnv",
                       
                       # name of the scenario class the experiment is running on
                       scenario="TriangleMergeScenario",
                       
                       # simulator that is used by the experiment
                       simulator='traci',
                       
                       # sumo-related parameters (see flow.core.params.SumoParams)
                       sim=SumoParams(render=False,
                                      sim_step=0.2,
                                      emission_path='/Users/apple/Desktop/Berkeley/Repo/Flow/triange-data/'),
                       
                       # environment related parameters (see flow.core.params.EnvParams)
                       env=EnvParams(additional_params=ADDITIONAL_ENV_PARAMS),
                       
                       # network-related parameters (see flow.core.params.NetParams and the
                       # scenario's documentation or ADDITIONAL_NET_PARAMS component)
                       net=NetParams(inflows=inflow,  # our inflows
                                     no_internal_links=False,
                                     additional_params=additional_net_params),
                       
                       # vehicles to be placed in the network at the start of a rollout (see
                       # flow.core.params.VehicleParams)
                       veh=vehicles,
                       
                       # parameters specifying the positioning of vehicles upon initialization/
                       # reset (see flow.core.params.InitialConfig)
                       initial=InitialConfig(spacing="random", perturbation=1),
                       )

def setup_exps():
    alg_run = "PPO"
    
    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config["num_workers"] = N_CPUS
    config["train_batch_size"] = HORIZON * N_ROLLOUTS
    config["gamma"] = 0.999  # discount rate
    config["model"].update({"fcnet_hiddens": [32, 32, 32]})
    config["use_gae"] = True
    config["lambda"] = 0.97
    config["kl_target"] = 0.02
    config["num_sgd_iter"] = 10
    config['clip_actions'] = False  # FIXME(ev) temporary ray bug
    config["horizon"] = HORIZON
    
    # save the flow params for replay
    flow_json = json.dumps(
                           flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run
                           
    create_env, gym_name = make_create_env(params=flow_params, version=0)
                           
    # Register as rllib env
    register_env(gym_name, create_env)
    return alg_run, gym_name, config

def run_simulations(emission_dir):
    for x in range(1000,4000,200):
        for r in range(200,1000,200):
            exp = triangle_scenario_example(int(x),
                                            r,
                                            emission_dir,
                                            render_=False)
                                        # run for a set number of rollouts / time steps
            exp.run(1, 2000, convert_to_csv=True)

    print("Done simulating")

if __name__ == "__main__":
    argumentList = sys.argv
    varInflows = []
    path_to_emissions = ""
    
    for index, elem in enumerate(argumentList):
        if elem == sys.argv[0]:
            pass
        elif elem == "sample":
            # we wanna simulate to later plot
            run_simulations(sys.argv[index+1])
            break
        elif elem == "train":
            # we wanna train
            break

