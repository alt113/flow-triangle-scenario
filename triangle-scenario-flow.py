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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as colors
import matplotlib.patches as patches
import pickle
from scipy.interpolate import griddata

"""
    Usage: python triangle-scenario-flow.py <mode> <path/to/csv-files>
    ----------------------------------------------------------------------
    *Note: if mode == train then no need for path to csv file
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

"""
    ---------------------------------------------------------------------------
    The following section is responsible for plotting the results from sampling
    ---------------------------------------------------------------------------
"""

cdict = {'red'  :  ((0., 0., 0.), (0.2, 1., 1.), (0.6, 1., 1.), (1., 0., 0.)),
    'green':  ((0., 0., 0.), (0.2, 0., 0.), (0.6, 1., 1.), (1., 1., 1.)),
    'blue' :  ((0., 0., 0.), (0.2, 0., 0.), (0.6, 0., 0.), (1., 0., 0.))}

my_cmap = colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

def space_time_diagram(pos, speed, time, title, max_speed=8):
    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes()
    norm = plt.Normalize(0, max_speed) # TODO: Make this more modular
    cols = []
    
    for indx_car in range(pos.shape[1]):
        unique_car_pos = pos[:,indx_car]
        
        # discontinuity from wraparound
        disc = np.where(np.abs(np.diff(unique_car_pos)) >= 100)[0]+1
        unique_car_time = np.insert(time, disc, np.nan)
        unique_car_pos = np.insert(unique_car_pos, disc, np.nan)
        unique_car_speed = np.insert(speed[:,indx_car], disc, np.nan)
        
        points = np.array([unique_car_time, unique_car_pos]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=my_cmap, norm=norm)
        
        # Set the values used for colormapping
        lc.set_array(unique_car_speed)
        lc.set_linewidth(1.75)
        cols = np.append(cols, lc)
    
    xmin, xmax = min(time), max(time)
    xbuffer = (xmax - xmin) * 0.025 # 2.5% of range
    ymin, ymax = np.amin(pos), np.amax(pos)
    ybuffer = (ymax - ymin) * 0.025 # 2.5% of range
    
    ax.set_xlim(xmin - xbuffer, xmax + xbuffer)
    ax.set_ylim(ymin - ybuffer, ymax + ybuffer)
    
    plt.title(title, fontsize=25)
    plt.ylabel('Position (m)', fontsize=20)
    plt.xlabel('Time (s)', fontsize=20)
    
    for col in cols:
        line = ax.add_collection(col)
    cbar = plt.colorbar(line, ax=ax)
    cbar.set_label('Velocity (m/s)', fontsize=20)
    cbar.ax.tick_params(labelsize=18)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    #     # NOTE: FOR OPEN NETWORK MERGE ONLY
    #     plt.plot(np.arange(pos.shape[0])*dt, [600]*pos.shape[0], "--", linewidth=2, color="k")
    plt.plot(np.arange(pos.shape[0])*dt, [0]*pos.shape[0], linewidth=3, color="white")
    plt.plot(np.arange(pos.shape[0])*dt, [-0.1]*pos.shape[0], linewidth=3, color="white")

    plt.show()

def import_edge_pos_speed(csv_filepath):
    """Import edge, position, and speed data from an emission csv.
        
        Parameters
        ----------
        csv_filepath : str
        location of the emission csv file
        
        Returns
        -------
        dict < dict >
        Key = vehicle id
        Key = "edge", "pos", "speed", "lane", or "time"
        Element = list of specified data for vehicle
        """
    # import the csv file into a pandas dataframe
    data = pd.read_csv(csv_filepath)
    
    # create an empty dict for the results
    res = dict().fromkeys(np.unique(data["id"]))
    
    for veh_id in res.keys():
        res[veh_id] = dict()
        indx_data = data["id"] == veh_id
        res[veh_id]["edge"] = list(data["edge_id"][indx_data])
        res[veh_id]["pos"] = list(data["relative_position"][indx_data])
        res[veh_id]["speed"] = list(data["speed"][indx_data])
        res[veh_id]["lane"] = list(data["lane_number"][indx_data])
        res[veh_id]["time"] = list(data["time"][indx_data])
    
    return res

inflow_edge_len = 100
merge = 100
premerge = 500
postmerge = 100
total_len = merge + premerge + postmerge + 2*inflow_edge_len + 8.1

edgestarts_merge = {
    "inflow_highway": 0,
    "left": inflow_edge_len + 0.1,
    "center": inflow_edge_len + premerge + 8.1,
    "inflow_merge": inflow_edge_len + premerge + postmerge + 8.1,
    "bottom": 2*inflow_edge_len + premerge + postmerge + 8.2,
    ":left_0": inflow_edge_len,
    ":center_0": inflow_edge_len + premerge + 0.1,
    ":center_1": inflow_edge_len + premerge + 0.1,
    ":bottom_0": 2*inflow_edge_len + premerge + postmerge + 8.1
}

num_steps = 18000  # number of simulation steps
offset = 0  # number of steps offset from the beginning
dt = 0.2  # simulation step size

def get_speed_pos_merge(merge_data,
                        edgestarts,
                        num_steps,
                        dt):
    """Compute the absolute position, speed, average speed from
        merge data.
        
        The data is converted to a representation that is compatible with
        plotting methods in place (e.g. space-time diagram).
        """
    # compute the absolute position
    for veh_id in merge_data.keys():
        merge_data[veh_id]["abs_pos"] = []
        for edge, pos in zip(merge_data[veh_id]["edge"],
                             merge_data[veh_id]["pos"]):
            merge_data[veh_id]["abs_pos"].append(pos + edgestarts[edge])

    # prepare the speed and absolute position in a way that is compatible
    # with the space-time diagram, and compute the number of vehicles at
    # each step
    pos = np.zeros((num_steps, len(merge_data.keys())))
    speed = np.zeros((num_steps, len(merge_data.keys())))
    num_veh = [0 for _ in range(num_steps)]
    for i, veh_id in enumerate(merge_data.keys()):
        for spd, abs_pos, ti, edge in zip(merge_data[veh_id]["speed"],
                                          merge_data[veh_id]["abs_pos"],
                                          merge_data[veh_id]["time"],
                                          merge_data[veh_id]["edge"]):
            if int(ti*(1/dt)) >= num_steps \
                or edge in ["inflow_merge", "bottom", ":bottom_0"]:
                    continue
            speed[int(ti*(1/dt)), i] = spd
                pos[int(ti*(1/dt)), i] = abs_pos
                num_veh[int(ti*(1/dt))] += 1

    # compute the average speed
    avg_speed = np.sum(speed, axis=1) / num_veh

    return pos, speed, avg_speed, num_veh

def speed_and_density_plots():
    # baseline simulation results
    avg_speed = []
    num_veh = []
    num_enter_baseline = []
    num_exit_baseline = []
    for i in range(10):
        print(i)
        baseline = import_edge_pos_speed("merge-baseline-emission-2000-{}.csv".format(i))
        
        # conversion to acceptable forms for plotting
        pos_baseline, speed_baseline, avg_speed_baseline, n_veh = get_speed_pos_merge(
                                                                                      merge_data=baseline,
                                                                                      edgestarts=edgestarts_merge,
                                                                                      num_steps=num_steps,
                                                                                      dt=dt
                                                                                      )
                                                                                      avg_speed.append(avg_speed_baseline)
                                                                                      num_veh.append(n_veh)
                                                                                      num_enter_baseline.append(pos_baseline.shape[1])
                                                                                      num_exit_baseline.append(sum(pos_baseline[-1,:]==0))

    avg_speed_baseline = np.mean(avg_speed, axis=0)
    num_veh_baseline = np.mean(num_veh, axis=0)

    # rl simulation results
    avg_speed = []
    num_veh = []
    rl_speed = []
    rl_num_veh = []
    num_enter_rl = []
    num_exit_rl = []
    for frac in [0.025, 0.05, 0.1]:
        for i in range(10):
            print(frac, i)
            rl = import_edge_pos_speed("merge-rl-emission-%.3f-%d.csv" % (frac, i))
            
            # conversion to acceptable forms for plotting
            pos_rl, speed_rl, avg_speed_rl, n_veh = get_speed_pos_merge(
                                                                        merge_data=rl,
                                                                        edgestarts=edgestarts_merge,
                                                                        num_steps=num_steps,
                                                                        dt=dt
                                                                        )
                                                                        avg_speed.append(avg_speed_rl)
                                                                        num_veh.append(n_veh)
                                                                        num_enter_rl.append(pos_rl.shape[1])
                                                                        num_exit_rl.append(sum(pos_rl[-1,:]==0))
        rl_speed.append(np.mean(avg_speed, axis=0))
        rl_num_veh.append(np.mean(num_veh, axis=0))

    int(len(avg_speed)/1)
    [np.mean(avg_speed_baseline[i*mult:(i+1)*mult]) for i in range(int(len(avg_speed_baseline)/mult))]

    mult = 250

    # plot the average speed
    plt.figure(figsize=(14,9))
    plt.title("Average Velocity of Vehicles on the Highway", fontsize=25)
    plt.xlabel("time (s)", fontsize=20)
    plt.ylabel("velocity (m/s)", fontsize=20)
    a = [np.mean(avg_speed_baseline[i*mult:(i+1)*mult]) for i in range(int(len(avg_speed_baseline)/mult))]
    a[0] = 0
    print(np.mean(a[5:]))
    plt.plot(np.arange(num_steps)[::mult]*dt, a, linewidth=2)
    for avg_speed in rl_speed:
        a = [np.mean(avg_speed[i*mult:(i+1)*mult]) for i in range(int(len(avg_speed)/mult))]
        a[0] = 0
        print(np.mean(a[5:]))
        plt.plot(np.arange(num_steps)[::mult]*dt, a, linewidth=2)
    plt.legend(["0% AV Penetration", "2.5% AV Penetration",
                "5% AV Penetration", "10% AV Penetration"], fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()

    # plot the number of vehicles
    plt.figure(figsize=(14,9))
    plt.title("Number of Vehicles on the Highway", fontsize=25)
    plt.xlabel("time (s)", fontsize=20)
    plt.ylabel("number of vehicles", fontsize=20)
    a = [np.mean(num_veh_baseline[i*mult:(i+1)*mult])
         for i in range(int(len(num_veh_baseline)/mult))]
    a[0] = 0
    plt.plot(np.arange(num_steps)[::mult]*dt, a, linewidth=2)
    for num_veh in rl_num_veh:
        a = [np.mean(num_veh[i*mult:(i+1)*mult]) for i in range(int(len(num_veh)/mult))]
        a[0] = 0
        plt.plot(np.arange(num_steps)[::mult]*dt, a, linewidth=2)
    plt.legend(["0% CAV Penetration", "2.5% CAV Penetration",
                "5% CAV Penetration", "10% CAV Penetration"], fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()

def heat_map_plots():
    # simulation results
    baseline = import_edge_pos_speed("merge-baseline-emission-2000-0.csv")

    pos_baseline, speed_baseline, _, _ = get_speed_pos_merge(
                                                             merge_data=baseline,
                                                             edgestarts=edgestarts_merge,
                                                             num_steps=num_steps,
                                                             dt=dt)

    pos = pos_baseline[::15]
    speed = speed_baseline[::15]
    x = pos.flatten()
    t = np.array([np.repeat(i * 15 * dt, pos.shape[1]) for i in range(pos.shape[0])]).flatten()
    values = speed.flatten()

    t = np.delete(t, np.where(x == 0)[0])
    values = np.delete(values, np.where(x == 0)[0])
    x = np.delete(x, np.where(x == 0)[0])

    points = np.stack((x,t)).T
    grid_x, grid_y = np.mgrid[0:708:5, 0:num_steps*dt:15*dt]

    a = griddata(points, values, (grid_x, grid_y))

    plt.figure(figsize=(16,9))
    norm = plt.Normalize(0, 15)
    # plt.title("Velocity Distribution for Vehicles in the Highway", fontsize=25)
    plt.xlabel("time (s)", fontsize=20)
    plt.ylabel("position (m)", fontsize=20)
    plt.imshow(a, extent=(0,3600,0,708), origin='lower', aspect='auto', cmap=my_cmap, norm=norm)
    cbar = plt.colorbar()
    cbar.set_label('Velocity (m/s)', fontsize=20)
    cbar.ax.tick_params(labelsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()

    sum(speed_rl[-1,:]==0)
    # simulation results
    rl = import_edge_pos_speed("merge-rl-emission-0.050-0.csv")

    pos_rl, speed_rl, _, _ = get_speed_pos_merge(
                                                 merge_data=rl,
                                                 edgestarts=edgestarts_merge,
                                                 num_steps=num_steps,
                                                 dt=dt)

    pos = pos_rl[::15]
    speed = speed_rl[::15]
    x = pos.flatten()
    t = np.array([np.repeat(i * 15 * dt, pos.shape[1]) for i in range(pos.shape[0])]).flatten()
    values = speed.flatten()

    t = np.delete(t, np.where(x == 0)[0])
    values = np.delete(values, np.where(x == 0)[0])
    x = np.delete(x, np.where(x == 0)[0])

    points = np.stack((x,t)).T
    grid_x, grid_y = np.mgrid[0:708:5, 0:num_steps*dt:15*dt]

    a = griddata(points, values, (grid_x, grid_y))

    plt.figure(figsize=(16,9))
    norm = plt.Normalize(0, 15)
    # plt.title("Velocity Distribution for Vehicles in the Highway", fontsize=25)
    plt.xlabel("time (s)", fontsize=20)
    plt.ylabel("position (m)", fontsize=20)
    plt.imshow(a, extent=(0,3600,0,708), origin='lower', aspect='auto', cmap=my_cmap, norm=norm)
    cbar = plt.colorbar()
    cbar.set_label('Velocity (m/s)', fontsize=20)
    cbar.ax.tick_params(labelsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()
"""
    -----------------------
    END OF PLOTTING SECTION
    -----------------------
"""
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
