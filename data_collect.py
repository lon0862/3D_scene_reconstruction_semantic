import numpy as np
from PIL import Image
import numpy as np
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import math
import os
import shutil
import csv
import json

GT = []
# This is the scene we are going to load.
# support a variety of mesh formats, such as .glb, .gltf, .obj, .ply
### put your scene path ###
test_scene = "env_apartment0/apartment_0/habitat/mesh_semantic.ply"
sim_settings = {
    "scene": test_scene,  # Scene path
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
    "width": 512,  # Spatial resolution of the observations
    "height": 512,
    # sensor pitch (x rotation in rads), lin: front view is 0
    "sensor_pitch": 0,
}

# This function generates a config for the simulator.
# It contains two parts:
# one for the simulator backend
# one for the agent, where you can attach a bunch of sensors


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


def transform_depth(image):
    depth_img = (image / 10 * 255).astype(np.uint8)
    return depth_img


def transform_semantic(semantic_obs):
    semantic_img = Image.new(
        "L", (semantic_obs.shape[1], semantic_obs.shape[0])) # origin is "P"
    semantic_img.putdata(semantic_obs.flatten())
    # semantic_img.putpalette(d3_40_colors_rgb.flatten())
    # semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    # semantic_img = semantic_img.convert("L") # RGB
    # semantic_img = cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_RGB2BGR)
    
    return semantic_img


def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    # depth snesor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    # semantic snesor
    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    agent_cfg.sensor_specifications = [
        rgb_sensor_spec, depth_sensor_spec, semantic_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


cfg = make_simple_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)

# initialize an agent
agent = sim.initialize_agent(sim_settings["default_agent"])

# Set agent state
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([0.0, 1.0, 0.0])  # agent in world space, (0,0,0) is floor1, (0,1,0) is floor2
agent.set_state(agent_state)

# obtain the default, discrete actions that an agent can perform
# default action space contains 3 actions: move_forward, turn_left, and turn_right
action_names = list(
    cfg.agents[sim_settings["default_agent"]].action_space.keys())
print("Discrete action space: ", action_names)


FORWARD_KEY = "w"
LEFT_KEY = "a"
RIGHT_KEY = "d"
FINISH = "f"
OMIT = "o"  # by lin
print("#############################")
print("use keyboard to control the agent")
print(" w for go forward  ")
print(" a for turn left  ")
print(" d for trun right  ")
print(" f for finish and quit the program")
print("#############################")

def load_scene_semantic_dict(dataset_path):
    with open(os.path.join(dataset_path, 'habitat', 'info_semantic.json'), 'r') as f:
        return json.load(f)

def fix_semantic_observation(semantic_observation, scene_dict):
    # The labels of images collected by Habitat are instance ids
    # transfer instance to semantic
    instance_id_to_semantic_label_id = np.array(scene_dict["id_to_label"])
    semantic_img = instance_id_to_semantic_label_id[semantic_observation]               
    return semantic_img

def navigateAndSee(action=""):
    if action in action_names:
        observations = sim.step(action)
        #print("action: ", action)

        rgb_img = transform_rgb_bgr(observations["color_sensor"])
        depth_img = transform_depth(observations["depth_sensor"])
        # semantic_img = transform_semantic(observations["semantic_sensor"])
        semantic_path = "env_apartment0/apartment_0"
        semantic_dict = load_scene_semantic_dict(semantic_path)
        semantic_obs = fix_semantic_observation(observations["semantic_sensor"], semantic_dict)
        semantic_img = transform_semantic(semantic_obs)
        cv2.imshow("RGB", rgb_img)
        cv2.imshow("depth", depth_img)
        #cv2.imshow("semantic", semantic_img)
        agent_state = agent.get_state()
        sensor_state = agent_state.sensor_states['color_sensor']
        #print("camera pose: x y z rw rx ry rz")
        #print(sensor_state.position[0], sensor_state.position[1], sensor_state.position[2],
        #      sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z)
        
        
        GT.append([sensor_state.position[0], sensor_state.position[1], sensor_state.position[2]])
            
        num = 0
        path_rgb = "hw1_data/"+floor+"/images/rgb_"+str(num)+".png"
        path_depth = "hw1_data/"+floor+"/depth/depth_"+str(num)+".png"
        path_semantic = "hw1_data/"+floor+"/annotations/semantic_"+str(num)+".png"
        while(1):
          if os.path.isfile(path_rgb):
            num+=1
            path_rgb = "hw1_data/"+floor+"/images/rgb_"+str(num)+".png"
            path_depth = "hw1_data/"+floor+"/depth/depth_"+str(num)+".png"
            path_semantic = "hw1_data/"+floor+"/annotations/semantic_"+str(num)+".png"
          else:
            break
        print(path_rgb)
        cv2.imwrite(path_rgb, rgb_img)
        cv2.imwrite(path_depth, depth_img)
        # cv2.imwrite(path_semantic, semantic_img)
        semantic_img.save(path_semantic)

floor = "floor2"
action = "move_forward"
navigateAndSee(action)

while True:
    keystroke = cv2.waitKey(0)
    if keystroke == ord(FORWARD_KEY):
        action = "move_forward"
        navigateAndSee(action)
        print("action: FORWARD")
    elif keystroke == ord(LEFT_KEY):
        action = "turn_left"
        navigateAndSee(action)
        print("action: LEFT")
    elif keystroke == ord(RIGHT_KEY):
        action = "turn_right"
        navigateAndSee(action)
        print("action: RIGHT")
    elif keystroke == ord(FINISH):
        with open('hw1_data/'+floor+'/GT/GT.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            print(len(GT))
            for i in range(len(GT)):
                writer.writerow(GT[i])
        print("action: FINISH")
        break
    elif keystroke == ord(OMIT):
        shutil.rmtree("hw1_data/"+floor+"/images")
        shutil.rmtree("hw1_data/"+floor+"/depth")
        shutil.rmtree("hw1_data/"+floor+"/annotations")
        shutil.rmtree("hw1_data/"+floor+"/GT")
        os.mkdir("hw1_data/"+floor+"/images")
        os.mkdir("hw1_data/"+floor+"/depth")
        os.mkdir("hw1_data/"+floor+"/annotations")
        os.mkdir("hw1_data/"+floor+"/GT")
        GT = []
        print("omit all files from images, depth, annotations, GT path")
    else:
        print("INVALID KEY")
        continue
