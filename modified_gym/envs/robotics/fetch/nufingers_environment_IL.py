import os
from modified_gym import utils
from modified_gym.envs.robotics import NuFingers_env_IL


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('scape', 'NuFingers_scape.xml')


class NuFingersEnvIL(NuFingers_env_IL.NuFingersEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', pert_type='none', n_actions=2):
        initial_qpos = {
                'Joint_1_L' : 0.537,
                'Joint_2_L' : -1.72,
                'Joint_1_R' : -0.537,
                'Joint_2_R' : -1.72,
        }
        NuFingers_env_IL.NuFingersEnv.__init__(
            self, MODEL_XML_PATH, n_substeps=20, target_range=0.7853981633974483, distance_threshold=0.09817477042468103,
            initial_qpos=initial_qpos, reward_type=reward_type, pert_type=pert_type, n_actions=n_actions)
        utils.EzPickle.__init__(self)
