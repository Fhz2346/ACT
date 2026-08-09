from dm_control.suite import base
from dm_control import mujoco
from .BimanualViperX.base import BimanualViperXBaseTask


class BaseTask(base.Task):

    def __init__(self, BVXtask: BimanualViperXBaseTask):
        super().__init__(random=False)
        self.BVXtask = BVXtask

    def randomize_initial_env(self):
        raise NotImplementedError
    
    def initialize_episode(self, physics: mujoco.Physics):
        raise NotImplementedError
        
    def before_step(self, action, physics: mujoco.Physics):
        self.BVXtask.before_step(action, physics)

    def get_env_state(self, physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_observation(self, physics):
        obs = self.BVXtask.get_observation(physics)
        obs.env_state = self.get_env_state(physics)
        return obs

    def get_reward(self, physics):
        raise NotImplementedError
