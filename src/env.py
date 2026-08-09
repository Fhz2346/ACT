import os
from dm_control import mujoco
from dm_control.rl import control

from Pathes import P
from Task import task_name2task
from Task.BimanualViperX import BimanualViperXEETask, BimanualViperXTask

def make_ee_sim_env(cfg):
    BVXtask = BimanualViperXEETask()
    task = task_name2task[cfg.task.name](BVXtask, cfg.task)
    xml_path = os.path.join(P.xml_dir, f"{cfg.robot.name}_ee_{cfg.task.name}.xml")
    physics = mujoco.Physics.from_xml_path(xml_path)
    env = control.Environment(
        physics,
        task,
        time_limit=20,
        control_timestep=cfg.sim.DT,
        n_sub_steps=None,
        flat_observation=False,
    )
    return env

def make_sim_env(cfg):
    BVXtask = BimanualViperXTask()
    task = task_name2task[cfg.task.name](BVXtask, cfg.task)
    xml_path = os.path.join(P.xml_dir, f"{cfg.robot.name}_{cfg.task.name}.xml")
    physics = mujoco.Physics.from_xml_path(xml_path)
    env = control.Environment(
        physics,
        task,
        time_limit=20,
        control_timestep=cfg.sim.DT,
        n_sub_steps=None,
        flat_observation=False,
    )
    return env
