import os
from omegaconf import OmegaConf
from Pathes import P
from utils import load_hdf5, save_videos, visualize_joints

def main(cfg):
    dataset_dir = P.data_dir / cfg.task.name
    dataset_name = f'episode_{0}'
    qpos, qvel, action, image_dict = load_hdf5(dataset_dir, dataset_name)
    save_videos(image_dict, cfg.sim.DT, video_path=os.path.join(dataset_dir, dataset_name + '_video.mp4'))
    visualize_joints(qpos, action, plot_path=os.path.join(dataset_dir, dataset_name + '_qpos.png'))
    # visualize_timestamp(t_list, dataset_path) # TODO addn timestamp back

if __name__ == '__main__':
    cfg = OmegaConf.load(P.config_dir / "demonstration.yaml")
    main(cfg)
