import sys
import os.path as osp

sys.path.insert(1, osp.join(osp.dirname(osp.realpath(__file__)), 'hrnet/pose_estimation'))
from gen_kpts import gen_img_kpts, gen_video_kpts, load_default_model, load_2d_model, gen_frame_kpts
sys.path.insert(2, osp.join(osp.dirname(osp.realpath(__file__)), 'hrnet/lib/utils'))
from utilitys import plot_keypoint, write, PreProcess, box_to_center_scale, load_json

sys.path.pop(1)
sys.path.pop(2)