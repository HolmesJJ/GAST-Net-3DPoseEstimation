import sys
import os.path as osp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from common.skeleton import Skeleton
from common.graph_utils import adj_mx_from_skeleton
from common.generators import *
from tools.utils import get_path
from tools.color_edge import h36m_color_edge

sys.path.insert(0, osp.dirname(osp.realpath(__file__)))
cur_dir, chk_root, data_root, lib_root, output_root = get_path(__file__)
model_dir = chk_root + 'gastnet/'
sys.path.insert(1, lib_root)
sys.path.pop(1)
sys.path.pop(0)

skeleton = Skeleton(parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
                    joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                    joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])
adj = adj_mx_from_skeleton(skeleton)

joints_left, joints_right = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]
kps_left, kps_right = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]
rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)
key_points_metadata = {'keypoints_symmetry': (joints_left, joints_right), 'layout_name': 'Human3.6M', 'num_joints': 17}


def render_animation(azim, fps=60, size=5):
    ax_3d = []
    lines_3d = []
    radius = 1.7

    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=15., azim=azim)
    ax.set_xlim3d([-radius, radius])
    ax.set_zlim3d([0, radius])
    ax.set_ylim3d([-radius, radius])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.dist = 7.5
    ax_3d.append(ax)
    lines_3d.append([])

    parents = skeleton.parents()
    index = [i for i in np.arange(17)]

    initialized = False

    def animate(i):
        nonlocal initialized

        joints_right_2d = key_points_metadata['keypoints_symmetry'][1]

        colors_2d = np.full(17, 'black')
        colors_2d[joints_right_2d] = 'red'

        try:
            key_points = np.load('key_points.npy', allow_pickle=True)
            if key_points is None:
                return
            key_points = np.array(key_points)
            pose = np.load('pose.npy', allow_pickle=True)
            if pose is None:
                return
        except Exception as e:
            print(e)
            return None

        if not initialized:
            for j, j_parent in zip(index, parents):
                if j_parent == -1:
                    continue

                # Apply different colors for each joint
                col = h36m_color_edge(j)
                lines_3d[0].append(ax.plot([pose[j, 0], pose[j_parent, 0]],
                                           [pose[j, 1], pose[j_parent, 1]],
                                           [pose[j, 2], pose[j_parent, 2]], zdir='z', c=col, linewidth=3))
            initialized = True
        else:
            for j, j_parent in zip(index, parents):
                if j_parent == -1:
                    continue

                lines_3d[0][j - 1][0].set_xdata([pose[j, 0], pose[j_parent, 0]])
                lines_3d[0][j - 1][0].set_ydata([pose[j, 1], pose[j_parent, 1]])
                lines_3d[0][j - 1][0].set_3d_properties([pose[j, 2], pose[j_parent, 2]], zdir='z')

    anim = FuncAnimation(fig, animate, interval=1000 / fps)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    render_animation(azim=np.array(70., dtype=np.float32))
