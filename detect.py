import torch
import sys
import os
import os.path as osp
import argparse
import cv2
from threading import Thread
from queue import Queue
import json
import time

from model.gast_net import SpatioTemporalModel, SpatioTemporalModelOptimized1f
from lib.pose import load_2d_model
from lib.pose import gen_frame_kpts
from common.skeleton import Skeleton
from common.graph_utils import adj_mx_from_skeleton
from common.generators import *
from tools.utils import get_path
from tools.preprocess import h36m_coco_format, revise_kpts
from tools.inference import gen_pose
from socket_helper.server import SocketServerHelper

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

demo_file = 'data/demo/demo.txt'


def demo(out_q):
    while True:
        with open(demo_file) as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip()
                # Produce data
                out_q.put(data)
                time.sleep(0.2)


def get_joints_info(num_joints=17):
    # Body+toe keypoints
    if num_joints == 19:
        skeleton = Skeleton(parents=[-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 10, 13, 14, 10, 16, 17],
                            joints_left=[5, 6, 7, 8, 13, 14, 15],
                            joints_right=[1, 2, 3, 4, 16, 17, 18])
    # Body keypoints
    else:
        joints_left = [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]

        h36m_skeleton = Skeleton(parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
                                 joints_left=[4, 5, 6, 11, 12, 13],
                                 joints_right=[1, 2, 3, 14, 15, 16])

    keypoints_metadata = {'keypoints_symmetry': (joints_left, joints_right), 'layout_name': 'Human3.6M',
                          'num_joints': num_joints}

    return joints_left, joints_right, h36m_skeleton, keypoints_metadata


def load_model_realtime(rf=27):
    if rf == 27:
        chk = model_dir + '27_frame_model_causal.bin'
        filters_width = [3, 3, 3]
        channels = 128
    else:
        raise ValueError('Only support 27 receptive field models for inference!')

    print('Loading GAST-Net ...')
    model_pos = SpatioTemporalModelOptimized1f(adj, 17, 2, 17, filter_widths=filters_width, causal=True,
                                               channels=channels, dropout=0.25)

    # Loading pre-trained model
    checkpoint = torch.load(chk)
    model_pos.load_state_dict(checkpoint['model_pos'])

    if torch.cuda.is_available():
        model_pos = model_pos.cuda()
    model_pos.eval()

    print('GAST-Net successfully loaded')

    return model_pos


def load_model_layer(rf=27, output_toe=False):
    if rf == 27:
        if output_toe:
            chk = model_dir + '27_frame_model_toe.bin'
        else:
            chk = model_dir + '27_frame_model.bin'
        filters_width = [3, 3, 3]
        channels = 128
    elif rf == 81:
        chk = model_dir + '81_frame_model.bin'
        filters_width = [3, 3, 3, 3]
        channels = 64
    else:
        raise ValueError('Only support 27 and 81 receptive field models for inference!')

    print('Loading GAST-Net ...')
    model_pos = SpatioTemporalModel(adj, 17, 2, 17, filter_widths=filters_width, channels=channels, dropout=0.05)

    # Loading pre-trained model
    checkpoint = torch.load(chk)
    model_pos.load_state_dict(checkpoint['model_pos'])

    if torch.cuda.is_available():
        model_pos = model_pos.cuda()
    model_pos = model_pos.eval()

    print('GAST-Net successfully loaded')

    return model_pos


def generate_skeletons(queue, detect_args, human_model, pose_model, people_sort, model_pos, frame, width=1920,
                       height=1080, rf=27, output_animation=False, ab_dis=False):
    """
    :param queue: for data interaction in different threads
    :param detect_args: detect args
    :param human_model: human detect model
    :param pose_model: 2D pose model
    :param people_sort:
    :param model_pos: 3D model
    :param frame: The input video frame.
    :param width: frame width
    :param height: frame height
    :param rf: receptive fields
    :param output_animation: Generating animation video
    :param ab_dis: Whether the 3D pose generates the absolute distance of the plane (x, y)
    """

    # Generate 2D human pose
    key_points, scores = gen_frame_kpts(detect_args=detect_args, human_model=human_model, pose_model=pose_model,
                                        people_sort=people_sort, frame=frame, det_dim=416)
    # None代表
    # 1. 检测不到人
    # 2. 检测错误
    # 3. 超过一个人
    if key_points is None or scores is None:
        return

    key_points, scores, valid_frames = h36m_coco_format(key_points, scores)
    re_kpts = revise_kpts(key_points, scores, valid_frames)
    num_person = len(re_kpts)

    if num_person > 1:
        print("More than one person!")
        return

    # Generate 3D human pose
    # pre-process key_points

    pad = (rf - 1) // 2  # Padding on each side
    causal_shift = 0

    # Generating 3D poses
    prediction = gen_pose(re_kpts, valid_frames, width, height, model_pos, pad, causal_shift)

    # Adding absolute distance to 3D poses and rebase the height
    if ab_dis:
        prediction[0][:, :, 2] -= np.expand_dims(np.amin(prediction[0][:, :, 2], axis=1), axis=1).repeat([17], axis=1)
    else:
        prediction[0][:, :, 2] -= np.amin(prediction[0][:, :, 2])

    if output_animation:
        # re_kpts: (M, T, N, 2) --> (T, M, N, 2)
        re_kpts = re_kpts.transpose(1, 0, 2, 3)
        # np.save("key_points.npy", re_kpts)
        # np.save("pose.npy", prediction[0][0])
        prediction_list = prediction[0][0].tolist()
        prediction_json = json.dumps(prediction_list)
        # with open("test.txt", "a") as test:
        #     test.write(prediction_json + ' ' + str(time.time()) + "\n")
        #     test.close()
        queue.put(prediction_json)


def open_webcam(queue, output_animation=False, output_toe=False, rf=27):
    # 加载2D模型
    detect_args, human_model, pose_model, people_sort = load_2d_model()

    # 加载3D模型
    model_pos = load_model_layer(rf, output_toe)

    # 打开摄像头
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    width = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("Preview -- Width: " + str(width) + ", Height: " + str(height))

    # 获取第一帧
    # 判断是否正常打开
    if vc.isOpened():
        # isRead的值为True或False，代表有没有读到图片
        # frame是当前截取一帧的图片
        is_read = vc.read()
    else:
        return -1

    while is_read:
        is_read, frame = vc.read()
        generate_skeletons(queue=queue, detect_args=detect_args, human_model=human_model, pose_model=pose_model,
                           people_sort=people_sort, model_pos=model_pos, frame=frame, width=width, height=height,
                           rf=rf, output_animation=output_animation)
        cv2.imshow("preview", frame)
        # exit on ESC
        if cv2.waitKey(10) & 0xFF == 27:
            break
    cv2.destroyWindow("preview")
    print("Detection end!")


def arg_parse():
    """
    Parse arguments for the skeleton module
    """
    parser = argparse.ArgumentParser('Generating skeleton demo.')
    parser.add_argument('-rf', '--receptive-field', type=int, default=81, help='number of receptive fields')
    parser.add_argument('-a', '--animation', action='store_true', help='output animation')
    parser.add_argument('-t', '--toe', action='store_true', help='output toe')
    parser.add_argument('-d', '--demo', action='store_true', help='output demo')
    # 限制只能检测一个人
    # parser.add_argument('-np', '--num-person', type=int, default=1, help='number of estimated human poses. [1, 2]')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    data_queue = Queue(maxsize=1)
    args = arg_parse()
    if args.demo:
        demo_thread = Thread(target=demo, args=(data_queue,))
        demo_thread.start()
    else:
        webcam_thread = Thread(target=open_webcam, args=(data_queue, args.animation, args.toe))
        webcam_thread.start()
    socket_server = SocketServerHelper()
    # 新开一个线程，用于接收新连接
    socket_server_thread = Thread(target=socket_server.accept_client, args=(data_queue,))
    socket_server_thread.setDaemon(True)
    socket_server_thread.start()
