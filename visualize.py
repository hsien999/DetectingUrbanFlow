import os
import os.path as osp
import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from utils import common

plt.rcParams['figure.constrained_layout.use'] = True


# adapted to display Chinese
# plt.rcParams['font.family'] = ['sans-serif']
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False


def save_subareas_fig_from_time():
    subareas_dir = 'output/subareas_split_time'
    fig, axes = plt.subplots(1, 2, sharey='all', figsize=(10, 5))
    for i in range(24):
        print(f'reading the {i} time partition of subarea ...')
        axes[0].cla()
        axes[1].cla()
        for _, data in pd.read_csv('output/wuchangroad_network.csv', index_col=None).iterrows():
            x1, y1, x2, y2 = data['XCoord_0'], data['YCoord_0'], data['XCoord_1'], data['YCoord_1']
            axes[0].plot((x1, x2), (y1, y2), color='C9')
            axes[1].plot((x1, x2), (y1, y2), color='C9')
        subareas_path_hole = [osp.join(subareas_dir, fname) for fname in os.listdir(subareas_dir) if
                              fname.startswith(f'{i}_hole')]
        subareas_path_volcano = [osp.join(subareas_dir, fname) for fname in os.listdir(subareas_dir) if
                                 fname.startswith(f'{i}_volcano')]
        all_hole, all_volcano = [], []
        hole_cnt, volcano_cnt = 0, 0
        for path in subareas_path_hole:
            result = np.load(path, allow_pickle=True)[0]
            hole_cnt += len(result.keys())
            for h in result.values():
                all_hole.extend(h[1])
        for path in subareas_path_volcano:
            result = np.load(path, allow_pickle=True)[0]
            volcano_cnt += len(result.keys())
            for v in result.values():
                all_volcano.extend(v[1])
        all_hole, all_volcano = np.asarray(all_hole), np.asarray(all_volcano)
        if len(all_hole) > 0:
            axes[0].scatter(all_hole[:, 0], all_hole[:, 1], s=10, c='g', alpha=0.5)
        if len(all_volcano) > 0:
            axes[1].scatter(all_volcano[:, 0], all_volcano[:, 1], s=10, c='r', alpha=0.5)
        axes[0].xaxis.set_major_locator(MultipleLocator(3000))
        axes[1].xaxis.set_major_locator(MultipleLocator(3000))
        common.set_axes_equal_2d(axes[0])
        common.set_axes_equal_2d(axes[1])
        plt.suptitle('TIME:{:02d}:00-{:02d}:00 -> hole={:,} volcano={:,}'.format(i, i + 1, hole_cnt, volcano_cnt),
                     fontsize=13)
        plt.savefig(f'output/images/{i}.png', dpi=200)  # bbox_inches='tight'
        print(f'saved to "output/images/{i}.png"')
    plt.close()
    return


def plot_network_all():
    net_dir = 'output/network_split_time'
    fig, axes = plt.subplots(1, 2, sharey='all', figsize=(8, 4))
    for i in range(24):
        axes[0].cla()
        axes[1].cla()
        for _, data in pd.read_csv('output/wuchangroad_network.csv', index_col=None).iterrows():
            x1, y1, x2, y2 = data['XCoord_0'], data['YCoord_0'], data['XCoord_1'], data['YCoord_1']
            axes[0].plot((x1, x2), (y1, y2), color='C9')
            axes[1].plot((x1, x2), (y1, y2), color='C9')
        net_path = osp.join(net_dir, f'network_{i}.npy')
        net_ran_path = osp.join(net_dir, f'network_random_{i}.npy')
        net = np.load(net_path, allow_pickle=True)[0]
        net_ran = np.load(net_ran_path, allow_pickle=True)[0]
        print(net.od_count, net_ran.od_count)
        matches, matches_ran = net.matches.values(), net_ran.matches.values()
        points, points_ran = [], []
        for mat in matches:
            points.extend(mat)
        for mat in matches_ran:
            points_ran.extend(mat)
        points, points_ran = np.asarray(points), np.asarray(points_ran)
        print(len(points), len(points_ran))
        axes[0].set_title('Generated Network')
        axes[0].set_title('Random Network')
        axes[0].scatter(points[:, 0], points[:, 1], s=3, c='g', alpha=0.5)
        axes[1].scatter(points_ran[:, 0], points_ran[:, 1], s=3, c='r', alpha=0.5)
        axes[0].xaxis.set_major_locator(MultipleLocator(3000))
        axes[1].xaxis.set_major_locator(MultipleLocator(3000))
        common.set_axes_equal_2d(axes[0])
        common.set_axes_equal_2d(axes[1])
        fig.suptitle('TIME:{:02d}:00-{:02d}:00'.format(i, i + 1))
        plt.pause(1)
    plt.close()
    return


def plot_matched_neighbours_example():
    net = np.load('output/network_split_time/network_0.npy', allow_pickle=True)[0]
    target_point = random.sample(list(net.matches.values()), 1)[0][0]
    cur = time.time()
    neighbours, o_cnt, d_cnt = net.network_constrained_neighbors(1000, target_point)
    print('cost: ', time.time() - cur)
    print('od count: ', o_cnt, d_cnt)
    print('all: ', len(neighbours))
    plt.scatter((target_point[0]), (target_point[1]), s=20, c='r', marker='D')
    neighbours = np.asarray(list(neighbours))
    plt.scatter((neighbours[:, 0]), (neighbours[:, 1]), s=1.5, c='b')
    plt.show()
    return


def plot_road_map():
    road_path = 'output/wuchangroad_network.csv'
    for _, data in pd.read_csv(road_path, index_col=None).iterrows():
        x1, y1, x2, y2 = data['XCoord_0'], data['YCoord_0'], data['XCoord_1'], data['YCoord_1']
        plt.plot((x1, x2), (y1, y2), color='C9')
    # od_cleaned_path = 'output/wuchangroad_od_cleaned.csv'
    # od_data = pd.read_csv(od_cleaned_path, index_col=None)
    # plt.scatter(od_data['XCoord'], od_data['YCoord'], c='g', s=1)
    return


if __name__ == '__main__':
    # save all figs for subareas in 'output/images/'
    save_subareas_fig_from_time()
    # plot figures showed the difference between generated and random ntetwork
    # plot_network_all()
    # plot fig for illustrative matched neighbourhood
    # plot_matched_neighbours_example()
    # plot road network
    # plot_road_map()
