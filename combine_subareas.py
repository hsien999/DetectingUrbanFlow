import os
import os.path as osp
import numpy as np
import pandas as pd
from tqdm import tqdm
import heapq
from graph.linear import bernoulli_lambda
from graph.base import DisjointSetTree
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.patches import Patch
from utils import common

min_lam = float('inf')
grade_label = ('Excellent', 'Good', 'Middle', 'Pass', 'Fail')


def count_od_number(pi_type, neighbour):
    """
    calculate the number of od points for given neighborhood
    """
    o_cnt, d_cnt = 0, 0
    for ne in neighbour:
        for tp in pi_type[ne]:
            if tp:
                o_cnt += 1
            else:
                d_cnt += 1
    return o_cnt, d_cnt


def calc_test_statistics(pi_type, neighbour, o_all, d_all):
    """
    calculate the test statistics
    """
    o_cnt, d_cnt = count_od_number(pi_type, neighbour)
    if o_cnt + d_cnt == 0 or o_all + d_all == 0 or o_all + d_all - o_cnt - d_cnt == 0:
        return min_lam
    return bernoulli_lambda(o_cnt, d_cnt, o_all, d_all)


def read_od_type_data(time_idx):
    """
    load a mapping (point => od type) from `output/wuchangroad_od_cleaned.csv` for given time index
    """
    save_path = f'output/od_type_split_time/{time_idx}.npy'
    if osp.exists(save_path):
        return np.load(save_path, allow_pickle=True)[0]
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    od_data = pd.read_csv('output/wuchangroad_od_cleaned.csv', index_col=None)
    pi_type = {}
    for idx in tqdm(od_data.index, desc="reading OD points data"):
        data = od_data.loc[idx]
        x_cor, y_cor, loc_time = data['XCoord'], data['YCoord'], data['LOC_TIME']
        hour = int(loc_time.split(' ')[1].split(':')[0])
        if hour != time_idx:
            continue
        pi = (x_cor, y_cor)
        if pi not in pi_type:
            pi_type[pi] = []
        pi_type[pi].append(False if idx & 1 != 0 else True)
    np.save(save_path, [pi_type], allow_pickle=True)
    return pi_type


def get_od_count(time_idx):
    """
    read od number from network
    """
    net = np.load(f'output/network_split_time/network_{time_idx}.npy', allow_pickle=True)[0]
    return net.o_count, net.d_count


def load_subareas(time_idx):
    """
    load subareas data in `output/subareas_split_time`
    """
    subareas_dir = 'output/subareas_split_time'
    subareas_path_hole = [osp.join(subareas_dir, fname) for fname in os.listdir(subareas_dir) if
                          fname.startswith(f'{time_idx}_hole')]
    subareas_path_volcano = [osp.join(subareas_dir, fname) for fname in os.listdir(subareas_dir) if
                             fname.startswith(f'{time_idx}_volcano')]
    hole_sub, volcano_sub = {}, {}
    for path in subareas_path_hole:
        hole_sub.update(np.load(path, allow_pickle=True)[0])
    for path in subareas_path_volcano:
        volcano_sub.update(np.load(path, allow_pickle=True)[0])
    return hole_sub, volcano_sub


def find_overlaps(subarea):
    """
    Calculate the set of points whose neighbourhood intersects
    """
    overlap_map = {}
    for p1, neighbour1 in tqdm(subarea.items(), desc="finding overlapping subareas"):
        overlap_map[p1] = []
        nb1 = neighbour1[1]
        for p2, neighbour2 in subarea.items():
            if p1 == p2:
                continue
            lam, nb2 = neighbour2
            if len(nb1 & nb2) > 0:
                heapq.heappush(overlap_map[p1], (-lam, p2))
    return overlap_map


def identify_hole_volcano(overlap_map, pi_type, subarea, o_cnt, d_cnt):
    """
    Combining subareas and select the subarea with highest lambda value in overlapping subareas
    """
    result = {}
    joint_set = DisjointSetTree()
    # Combining the first subarea in A-overlap with the candidate urban black hole and
    # calculating log lambda-new for the newly built urban black hole new
    for pi, overs in tqdm(overlap_map.items(), desc="identify candidate urban black hole or volcano"):
        lam_old, nb_old = subarea[pi]
        while overs:
            ov = heapq.heappop(overs)
            nb_j = subarea[ov[1]][1]
            nb_new = nb_old | nb_j
            lam_new = calc_test_statistics(pi_type, nb_new, o_cnt, d_cnt)
            # print(lam_new, lam_old)
            if lam_new >= lam_old:
                lam_old, nb_old = lam_new, nb_new
            else:
                break
        o_cnt, d_cnt = count_od_number(pi_type, nb_old)
        result[pi] = (lam_old, nb_old, o_cnt, d_cnt)
        joint_set.make_set(pi)

    # Storage of minimum connected sets by joint set
    for pi_1 in result:
        nb_1 = result[pi_1][1]
        for pi_2 in result:
            nb_2 = result[pi_2][1]
            if len(nb_1 & nb_2) > 0:
                joint_set.union(pi_1, pi_2)

    # The urban black hole with the highest log-likelihood ratio test statistic value
    # is selected as an urban black hole, and all the urban black holes overlapped with
    # this urban black hole are deleted.
    determine = {}
    for pi_group in joint_set.group().values():
        sorted(pi_group, key=lambda _p: result[_p][0], reverse=True)
        max_pi = pi_group[0]
        determine[max_pi] = result[max_pi]
    # sorted(determine, key=lambda _x: _x[0], reverse=True)

    # clean and sort result
    cleaned = []
    for data in determine.values():
        lam, nb, o_cnt, d_cnt = data
        cleaned.append((o_cnt / (o_cnt + d_cnt), nb))
    sorted(cleaned, key=lambda _x: _x[0])

    return cleaned


def multi_scale_hole_volcano(time_id):
    """
    Multi directional optimization method for detecting arbitrarily shaped urban black holes and volcanoes
    """
    pi2od = read_od_type_data(time_id)
    ori_cnt, des_cnt = get_od_count(time_id)
    hole_sub, volcano_sub = load_subareas(time_id)
    over_hole, over_volcano = find_overlaps(hole_sub), find_overlaps(volcano_sub)
    hole = identify_hole_volcano(over_hole, pi2od, hole_sub, ori_cnt, des_cnt)
    volcano = identify_hole_volcano(over_volcano, pi2od, volcano_sub, ori_cnt, des_cnt)
    return hole, volcano


def save_identified_result(time_id, hole, volcano):
    """
    save results of the combined black hole and volcano in the directory `output/hole_volcano/`
    """
    np.save(f'output/hole_volcano/{time_id}_hole.npy', [hole], allow_pickle=True)
    print(f'result saved to "output/hole_volcano/{time_id}_hole.npy"')
    np.save(f'output/hole_volcano/{time_id}_volcano.npy', [volcano], allow_pickle=True)
    print(f'result saved to "output/hole_volcano/{time_id}_volcano.npy"')


def load_identified_result(time_id):
    """
    Load the cached results of the combined black hole and volcano in the directory `output/hole_volcano/`
    """
    hole = np.load(f'output/hole_volcano/{time_id}_hole.npy', allow_pickle=True)[0]
    volcano = np.load(f'output/hole_volcano/{time_id}_volcano.npy', allow_pickle=True)[0]
    return hole, volcano


def plot_determine_hole_volcano(time_id, hole, volcano):
    """
    Plotting images of black holes and volcanoes for each time period
    """
    plt.cla()
    for _, data in pd.read_csv('output/wuchangroad_network.csv', index_col=None).iterrows():
        x1, y1, x2, y2 = data['XCoord_0'], data['YCoord_0'], data['XCoord_1'], data['YCoord_1']
        plt.plot((x1, x2), (y1, y2), color='C9')
    all_hole, all_volcano = [], []
    for h in hole:
        # if h[0] >= 0.3:
        #     continue
        all_hole.extend(h[1])
    for v in volcano:
        # if v[0] <= 0.7:
        #     continue
        all_volcano.extend(v[1])
    all_hole, all_volcano = np.array(all_hole), np.array(all_volcano)
    if len(all_volcano) > 0:
        plt.scatter(all_volcano[:, 0], all_volcano[:, 1], s=30, c='r', alpha=0.5)
    if len(all_hole) > 0:
        plt.scatter(all_hole[:, 0], all_hole[:, 1], s=30, c='g', alpha=0.5)
    plt.gca().xaxis.set_major_locator(MultipleLocator(3000))
    common.set_axes_equal_2d(plt.gca())
    plt.suptitle('Urban black holes and volcanoes in Wuchang')
    plt.title(
        'TIME:2014-05-07 {:02d}:00-{:02d}:00 Wed. '
        '(hole={:,}  volcano={:,})'.format(time_id, time_id + 1, len(hole), len(volcano)),
        fontsize=10, pad=15, loc='center')
    plt.legend(loc='best', handles=[Patch(color='C9', label='Road'), Patch(color='r', label='Black hole'),
                                    Patch(color='g', label='Volcano')])
    plt.savefig(f'output/results/{time_id}.png', dpi=200)  # bbox_inches='tight'
    print(f'result saved to "output/results/{time_id}.png"')


def analyse_result(hole, volcano, hole_score, volcano_score):
    """
    Quality assessment based on five-grade marking using generated black hole and volcano data
    """
    grade = np.arange(0, 0.5, 0.1)
    for ho in hole:
        sc = ho[0]
        for lv in range(len(grade)):
            gr = grade[lv]
            if gr <= sc <= gr + 0.1:
                hole_score[lv] += 1
                break
    grade = np.arange(1.0, 0.5, -0.1)
    for vo in volcano:
        sc = vo[0]
        for lv in range(len(grade)):
            gr = grade[lv]
            if gr - 0.1 <= sc <= gr:
                volcano_score[lv] += 1
                break
    return


if __name__ == '__main__':
    """
    Combination of subareas of urban black holes and volcanoes based on multi directional optimization
    based on a 1 hour division (total 24hours)
    """
    plt.figure(figsize=(5, 5))
    _hole_score, _volcano_score = {}, {}
    for _i in range(5):
        _hole_score[_i] = _volcano_score[_i] = 0
    for _id in range(24):
        # _result = multi_scale_hole_volcano(_id)
        # NOTE: The local cache is read by default, if you need to re-run the program,
        # comment out next line(228) and switch to the previous line(225)
        _result = load_identified_result(_id)
        plot_determine_hole_volcano(_id, *_result)
        save_identified_result(_id, *_result)
        analyse_result(*_result, _hole_score, _volcano_score)
    print('====== five-grade marking ======')
    print('hole:')
    for _lv, _hs in _hole_score.items():
        print(f'\t{grade_label[_lv]} = {_hs}')
    print('volcano:')
    for _lv, _vs in _volcano_score.items():
        print(f'\t{grade_label[_lv]} = {_vs}')
    print('===============================')
