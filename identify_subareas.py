import multiprocessing as mp
from utils.common import split_worker
import numpy as np
from tqdm import tqdm
from graph.roadnet import identify_subareas


def combine_subareas(_st, _ed):
    """
    Identify the subarea of the time index from _st,to _ed
    time index:
        index '0' indicates `00:00:00-01:00:00`
        and the like
    """
    for i in range(_st, _ed):
        net = np.load(f'output/network_split_time/network_{i}.npy', allow_pickle=True)[0]
        ran_net = np.load(f'output/network_split_time/network_random_{i}.npy', allow_pickle=True)[0]
        result_hole, result_volcano = {}, {}
        _all, _cnt, add_end = 0, 0, False
        all_matches = []
        for val in net.matches.values():
            all_matches.extend(val)
        for pi in tqdm(all_matches, miniters=100, mininterval=30, maxinterval=300):
            res = identify_subareas(net, ran_net, pi, r_time=r_time, alpha=alpha, epsilon=epsilon)
            if res is None:
                continue
            flag, lambda_obs, neighbour = res
            if flag:
                result_hole[pi] = (lambda_obs, neighbour)
            else:
                result_volcano[pi] = (lambda_obs, neighbour)
            _all += 1
            add_end = True
            if _all > 5000:
                np.save(f'output/subareas_split_time/{i}_hole_{_cnt}.npy', [result_hole], allow_pickle=True)
                np.save(f'output/subareas_split_time/{i}_volcano_{_cnt}.npy', [result_volcano], allow_pickle=True)
                _cnt += 1
                result_hole, result_volcano = {}, {}
                _all, add_end = 0, False
        if add_end:
            np.save(f'output/subareas_split_time/{i}_hole_{_cnt}.npy', [result_hole], allow_pickle=True)
            np.save(f'output/subareas_split_time/{i}_volcano_{_cnt}.npy', [result_volcano], allow_pickle=True)


if __name__ == '__main__':
    """
    Identification of subareas of urban black holes and volcanoes
    based on a 1 hour division (total 24hours)
    """
    worker = 8  # Number of processes (recommended not to be larger than the number of cpu cores)
    r_time = 99  # the number of repetitions of Monte Carlo simulation
    alpha = 0.05  # significance level
    epsilon = 1200  # the neighbourhood cutoff radius
    split = split_worker(24, worker)
    with mp.Pool() as pool:
        pool.starmap(combine_subareas, split)
