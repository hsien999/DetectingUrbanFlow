import os
import os.path as osp
import numpy as np
import pandas as pd
import csv
import shapefile
from tqdm import tqdm
from graph.roadnet import RoadNetWork, generate_random_network
from graph.linear import euclidean_distance


def save_net_info_form_shapefile(file_path, save_path):
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    name = ['roadID', 'XCoord_0', 'YCoord_0', 'XCoord_1', 'YCoord_1']
    with open(save_path, 'w+', newline='') as fd:
        writer = csv.writer(fd)
        writer.writerow(name)
        sf = shapefile.Reader(file_path)
        # we assume that sf.shapeType = 3 (POLYLINE)
        assert sf.shapeType == 3
        print(f'reading shapefile [{file_path}] ...')
        print("\tshapefile shapeType = ", sf.shapeType)
        print("\tshapefile fields = ", sf.fields)
        # for i, sr in enumerate(sf.shapeRecords()):
        #     print(f"shape {i} :", sr.shape.shapeTypeName, sr.shape.bbox)
        #     for k, v in sr.record.as_dict().items():
        #         print('\t-> ', k, ' = ', v)
        point_cnt = 0
        road_cnt = 0
        for sr in sf.shapeRecords():
            records = sr.record.as_dict()
            points = [["{:.0f}".format(p) for p in pp] for pp in sr.shape.points]  # or bbox
            temp = [records[sf.fields[1][0]], points[0][0], points[0][1], points[1][0], points[1][1]]
            writer.writerow(temp)
            point_cnt += len(sr.shape.points)
            road_cnt += 1
        print(f'[{osp.splitext(osp.split(save_path)[1])[0]}] has {point_cnt} nodes and {road_cnt} roads')
    return


def calc_average_velocity():
    df = pd.read_csv(u'data/武昌订单点.csv', index_col=None)
    velocity = df[df['V'] != 0]['V']
    print('average velocity = ', velocity.sum() / velocity.size)
    return


def calc_average_road_length():
    # average velocity = 29.255637000899522 km/h => epsilon >= 500m
    df = pd.read_csv('output/wuchangroad_network.csv', index_col=None)
    sum_len = 0
    for _, data in df.iterrows():
        sum_len += euclidean_distance((data['XCoord_0'], data['YCoord_0']), (data['XCoord_1'], data['YCoord_1']))
    print('average length of roads', sum_len / len(df))
    return


def clean_od_data(origin_path, save_path):
    if not isinstance(origin_path, (list, tuple)):
        origin_path = [origin_path]
    od = pd.DataFrame()
    for path in origin_path:
        od = od.append(pd.read_csv(path, index_col=None))
    od = od.drop_duplicates()
    excluded_id = []
    for gid, data in od.groupby(['ID']):
        lines = len(data)
        date_multi = len(set(data['LOC_TIME']))
        if lines != 2 or date_multi != 2:
            excluded_id.append(gid)
    od = od[~np.in1d(od['ID'], excluded_id)].sort_values(by=['ID', 'LOC_TIME'], ascending=True)
    od.to_csv(save_path, index=False)
    # verify
    print('lines of od data: ', len(od))
    print('number of unique points: ', len(od[['XCoord', 'YCoord']].drop_duplicates()))
    point2road = od.groupby(['XCoord', 'YCoord'])['ROADID'].agg(lambda _x: len(set(_x)))
    print('duplicate mapping (pi -> road): ', (point2road - 1).sum())
    point2road = pd.DataFrame(point2road[point2road > 1]).rename(columns={"ROADID": "DuplicateRoad"})
    point2road.to_csv('output/wuchangroad_od_duplicate.csv')
    return


def get_road_net_from_time(road_path, od_path):
    road_net = [RoadNetWork() for _ in range(24)]
    road_data = pd.read_csv(road_path, index_col=None)
    od_data = pd.read_csv(od_path, index_col=None)
    for _, data in road_data.iterrows():
        road_id, x1, y1, x2, y2 = data['roadID'], data['XCoord_0'], data['YCoord_0'], data['XCoord_1'], data['YCoord_1']
        for rn in road_net:
            rn.add_edge(road_id, x1, y1, x2, y2)
    for idx, data in od_data.iterrows():
        road_id, x_cor, y_cor, loc_time = data['ROADID'], data['XCoord'], data['YCoord'], data['LOC_TIME']
        hour = int(loc_time.split(' ')[1].split(':')[0])
        road_net[hour].add_matches(road_id, x_cor, y_cor, False if idx & 1 != 0 else True)
    for rn in road_net:
        rn.build_kd_trees()
    return road_net


def save_network_from_time(save_dir, road_path, od_cleaned_path):
    road_net_24 = get_road_net_from_time(road_path, od_cleaned_path)
    for i, net in tqdm(enumerate(road_net_24), colour='green',
                       desc='save observed and random network(object:RoadNetwork) at different time periods'):
        np.save(osp.join(save_dir, f'network_{i}.npy'), [net], allow_pickle=True)
        ran_net_i = generate_random_network(net)
        np.save(osp.join(save_dir, f'network_random_{i}.npy'), [ran_net_i], allow_pickle=True)
    return


if __name__ == '__main__':
    # load and save network data from shape file
    save_net_info_form_shapefile('data/wuchangroad_1', 'output/wuchangroad_network.csv')
    # clean and verify od data
    clean_od_data('data/WUCHANG0.csv', 'output/wuchangroad_od_cleaned.csv')
    # save all network data (in form of npy file) defined in graph.roadnet.RoadNetwork
    save_network_from_time('output/network_split_time', 'output/wuchangroad_network.csv',
                           'output/wuchangroad_od_cleaned.csv')
    # calculate
    calc_average_road_length()
    # calc_average_velocity()
    pass
