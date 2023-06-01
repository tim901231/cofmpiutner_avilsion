from threshold import exec_threshold
from exec_3d import exec_3d
from ITRI_DLC.ICP import exec_ICP
from tqdm import tqdm
import numpy as np


seqs = ['ITRI_dataset/seq1', 'ITRI_dataset/seq2', 'ITRI_dataset/seq3']
# seqs = ['ITRI_DLC/test1', 'ITRI_DLC/test2']

for seq in seqs:
    all_timestamp_file = f"{seq}/all_timestamp.txt"
    local_timestamp_file = f"{seq}/localization_timestamp.txt"

    all_timestamp = []
    local_timestamp = []

    with open(all_timestamp_file, 'r') as f:
        for line in f.readlines():
           all_timestamp.append(line.strip('\n'))

    with open(local_timestamp_file, 'r') as f:
        for line in f.readlines():
           local_timestamp.append(line.strip('\n'))


    for timestamp in tqdm(all_timestamp):
        fname = f"{seq}/dataset/{timestamp}"
        try:
            exec_threshold(fname)
            exec_3d(fname)
        except:
            pass
            



    last_idx = 0
    cur_idx = 0
    x = []
    y = []

    for local in tqdm(local_timestamp):
        cur_idx = all_timestamp.index(local)
        
        all_data = []
        for idx in range(last_idx, cur_idx+1):
            fname = f"{seq}/dataset/{all_timestamp[idx]}/predict.npy"
            try:
                arr = np.load(fname)
                # print(arr.shape)
                if (len(arr.shape)>1):
                    all_data.append(arr)
            except:
                pass
        last_idx = cur_idx+1
        try:
            all_data = np.vstack(all_data)
            # print(all_data.shape)
        except:
            all_data = np.array([[0, 0, -1.63], [0, 1, -1.63]])
            # print(all_data.shape)

        np.savetxt(f"{seq}/dataset/{all_timestamp[idx]}/predict.csv", all_data, delimiter=',')
        
        
        pred_x, pred_y = exec_ICP(f"{seq}/dataset/{all_timestamp[idx]}")
        x.append(pred_x)
        y.append(pred_y)

    with open(f"{seq}/pred_pose.txt", 'w') as f:
        for i in range(len(x)):
            f.write(str(x[i]) + " " + str(y[i]) + "\n")

        
    



    
