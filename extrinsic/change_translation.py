import numpy as np
import os

if __name__ == '__main__':
    base_dataset_dir = "/Users/simjoonyeol/PycharmProjects/extrinsic/dataset"
    pose_dir = base_dataset_dir + "/pose"
    pose_edited_dir = base_dataset_dir + "/pose_edited"

    if not os.path.exists(pose_edited_dir):
        os.mkdir(pose_edited_dir)

    for pose_file in os.listdir(pose_dir):
        data = np.loadtxt(os.path.join(pose_dir, pose_file))
        tmp = data[1][3]
        data[1][3] = data[2][3]
        data[2][3] = tmp
        np.savetxt(os.path.join(pose_edited_dir, pose_file.split('.')[0] + '.txt'), data, fmt='%f')