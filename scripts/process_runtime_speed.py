import time

import numpy as np
import open3d as o3d
import glob
import matplotlib.pyplot as plt


if __name__ == '__main__':
    DSO_tracking_info = open("/home/jzhang72/NetBeansProjects/ROS_Img2PointCloud/ros_ws/mapping_article_data/DSO_tracking_time_ms.txt", "r")
    DSO_keyframe_info = open("/home/jzhang72/NetBeansProjects/ROS_Img2PointCloud/ros_ws/mapping_article_data/DSO_keyframe_time_ms.txt", "r")
    SDSO_tracking_info = open("/home/jzhang72/NetBeansProjects/ROS_Img2PointCloud/ros_ws/mapping_article_data/SDSO_tracking_time_ms.txt", "r")
    SDSO_keyframe_info = open("/home/jzhang72/NetBeansProjects/ROS_Img2PointCloud/ros_ws/mapping_article_data/SDSO_keyframe_time_ms.txt", "r")
    DSOL_info = open("/home/jzhang72/NetBeansProjects/ROS_Img2PointCloud/ros_ws/mapping_article_data/DSOL_times_ms.txt", "r")

    track_time_thresh = 100
    keyframe_time_thresh = 500
    # process DSO log
    dso_track_time = {}
    dso_keyframe_time = {}      # use dict instead of list to avoid same keyframe IDs to be recorded (especially for dsol data)
    dso_track_lines = DSO_tracking_info.readlines()
    dso_keyframe_lines = DSO_keyframe_info.readlines()
    for line in dso_track_lines:
        frame_time = line.split('|')
        frame = frame_time[0]
        time = frame_time[1]
        if float(time) > track_time_thresh or float(time) < 0:
            continue
        # dso_track_time.append([frame, time])
        dso_track_time[int(frame)] = float(time)
    for line in dso_keyframe_lines:
        frame_time = line.split('|')
        frame = frame_time[0]
        time = frame_time[1]
        if float(time) > keyframe_time_thresh or float(time) < 0:
            continue
        dso_keyframe_time[int(frame)] = float(time)
    dso_track_time = np.asarray(list(dso_track_time.items()), dtype=float)
    dso_keyframe_time = np.asarray(list(dso_keyframe_time.items()), dtype=float)

    # process SDSO log
    sdso_track_time = {}
    sdso_keyframe_time = {}
    sdso_track_lines = SDSO_tracking_info.readlines()
    sdso_keyframe_lines = SDSO_keyframe_info.readlines()
    for line in sdso_track_lines:
        frame_time = line.split('|')
        frame = frame_time[0]
        time = frame_time[1]
        if float(time) > track_time_thresh or float(time) < 0:
            continue
        sdso_track_time[int(frame)] = float(time)
    for line in sdso_keyframe_lines:
        frame_time = line.split('|')
        frame = frame_time[0]
        time = frame_time[1]
        if float(time) > keyframe_time_thresh or float(time) < 0:
            continue
        sdso_keyframe_time[int(frame)] = float(time)
    sdso_track_time = np.asarray(list(sdso_track_time.items()), dtype=float)
    sdso_keyframe_time = np.asarray(list(sdso_keyframe_time.items()), dtype=float)

    # process DSOL log
    dsol_track_time = {}
    dsol_keyframe_time = {}
    text = ["All_Tracking", "All_Keyframe", ]
    lines = DSOL_info.readlines()
    curr_frame_id = 0
    old_kf_cnt = 0
    for line in lines:
        if text[0] in line:
            track_id = line[45:54]
            track_time = line[62:77]
            if "ms" in track_time:
                track_time = track_time.replace("ms", "")
            if "us" in track_time:
                track_time = float(track_time.replace("us", "")) / 500
            if float(track_time) > track_time_thresh or float(track_time) < 0:
                continue
            dsol_track_time[int(track_id)] = float(track_time)
        if text[1] in line:
            curr_frame_id += 1  # current frame ID
            curr_kf_cnt = line[45:54]    # dsol records keyframe total count not actual IDs.
            kf_time = line[62:77]
            if "ms" in kf_time:
                kf_time = kf_time.replace("ms", "")
            if "us" in kf_time:
                kf_time = float(kf_time.replace("us", "")) / 500
            if float(kf_time) > 500 or float(kf_time) < 0:
                continue
            if int(curr_kf_cnt) != old_kf_cnt:     # new key count number means new keyframe created. Current frame ID is the keyframe ID.
                old_kf_cnt = int(curr_kf_cnt)
                dsol_keyframe_time[curr_frame_id] = float(kf_time)
    dsol_track_time = np.asarray(list(dsol_track_time.items()), dtype=float)
    dsol_keyframe_time = np.asarray(list(dsol_keyframe_time.items()), dtype=float)

    print("Keyframes amount for DSO = ", len(dso_keyframe_time[:, 1]))
    print("Keyframes amount for SDSO = ", len(sdso_keyframe_time[:, 1]))
    print("Keyframes amount for DSOL = ", len(dsol_keyframe_time[:, 1]))
    print("Minimum keyframe time for DSO = ", np.min(dso_keyframe_time[:, 1]))
    print("Minimum keyframe time for SDSO = ", np.min(sdso_keyframe_time[:, 1]))
    print("Minimum keyframe time for DSOL = ", np.min(dsol_keyframe_time[:, 1]))
    print("Maximum keyframe time for DSO = ", np.max(dso_keyframe_time[:, 1]))
    print("Maximum keyframe time for SDSO = ", np.max(sdso_keyframe_time[:, 1]))
    print("Maximum keyframe time for DSOL = ", np.max(dsol_keyframe_time[:, 1]))
    print("Average keyframe time for DSO = ", np.mean(dso_keyframe_time[:, 1]))
    print("Average keyframe time for SDSO = ", np.mean(sdso_keyframe_time[:, 1]))
    print("Average keyframe time for DSOL = ", np.mean(dsol_keyframe_time[:, 1]))
    print("Keyframe time for DSO = ", np.std(dso_keyframe_time[:, 1]))
    print("Keyframe time for SDSO = ", np.std(sdso_keyframe_time[:, 1]))
    print("Keyframe time for DSOL = ", np.std(dsol_keyframe_time[:, 1]))

    plt.rcParams.update({'font.size': 24})

    fig = plt.figure()
    plt.scatter(dso_track_time[:, 0], dso_track_time[:, 1], marker='o', label="DSO", s=50)
    plt.scatter(sdso_track_time[:, 0], sdso_track_time[:, 1], marker='o', label="SDSO", s=50)
    plt.scatter(dsol_track_time[:, 0], dsol_track_time[:, 1], marker='o', label="DSOL", s=50)
    plt.xlabel("frame index")
    plt.ylabel("tracking time (ms)")
    plt.legend()
    # plt.title("Tracking time for different mapping algorithms")
    
    fig2 = plt.figure()
    plt.plot(dso_keyframe_time[:, 0], dso_keyframe_time[:, 1], marker='.', ms=8, label="DSO")
    plt.plot(sdso_keyframe_time[:, 0], sdso_keyframe_time[:, 1], marker='.', ms=8, label="SDSO")
    plt.plot(dsol_keyframe_time[:, 0], dsol_keyframe_time[:, 1], marker='.', ms=8, label="DSOL")
    plt.xlabel("frame index")
    plt.ylabel("keyframe time (ms)")
    plt.legend(loc='best')

    # fig = plt.figure()
    # plt.scatter(dso_track_time[15:25, 0], dso_track_time[15:25, 1], marker='o', label="DSO-track", s=5)
    # plt.scatter(sdso_track_time[80:90, 0], sdso_track_time[80:90, 1], marker='o', label="SDSO-track", s=5)
    # plt.scatter(dso_keyframe_time[4:7, 0], dso_keyframe_time[4:7, 1], marker='^', facecolors='none', label="DSO-kf", s=10)
    # plt.scatter(sdso_keyframe_time[20:24, 0], sdso_keyframe_time[20:24, 1], marker='^', facecolors='none', label="SDSO-kf", s=10)
    # plt.xlabel("frame index")
    # plt.ylabel("keyframe time (ms)")
    # plt.legend()

    plt.show()
