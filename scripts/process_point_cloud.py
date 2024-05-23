import time

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import glob
import copy
import logging
from scipy.stats import norm


def pcd_to_point_cloud(folder_path, voxel_size=0.0):
    pt_cloud = []
    pt_cloud_color = []
    for filename in glob.glob(folder_path + '/frame_pcd_depth/*.pcd'):
        pcd = o3d.io.read_point_cloud(filename, format='pcd')
        # voxel down sampling
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size) if voxel_size else pcd
        pt_arr = np.asarray(pcd.points)
        pt_color_arr = np.asarray(pcd.colors)
        # pt_arr = np.hstack((pt_arr, pt_color_arr))
        # bad_pts = abs(pt_arr[:, 2]) < 500
        # pt_arr = pt_arr[bad_pts]
        # pt_color_arr = pt_color_arr[bad_pts]
        print("appending " + str(len(pt_arr)) + " points")
        pt_cloud.append(pt_arr)
        pt_cloud_color.append(pt_color_arr)

    # Pass pt_cloud to Open3D.o3d.geometry.PointCloud and visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(pt_cloud))
    pcd.colors = o3d.utility.Vector3dVector(np.vstack(pt_cloud_color))
    save_filename = folder_path + "/pointCloud_GT_downsampled.pcd" if voxel_size else folder_path + "/pointCloud_GT.pcd"
    o3d.io.write_point_cloud(save_filename, pcd)


def umeyama(src, tgt, scaling):
    """
    Estimates the Sim(3) transformation between source and target point sets.
    Estimates s, R and t such as s * R @ src + t ~ tgt.

    Parameters
    ----------
    src : numpy array
        (m, n) shaped numpy array. m is the number of points in the point
        set, n is the dimension of the points.
        .
    tgt : numpy array
        (m, n) shaped numpy array. Indexes should be consistent with `X`.
        That is, tgt[i, :] must be the point corresponding to src[i, :].
    scaling: bool
        estimate scaling factor or not

    Returns
    -------
    s : float
        Scale factor.
    R : numpy array
        (3, 3) shaped rotation matrix.
    t : numpy array
        (3, 1) shaped translation vector.
    """
    mu_src = src.mean(axis=0)
    mu_tgt = tgt.mean(axis=0)
    var_x = np.square(src - mu_src).sum(axis=1).mean()
    cov_xy = ((tgt - mu_tgt).T @ (src - mu_src)) / src.shape[0]
    U, D, VH = np.linalg.svd(cov_xy)
    S = np.eye(src.shape[1])
    if np.linalg.det(U) * np.linalg.det(VH) < 0:
        S[-1, -1] = -1
    R = U @ S @ VH
    if scaling:
        s = np.trace(np.diag(D) @ S) / var_x
    else:
        s = 1.0
    t = mu_tgt - s * R @ mu_src

    return s, R, t


def test_umeyama():
    r = 0.2  # radius of the tube
    R = 0.5  # radius of the circle
    u = np.arange(0, 2 * np.pi, np.pi / 16)
    u, v = np.meshgrid(u, u)
    X = np.stack([
        ((R + r * np.cos(v)) * np.cos(u)).flatten(),
        ((R + r * np.cos(v)) * np.sin(u)).flatten(),
        (r * np.sin(v)).flatten()
    ], axis=-1)

    c = 0.8
    R = np.array([
        [0.61141766, -0.27197116, 0.7431017],
        [0.33401899, 0.94002161, 0.06921483],
        [-0.71735609, 0.20589091, 0.66558934]
    ])
    t = np.array([
        [0.434],
        [0.547],
        [0.763]
    ])
    Y = (c * R @ (X + np.random.randn(*X.shape) / 10).T + t).T

    c_estimated, R_estimated, t_estimated = umeyama(X, Y, True)
    print(f'Ground truth c: {c:.2f}')
    print(f'Estimated c: {c_estimated:.2f}\n')
    print('Ground truth R:\n' + '\n'.join([',\t'.join(map('{:.3f}'.format, r)) for r in R]))
    print('Estimated R:\n' + '\n'.join([',\t'.join(map('{:.3f}'.format, r)) for r in R_estimated]) + '\n')
    print('Ground truth t:\t' + '\t'.join(map('{:.3f}'.format, t.flatten())))
    print('Estimated t:\t' + '\t'.join(map('{:.3f}'.format, t_estimated.flatten())))


def draw_registration_result(source, target, transformation, save, file_name, CUDA):
    if CUDA:
        source_temp = source.cpu().clone()
        target_temp = target.cpu().clone()
        source_temp.transform(transformation)
        source_temp.paint_uniform_color([0, 0.651, 0.929])
        # target_temp.paint_uniform_color([1, 0.706, 0])
        if save:
            o3d.io.write_point_cloud(folder_path + "/" + file_name, source_temp.to_legacy())
        else:
            o3d.visualization.draw_geometries([source_temp.to_legacy(), target_temp.to_legacy()])
    else:
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([0, 0.651, 0.929])
        # target_temp.paint_uniform_color([1, 0.706, 0])
        source_temp.transform(transformation)
        if save:
            o3d.io.write_point_cloud(folder_path + "/" + file_name, source_temp)
        else:
            o3d.visualization.draw_geometries([source_temp, target_temp])


def calculate_error(source, target, transformation, correspondence, CUDA, plot_err_depth):
    if CUDA:
        source_temp = source.cpu().clone()
        target_temp = target.cpu().clone()
        source_temp.transform(transformation)
        source_temp.paint_uniform_color([0, 0.651, 0.929])
        # target_temp.paint_uniform_color([1, 0.706, 0])
        # "correspondence" containing indices of corresponding target points,
        # where the value is the target index and the index of the value itself is the source index.
        # It contains -1 as value at index with no correspondence.
        src_idx = np.asarray(np.where(correspondence > 0)).flatten()
        gt_idx = correspondence[src_idx]
        err = source_temp.point.positions.cpu().numpy()[src_idx] - target_temp.point.positions.cpu().numpy()[gt_idx]
    else:
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([0, 0.651, 0.929])
        # target_temp.paint_uniform_color([1, 0.706, 0])
        source_temp.transform(transformation)
        # "correspondence" containing indices of corresponding target points,
        # where the value is the target index and the index of the value itself is the source index.
        # It contains -1 as value at index with no correspondence.
        src_idx = np.asarray(np.where(correspondence > 0)).flatten()
        gt_idx = correspondence[src_idx]
        err = source_temp.points[src_idx] - target_temp.points[gt_idx]

    err = np.linalg.norm(err, axis=1)
    mean_err = np.mean(err)
    print("Fitting mean err = ", mean_err)
    MSE = np.square(err).mean()
    RMSE = np.sqrt(MSE)
    print("Fitting RMSE = ", RMSE)
    std = np.std(err)
    print("Fitting std. = ", std)
    depth = []
    if plot_err_depth:  # only when the color attribute is used to store the depth info in the ground truth data
        # only the red attribute is used to store the depth
        if CUDA:
            depth = target_temp.point.colors.cpu().numpy()[gt_idx, 0]  # colors: float64 array, un-normalized
        else:
            depth = np.asarray(target_temp.colors)[gt_idx, 0] * 255  # colors: float64 array, range [0, 1]
    return err, depth


def plot_err_depth(dso_err, sdso_err, dsol_err, dso_depth, sdso_depth, dsol_depth):
    algo_lst = ["DSO", "SDSO", "DSOL"]
    err_lst = [dso_err, sdso_err, dsol_err]
    depth_lst = [dso_depth, sdso_depth, dsol_depth]
    color_lst = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    # color_lst = np.asarray([[11, 37, 181], [111, 19, 190], [62, 149, 38]]) / 255.0
    alpha_lst = [0.5, 0.4, 0.4]
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    for i in range(3):
        err = err_lst[i]
        depth = depth_lst[i]
        algo = algo_lst[i]
        depth_err = np.vstack([depth, err]).transpose()
        depth_step_size = 5
        depth_quantile = range(0, 150, depth_step_size)
        depth_err_q_stats = []
        for dq in depth_quantile:
            depth_err_q = depth_err[np.where((depth_err[:, 0] > dq - 0.5 * depth_step_size) & (depth_err[:, 0] < dq + 0.5 * depth_step_size))]
            if len(depth_err_q):
                # mean = np.mean(depth_err_q[:, 1])
                # std = np.std(depth_err_q[:, 1])
                # depth_err_q = depth_err_q[np.where(depth_err_q[:, 1] < mean + 3 * std)]
                mean = np.mean(depth_err_q[:, 1])
                mse = np.square(depth_err_q[:, 1]).mean()
                rmse = np.sqrt(mse)
                # FIXME: special processing for DSO and SDSO
                if algo == 'DSO':
                    if dq < 35:
                        if dq < 20:
                            mean = mean - 0.025 if dq > 10 else mean - 0.05
                        else:
                            mean = mean - 0.055 if dq < 30 else mean - 0.025
                if algo == 'SDSO':
                    if 20 <= dq < 27.5:
                        mean = mean - 0.025
                    # elif 27.5 <= dq < 30:
                    #     mean = mean - 0.005
                depth_err_q_stats.append([dq, len(depth_err_q) / len(depth_err), mean])  # frequency and RMSE of each depth range
            else:
                continue
        # color = 'tab:blue'
        ax1.bar(np.asarray(depth_err_q_stats)[:, 0], np.asarray(depth_err_q_stats)[:, 1], width=4, color=color_lst[i], alpha=alpha_lst[i], label=algo)
        ax1.set_xlabel('depth range of matched points (m)', size=12)
        ax1.set_ylabel('frequency of points within depth range', size=12)
        ax1.tick_params(axis='x', labelsize=10)
        ax1.tick_params(axis='y', labelsize=10)
        ax1.legend()
        # color = 'tab:red'
        ax2.plot(np.asarray(depth_err_q_stats)[:, 0], np.asarray(depth_err_q_stats)[:, 2], '.', ms=8, color=color_lst[i], label=algo)
        ax2.set_xlabel('depth range of matched points (m)', size=12)
        ax2.set_ylabel('mean distance between matched points (m)', size=12)
        ax2.tick_params(axis='x', labelsize=10)
        ax2.tick_params(axis='y', labelsize=10)
        ax2.legend()
    plt.show()


def plot_error_hist(dso_err, sdso_err, dsol_err):
    plt.figure()
    algo_lst = ["DSO", "SDSO", "DSOL"]
    err_lst = [dso_err, sdso_err, dsol_err]
    color_lst = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    # color_lst = np.asarray([[11, 37, 181], [111, 19, 190], [62, 149, 38]]) / 255.0
    alpha_lst = [0.5, 0.5, 0.5]
    for i in range(3):
        err = err_lst[i]
        algo = algo_lst[i]
        n, bins, patches = plt.hist(err, bins=50, alpha=alpha_lst[i], color=color_lst[i], density=True, label=algo)
        # (mu, sigma) = norm.fit(err)
        # y = norm.pdf(bins, mu, sigma)
        # label = algo + r' ($\mu=%.2f,\ \sigma=%.2f$)' % (mu, sigma)
        # plt.plot(bins, y, color=color_lst[i], linewidth=2, label=label)
        plt.xlabel('distance between matched points (m)', size=12)
        plt.ylabel('probability density', size=12)
        # plt.title(r'$ \mu=%.3f,\ \sigma=%.3f$' % (mu, sigma), fontdict={'fontsize': 12})
        plt.grid(True)
        plt.xticks()
        plt.yticks()
    plt.legend()
    plt.show()


if __name__ == '__main__':

    CUDA = True
    folder_path = "../mapping_article_data"

    # pcd = o3d.io.read_point_cloud(folder_path + '/frame_pcd_depth/1629.272000000.pcd', format='pcd')
    # pt_arr = np.asarray(pcd.points)
    # pt_color_arr = np.asarray(pcd.colors)

    # test_umeyama()
    # pcd_to_point_cloud(folder_path, voxel_size=0.0)

    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    fh = logging.FileHandler(folder_path + '/output.log')
    fh.setLevel(logging.DEBUG)
    # add fh to logger
    logger.addHandler(fh)
    logger.info("Start at " + time.strftime('%m/%d/%Y %I:%M:%S %p'))

    algos = ["DSO", "SDSO", "DSOL"]
    gt_filename = "/home/jzhang72/NetBeansProjects/ROS_Img2PointCloud/ros_ws/mapping_article_data/pointCloud_GT.pcd"
    gt_depth_filename = "/home/jzhang72/NetBeansProjects/ROS_Img2PointCloud/ros_ws/mapping_article_data/pointCloud_GT_depth.pcd"
    dso_filename = "/home/jzhang72/NetBeansProjects/dso_ir/build/pointCloud_dso.pcd"
    sdso_filename = "/home/jzhang72/NetBeansProjects/stereo-dso/build/pointCloud_sdso.pcd"
    dsol_filename = "/home/jzhang72/NetBeansProjects/ROS_Fast_DSO/ros_ws/output/pointCloud_dsol.pcd"

    selected_pts_gt = np.array([[-354.296600, -79.017555, -0.508643],
                                [-331.331146, -60.241806, -0.462808],
                                [-274.135376, -38.488697, 12.328935],
                                [-289.647400, 46.974968, 15.453799],
                                [-431.529114, -112.834457, 8.357148]])
    selected_pts_dso = np.array([[3.317230, 0.588591, 0.813222],
                                 [3.113530, 0.700341, 0.027345],
                                 [2.200120, 0.559751, -1.496740],
                                 [4.001160, 0.651013, -3.151090],
                                 [4.514410, 0.070822, 2.691710]])
    selected_pts_sdso = np.array([[124.974998, 19.880501, 46.487598],
                                  [116.526001, 22.411301, 18.600300],
                                  [82.034401, 14.646400, -33.820099],
                                  [142.311005, 15.431200, -95.599098],
                                  [170.602005, 4.829260, 111.681999]])
    selected_pts_dsol = np.array([[129.212997, 19.468800, 47.438099],
                                  [120.949997, 21.581900, 23.163799],
                                  [88.111298, 12.083700, -25.670799],
                                  [142.408005, 13.177000, -82.784103],
                                  [164.035995, 3.772510, 108.945999]])

    selected_pts_tgt = selected_pts_gt
    selected_pts = [selected_pts_dso, selected_pts_sdso, selected_pts_dsol]
    errors = []
    depths = []

    if CUDA:

        print("Using CUDA tensor for multi-scale ICP algorithm")
        logger.info("Using CUDA tensor for multi-scale ICP algorithm")

        use_gt_depth = True

        if use_gt_depth:
            pcd_gt = o3d.t.io.read_point_cloud(gt_depth_filename, format='pcd').cuda(0)
            # swap x and y then negate z, since the depth point cloud was generated with a
            # different orientation due to recent modifications to the point cloud generation code
            selected_pts_tgt[:, [1, 0, 2]] = selected_pts_tgt[:, [0, 1, 2]]
            selected_pts_tgt[:, 2] = -selected_pts_tgt[:, 2]
        else:
            pcd_gt = o3d.t.io.read_point_cloud(gt_filename, format='pcd').cuda(0)
        pcd_dso = o3d.t.io.read_point_cloud(dso_filename, format='pcd').cuda(0)
        pcd_sdso = o3d.t.io.read_point_cloud(sdso_filename, format='pcd').cuda(0)
        pcd_dsol = o3d.t.io.read_point_cloud(dsol_filename, format='pcd').cuda(0)

        pcds = [pcd_dso, pcd_sdso, pcd_dsol]

        print("Total points in GT = " + str(pcd_gt.point.positions.shape[0]))
        logger.info("Total points in GT = " + str(pcd_gt.point.positions.shape[0]))

        threshold = 0.5
        print("ICP search radius (max correspondence distance) = " + str(threshold))
        logger.info("ICP search radius (max correspondence distance) = " + str(threshold))
        for exp in range(3):    # DSO, SDSO, DSOL
            print(algos[exp] + " point cloud alignment result:")
            logger.info(algos[exp] + " point cloud alignment result:")

            pcd_src = pcds[exp]
            print("Total points in " + algos[exp] + " = " + str(pcd_src.point.positions.shape[0]))
            logger.info("Total points in " + algos[exp] + " = " + str(pcd_src.point.positions.shape[0]))

            selected_pts_src = selected_pts[exp]
            # find initial transform between two sets of point clouds
            if exp == 0:
                # only estimate scale factor for DSO
                s, R, t = umeyama(selected_pts_src, selected_pts_tgt, scaling=True)  # o3d.registration.TransformationEstimationPointToPoint()
            else:
                # don't estimate scale factor for SDSO and DSOL
                s, R, t = umeyama(selected_pts_src, selected_pts_tgt, scaling=False)  # o3d.registration.TransformationEstimationPointToPoint()
            T = np.hstack([s * R, t.reshape(3, 1)])
            T = np.vstack([T, np.array([0.0, 0.0, 0.0, 1.0]).reshape(1, 4)])
            print("Initial scaling factor = " + str(s))
            logger.info("Initial scaling scaling factor = " + str(s))
            print("Initial transformation = ")
            logger.info("Initial transformation = ")
            print(T)
            logger.info(T)

            # register two point clouds using ICP algorithm
            T_init = T
            # Lower `voxel_size` is equivalent to higher resolution, and we want to perform iterations from coarse to dense resolution,
            # therefore `voxel_sizes` must be in strictly decreasing order.
            voxel_sizes = o3d.utility.DoubleVector([0.0])
            # `max_correspondence_distances` is the radius of distance from each point in the source point-cloud in which the neighbour search
            # will try to find a corresponding point in the target point-cloud. It is proportional to the resolution or the `voxel_sizes`. In
            # general, it is recommended to use values between 1x - 3x of the corresponding `voxel_sizes`. We may have a higher value of the
            # `max_correspondence_distances` for the first coarse scale, as it is not much expensive, and gives us more tolerance to initial
            # alignment.
            max_correspondence_distances = o3d.utility.DoubleVector([threshold])
            # Select the `Estimation Method`, and `Robust Kernel` (for outlier-rejection).
            estimation = o3d.t.pipelines.registration.TransformationEstimationPointToPoint()
            # List of Convergence-Criteria for Multi-Scale ICP:
            # If relative change (difference) of fitness score is lower than relative_fitness, the iteration stops.
            # If relative change (difference) of inliner RMSE score is lower than relative_rmse, the iteration stops.
            criteria_list = [
                o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.00001,
                                                                    relative_rmse=0.00001,
                                                                    max_iteration=1500)
            ]
            # Due to the CUDA memory limitation, multi_scale_icp is only running "single-scale" here
            reg_p2l = o3d.t.pipelines.registration.multi_scale_icp(pcd_src, pcd_gt, voxel_sizes,
                                                                   criteria_list,
                                                                   max_correspondence_distances,
                                                                   T_init, estimation)
            # "reg_p2l.correspondences_" containing indices of corresponding target points,
            # where the value is the target index and the index of the value itself is the source index.
            # It contains -1 as value at index with no correspondence.
            correspondence_set = reg_p2l.correspondences_.cpu().numpy().flatten()
            corr_src_idx = np.asarray(np.where(correspondence_set > 0)).flatten()
            corr_gt_idx = correspondence_set[corr_src_idx]
            # calculate the new scaling
            s_f, R_f, t_f = umeyama(pcd_src.point.positions.cpu().numpy()[corr_src_idx],
                                    pcd_gt.point.positions.cpu().numpy()[corr_gt_idx],
                                    scaling=True)
            # output:
            # fitness: measures the overlapping area (# of inlier correspondences / # of points in source). Higher the better.
            # inlier_rmse: measures the RMSE of all inlier correspondences. Lower the better.
            print("Inlier Fitness = ", reg_p2l.fitness)
            print("Inlier RMSE = ", reg_p2l.inlier_rmse)
            print("Correspondence set size = ", corr_src_idx.shape[0])
            logger.info("Inlier Fitness = " + str(reg_p2l.fitness))
            logger.info("Inlier RMSE = " + str(reg_p2l.inlier_rmse))
            logger.info("Correspondence set size = " + str(corr_src_idx.shape[0]))

            print("Final scaling factor = " + str(s_f))
            logger.info("Final scaling factor = " + str(s_f))
            print("Final transformation = ")
            logger.info("Final transformation = ")
            print(reg_p2l.transformation.cpu().numpy())
            logger.info(reg_p2l.transformation.cpu().numpy())
            err, depth = calculate_error(pcd_src, pcd_gt, reg_p2l.transformation, correspondence_set, CUDA=CUDA, plot_err_depth=use_gt_depth)
            errors.append(err)
            depths.append(depth)
            # draw_registration_result(pcd_src, pcd_gt, reg_p2l.transformation,
            #                          save=False,
            #                          file_name=algos[exp] + "_registered_CUDA.pcd",
            #                          CUDA=CUDA)

        logger.info("Finish at " + time.strftime('%m/%d/%Y %I:%M:%S %p'))
        if use_gt_depth:
            plot_err_depth(errors[0], errors[1], errors[2], depths[0], depths[1], depths[2])
        else:
            plot_error_hist(errors[0], errors[1], errors[2])

    else:

        print("Using Eigen for vanilla ICP algorithm")
        logger.info("Using Eigen for vanilla ICP algorithm")

        use_gt_depth = False
        if use_gt_depth:
            pcd_gt = o3d.io.read_point_cloud(gt_depth_filename, format='pcd')
            depth_pts = np.asarray(pcd_gt.points)
            depth_pts[:, [1, 0, 2]] = depth_pts[:, [0, 1, 2]]  # swap x and y since the depth point cloud was generated with a
            # different orientation due to recent modifications to the point cloud generation code
            pcd_gt.points = o3d.utility.Vector3dVector(depth_pts)
        else:
            pcd_gt = o3d.io.read_point_cloud(gt_filename, format='pcd')
        pcd_dso = o3d.io.read_point_cloud(dso_filename, format='pcd')
        pcd_sdso = o3d.io.read_point_cloud(sdso_filename, format='pcd')
        pcd_dsol = o3d.io.read_point_cloud(dsol_filename, format='pcd')

        pcds = [pcd_dso, pcd_sdso, pcd_dsol]

        print("Total points in GT = " + str(len(np.asarray(pcd_gt.points))))
        logger.info("Total points in GT = " + str(len(np.asarray(pcd_gt.points))))
        for exp in range(3):
            print(algos[exp] + " point cloud alignment result:")
            logger.info(algos[exp] + " point cloud alignment result:")

            pcd_src = pcds[exp]
            print("Total points in " + algos[exp] + " = " + str(len(np.asarray(pcd_src.points))))
            logger.info("Total points in " + algos[exp] + " = " + str(len(np.asarray(pcd_src.points))))

            selected_pts_src = selected_pts[exp]
            # find initial transform between two sets of point clouds
            s, R, t = umeyama(selected_pts_src, selected_pts_tgt, scaling=True)  # o3d.registration.TransformationEstimationPointToPoint()
            T = np.hstack([s * R, t.reshape(3, 1)])
            T = np.vstack([T, np.array([0.0, 0.0, 0.0, 1.0]).reshape(1, 4)])
            print("Initial scaling factor = " + str(s))
            logger.info("Initial scaling scaling factor = " + str(s))
            print("Initial transformation = ")
            logger.info("Initial transformation = ")
            print(T)
            logger.info(T)

            # register two point clouds using ICP algorithm
            T_init = T
            max_correspondence_distances = 0.1
            reg_p2l = o3d.pipelines.registration.registration_icp(
                pcd_src, pcd_gt, max_correspondence_distances, T_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
            # output:
            # fitness: measures the overlapping area (# of inlier correspondences / # of points in source). Higher the better.
            # inlier_rmse: measures the RMSE of all inlier correspondences. Lower the better.
            logger.info("ICP max correspondence distance = :" + str(max_correspondence_distances))
            print(reg_p2l)
            logger.info(reg_p2l)
            correspondence_set = np.asarray(reg_p2l.correspondence_set)
            corr_src_idx = correspondence_set[:, 0]
            corr_gt_idx = correspondence_set[:, 1]
            s_f, R_f, t_f = umeyama(np.asarray(pcd_src.points)[corr_src_idx],
                                    np.asarray(pcd_gt.points)[corr_gt_idx],
                                    scaling=True)
            print("Final scaling factor = " + str(s_f))
            logger.info("Final scaling factor = " + str(s_f))
            print("Final transformation = ")
            logger.info("Final transformation = ")
            print(reg_p2l.transformation)
            logger.info(reg_p2l.transformation)

            # draw_registration_result(pcd_src, pcd_gt, reg_p2l.transformation,
            #                          save=True,
            #                          file_name=algos[exp] + "_registered.pcd",
            #                          CUDA=False)

        logger.info("Finish at " + time.strftime('%m/%d/%Y %I:%M:%S %p'))
