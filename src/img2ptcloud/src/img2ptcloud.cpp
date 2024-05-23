/* 
 * File:   image_subscriber_publisher.hpp
 * Author: jzhang72
 *
 * Created on Dec 17, 2023, 22:40 PM
 */

#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <img2ptcloud/img2ptcloud.hpp>
#include <pcl_conversions/pcl_conversions.h>

#define MAX_DEPTH 150

void ROS_img2ptcloud::transformPointCloud(const sensor_msgs::PointCloud2ConstPtr& cloud, const nav_msgs::OdometryConstPtr& msg) {   
    
    // transform matrix from odometry
    // process the pose
    float transX = msg->pose.pose.position.x; // msg->pose.pose.position.y;
    float transY = msg->pose.pose.position.y; // msg->pose.pose.position.x;
    float transZ = msg->pose.pose.position.z; // -msg->pose.pose.position.z;
    float qX = msg->pose.pose.orientation.x; // msg->pose.pose.orientation.y;
    float qY = msg->pose.pose.orientation.y; // msg->pose.pose.orientation.x;
    float qZ = msg->pose.pose.orientation.z; // -msg->pose.pose.orientation.z;
    float qW = msg->pose.pose.orientation.w;
    Eigen::Quaternionf quat(qW, qX, qY, qZ);
    
    // bad code (next 8 lines)
//    float transX = msg->pose.pose.position.y; // msg->pose.pose.position.y;
//    float transY = msg->pose.pose.position.x; // msg->pose.pose.position.x;
//    float transZ = -msg->pose.pose.position.z; // -msg->pose.pose.position.z;
//    float qX = msg->pose.pose.orientation.y; // msg->pose.pose.orientation.y;
//    float qY = msg->pose.pose.orientation.x; // msg->pose.pose.orientation.x;
//    float qZ = -msg->pose.pose.orientation.z; // -msg->pose.pose.orientation.z;
//    float qW = msg->pose.pose.orientation.w;
//    Eigen::Quaternionf quat(qW, qX, qY, qZ);
    
    transform.block(0, 0, 3, 3) = quat.toRotationMatrix();
    transform(0, 3) = transX;
    transform(1, 3) = transY;
    transform(2, 3) = transZ;
    // bad code (next 4 lines)
//    transform.block(0, 0, 3, 3) = quat.toRotationMatrix();
//    transform(0, 3) = transY;
//    transform(1, 3) = transX;
//    transform(2, 3) = -transZ;
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcd(new pcl::PointCloud <pcl::PointXYZRGB> ());
    for(int i = 0 ; i < cloud->width * cloud->height; ++i){
       auto* ptr = reinterpret_cast<const float*>(cloud->data.data() + i * cloud->point_step);       
       if (std::isnan(ptr[0]) or std::isnan(ptr[1]) or std::isnan(ptr[2]) or ptr[2] > MAX_DEPTH)
           continue;
       pcl::PointXYZRGB point;
       point.x = ptr[2]; // ptr[2] is the depth
       point.y = ptr[0]; // ptr[0] is the x
       point.z = ptr[1]; // ptr[1] is the y
       
       // bad code (next 3 lines)
//        point.x = ptr[0];
//        point.y = ptr[2];
//        point.z = -ptr[1];
        
       // Process Depth: pack point depth values as rgb color
//       std::uint8_t r = ptr[2]; // ptr[2] / MAX_DEPTH * 255; // depends on what MAX_DEPTH is. If MAX_DEPTH > 255, then normalization is necessary.
//       std::uint8_t g = 0, b = 0;    // Example: Red color represented by depth
//       std::uint32_t rgb = ((std::uint32_t)r << 16 | (std::uint32_t)g << 8 | (std::uint32_t)b);
//       point.rgb = *reinterpret_cast<float*>(&rgb);
       // Process RGB: pack actual rgb color
       point.rgb = ptr[4]; // ptr[4] is the uint32_t RGB color  // FIELDS x y z _ rgb _ // SIZE 4 4 4 1 4 1 // TYPE F F F U F U // COUNT 1 1 1 4 1 12
       pcd->points.push_back(point);
       
       // DEBUG
//       std::uint8_t r = ((*reinterpret_cast<const int*>(&ptr[4])) >> 16) & 0x0000ff;
//       std::uint8_t g = ((*reinterpret_cast<const int*>(&ptr[4])) >> 8)  & 0x0000ff;
//       std::uint8_t b = (*reinterpret_cast<const int*>(&ptr[4]))       & 0x0000ff;
//       if (i==205120) 
//           std::cout << "center point = (" << ptr[0] << ", " << ptr[1] << ", " << ptr[2] << "); color = (" 
//               << int(r) << ", " << int(g) << ", " << int(b) << ") " << std::endl;
    }

    // https://wiki.ros.org/hydro/Migration#PCL
    pcd->header = pcl_conversions::toPCL(cloud->header); 
    pcd->header.frame_id = "drone_1";   // rename the frame id
    pcd->width = pcd->points.size();
    pcd->height = 1;
    pcd->is_dense = false;
    // Executing the transformation
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tfPCD(new pcl::PointCloud<pcl::PointXYZRGB> ());
    pcl::transformPointCloud (*pcd, *tfPCD, transform);
    // filter out bad points
//    pcl::PassThrough<pcl::PointXYZRGB> pass;
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filteredPCD(new pcl::PointCloud<pcl::PointXYZRGB> ());
//    pass.setInputCloud (tfPCD);
//    pass.setFilterFieldName ("z");
//    pass.setFilterLimits (0.0, 500.0);
//    pass.filter (*filteredPCD);
    pointCloud_pub.publish(tfPCD);
}

void ROS_img2ptcloud::initializeImg2Ptcloud() {
    
    int queue_size = 10;
    
    // odometry 
    sub_odom.subscribe(*nodeptr, "/airsim_node/drone_1/odom_local_ned", 1);
    
    // point cloud from depth_image_proc (camera coordinate)
    sub_pt.subscribe(*nodeptr, "/airsim_node/drone_1/leftCamera_forward_custom/points", 1);
        
    // sync point cloud with odometry
    typedef message_filters::sync_policies::ApproximateTime
                <sensor_msgs::PointCloud2, nav_msgs::Odometry> MyApproximateSyncPolicy;
    static message_filters::Synchronizer<MyApproximateSyncPolicy> mySyncApprox(MyApproximateSyncPolicy(queue_size),
                sub_pt, sub_odom);
    mySyncApprox.registerCallback(boost::bind(&ROS_img2ptcloud::transformPointCloud,
                this, _1, _2));
}

int main(int argc, char** argv)
{
    /**
     * The ros::init() function needs to see argc and argv so that it can perform
     * any ROS arguments and name remapping that were provided at the command line.
     * For programmatic remappings you can use a different version of init() which takes
     * remappings directly, but for most command-line programs, passing argc and argv is
     * the easiest way to do it.  The third argument to init() is the name of the node.
     *
     * You must call one of the versions of ros::init() before using any other
     * part of the ROS system.
     */
    ros::init(argc, argv, "image_to_point_cloud");
    /**
     * NodeHandle is the main access point to communications with the ROS system.
     * The first NodeHandle constructed will fully initialize this node, and the last
     * NodeHandle destructed will close down the node.
     */
    ROS_img2ptcloud img2ptcloud;

    img2ptcloud.initializeImg2Ptcloud();
    ros::spin();
    return EXIT_SUCCESS;
}

