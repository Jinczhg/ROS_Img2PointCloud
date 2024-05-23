/* 
 * File:   image_subscriber_publisher.hpp
 * Author: jzhang72
 *
 * Created on Dec 17, 2023, 22:40 PM
 */

// ROS includes
#include <message_filters/subscriber.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_ros/point_cloud.h>
#include <nav_msgs/Odometry.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <opencv2/highgui/highgui.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/point_types.h>


typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud;

struct PCL {
    cv::Vec3f pts;
    cv::Vec3b clr;
};

class ROS_img2ptcloud {
public:
    typedef boost::shared_ptr<ROS_img2ptcloud> Ptr;

    ROS_img2ptcloud() :
    nodeptr(new ros::NodeHandle),
    nh("~"), it(nh) {
        pointCloud_pub = nh.advertise<pcl::PointCloud <pcl::PointXYZRGB> >("/pointCloud", 1);
    };

    virtual ~ROS_img2ptcloud() {};

    void initializeImg2Ptcloud();
    
    void transformPointCloud(const sensor_msgs::PointCloud2ConstPtr& cloud, const nav_msgs::OdometryConstPtr& msg);

private:

    ros::NodeHandle nh;
    ros::NodeHandlePtr nodeptr;    
    
    // image publisher
    image_transport::ImageTransport it;
    
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    
    // point cloud publisher
    ros::Publisher pointCloud_pub;
    
    message_filters::Subscriber<nav_msgs::Odometry> sub_odom;   // real-time odometry
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_pt;
};