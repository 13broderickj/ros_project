//example_tf_listener.cpp:
//wsn, March 2016
//illustrative node to show use of tf listener, with reference to the simple mobile-robot model
// specifically, frames: odom, base_frame, link1 and link2

// this header incorporates all the necessary #include files and defines the class "DemoTfListener"
#include "example_tf_listener.h"
#include <ros/ros.h> //ALWAYS need to include this

//#include <tf/transform_listener.h>
#include <xform_utils/xform_utils.h>
//#include <sensor_msgs/LaserScan.h>
#include <iostream>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h> 
#include <pcl_ros/point_cloud.h> //use these to convert between PCL and ROS datatypes
#include <pcl/ros/conversions.h>
#include <pcl-1.7/pcl/point_cloud.h>
#include <pcl-1.7/pcl/PCLHeader.h>
#include <pcl_utils/pcl_utils.h>

using namespace std;

//main pgm to illustrate transform operations
//pcl::PointCloud<pcl::PointXYZRGB>::Ptr pclKinect_clr_ptr(new pcl::PointCloud<pcl::PointXYZRGB>); //readInImage

bool got_kinect_image = false; //snapshot indicator
pcl::PointCloud<pcl::PointXYZRGB>::Ptr pclKinect_clr_ptr(new pcl::PointCloud<pcl::PointXYZRGB>); //pointer for color version of pointcloud
pcl::PointCloud<pcl::PointXYZRGB>::Ptr pclWorld_clr_ptr(new pcl::PointCloud<pcl::PointXYZRGB>); //transformedImage
void kinectCB(const sensor_msgs::PointCloud2ConstPtr& cloud) {
    if (!got_kinect_image) { // once only, to keep the data stable
        ROS_INFO("got new selected kinect image");
        pcl::fromROSMsg(*cloud, *pclKinect_clr_ptr);
        ROS_INFO("image has  %d * %d points", pclKinect_clr_ptr->width, pclKinect_clr_ptr->height);
        got_kinect_image = true;
    }
}
int main(int argc, char** argv) {
    // ROS set-ups:
    cout << endl << "starting kinect to world " << endl;
    ros::init(argc, argv, "demoTfListener"); //node name
    ros::NodeHandle nh; // create a node handle; need to pass this to the class constructor
    ROS_INFO("main: instantiating an object of type DemoTfListener");
    DemoTfListener demoTfListener(&nh); //instantiate an ExampleRosClass object and pass in pointer to nodehandle for constructor to use

    tf::StampedTransform stfBaseToLink2, stfBaseToLink1, stfLink1ToLink2;
    tf::StampedTransform testStfBaseToLink2;

    tf::Transform tfBaseToLink1;

    demoTfListener.tfListener_->lookupTransform("base_link", "kinect_pc_frame", ros::Time(0), stfBaseToLink1);
    cout << endl << "base to kinect: " << endl;
    demoTfListener.printStampedTf(stfBaseToLink1);
    tfBaseToLink1 = demoTfListener.get_tf_from_stamped_tf(stfBaseToLink1);
	XformUtils xformutils;

    //XformUtils xFromUtilsObj= new XformUtils();
	Eigen::Affine3f affineTransform= xformutils.transformTFToAffine3f( tfBaseToLink1);


    ros::Subscriber pointcloud_subscriber = nh.subscribe("kinect/depth/points", 1, kinectCB);
	ROS_INFO("waiting for kinect data");
    while (!got_kinect_image) {
        ROS_INFO("waiting...");
        ros::spinOnce();
        ros::Duration(0.5).sleep();
    }
    ROS_INFO("got snapshot; saving to file kinect_snapshot.pcd");
	PclUtils pclutils(&nh);
	pclutils.transform_cloud(affineTransform,pclKinect_clr_ptr,pclWorld_clr_ptr);
	string snapshot_name=argv[1];
	pcl::io::savePCDFile(snapshot_name, *pclWorld_clr_ptr, true);


    return 0;
}
