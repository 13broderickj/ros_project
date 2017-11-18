//example_tf_listener.cpp:
//wsn, March 2016
//illustrative node to show use of tf listener, with reference to the simple mobile-robot model
// specifically, frames: odom, base_frame, link1 and link2

// this header incorporates all the necessary #include files and defines the class "DemoTfListener"
#include "example_tf_listener.h"
using namespace std;

//main pgm to illustrate transform operations
pcl::PointCloud<pcl::PointXYZRGB>::Ptr pclKinect_clr_ptr(new pcl::PointCloud<pcl::PointXYZRGB>); //readInImage
pcl::PointCloud<pcl::PointXYZRGB>::Ptr pclWorld_clr_ptr(new pcl::PointCloud<pcl::PointXYZRGB>); //transformedImage

int main(int argc, char** argv) {
    // ROS set-ups:
    ros::init(argc, argv, "demoTfListener"); //node name
    ros::NodeHandle nh; // create a node handle; need to pass this to the class constructor
    ROS_INFO("main: instantiating an object of type DemoTfListener");
    DemoTfListener demoTfListener(&nh); //instantiate an ExampleRosClass object and pass in pointer to nodehandle for constructor to use

    tf::StampedTransform stfBaseToLink2, stfBaseToLink1, stfLink1ToLink2;
    tf::StampedTransform testStfBaseToLink2;

    tf::Transform tfBaseToLink1;

    demoTfListener.tfListener_->lookupTransform("world", "kinect_pc_frame", ros::Time(0), stfBaseToLink1);
    cout << endl << "base to kinect: " << endl;
    demoTfListener.printStampedTf(stfBaseToLink1);
    tfBaseToLink1 = demoTfListener.get_tf_from_stamped_tf(stfBaseToLink1);
	Eigen::Affine3f affineTransform= XformUtils::transformStampedTfToEigenAffine3f(tfBaseToLink1);
	string snapshot_name=argv[1];
	PclUtils::transform_cloud(affineTransform,pclKinect_clr_ptr,pclWorld_clr_ptr);
    return 0;
}
