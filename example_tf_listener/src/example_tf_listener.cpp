//example_tf_listener.cpp:
//wsn, March 2016
//illustrative node to show use of tf listener, with reference to the simple mobile-robot model
// specifically, frames: odom, base_frame, link1 and link2

// this header incorporates all the necessary #include files and defines the class "DemoTfListener"
#include "example_tf_listener.h"
using namespace std;

//main pgm to illustrate transform operations

int main(int argc, char** argv) {
    // ROS set-ups:
    ros::init(argc, argv, "demoTfListener"); //node name
    ros::NodeHandle nh; // create a node handle; need to pass this to the class constructor
    ROS_INFO("main: instantiating an object of type DemoTfListener");
    DemoTfListener demoTfListener(&nh); //instantiate an ExampleRosClass object and pass in pointer to nodehandle for constructor to use

    tf::StampedTransform stfBaseToLink2, stfBaseToLink1, stfLink1ToLink2;
    tf::StampedTransform testStfBaseToLink2;

    tf::Transform tfBaseToLink1, tfLink1ToLink2, tfBaseToLink2, altTfBaseToLink2;

    demoTfListener.tfListener_->lookupTransform("base_link", "kinect_pc_frame", ros::Time(0), stfBaseToLink1);
    cout << endl << "base to link1: " << endl;
    demoTfListener.printStampedTf(stfBaseToLink1);
    tfBaseToLink1 = demoTfListener.get_tf_from_stamped_tf(stfBaseToLink1);

    return 0;
}
