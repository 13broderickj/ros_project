//find_plane_pcd_file.cpp
// prompts for a pcd file name, reads the file, and displays to rviz on topic "pcd"
// can select a patch; then computes a plane containing that patch, which is published on topic "planar_pts"
// illustrates use of PCL methods: computePointNormal(), transformPointCloud(), 
// pcl::PassThrough methods setInputCloud(), setFilterFieldName(), setFilterLimits, filter()
// pcl::io::loadPCDFile() 
// pcl::toROSMsg() for converting PCL pointcloud to ROS message
// voxel-grid filtering: pcl::VoxelGrid,  setInputCloud(), setLeafSize(), filter()
//wsn March 2016

#include<ros/ros.h> 
#include <stdlib.h>
#include <math.h>

#include <sensor_msgs/PointCloud2.h> 
#include <pcl_ros/point_cloud.h> //to convert between PCL and ROS
#include <pcl/ros/conversions.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
//#include <pcl/PCLPointCloud2.h> //PCL is migrating to PointCloud2 

#include <pcl/common/common_headers.h>
#include <pcl-1.7/pcl/point_cloud.h>
#include <pcl-1.7/pcl/PCLHeader.h>

//will use filter objects "passthrough" and "voxel_grid" in this example
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h> 

#include <pcl_utils/pcl_utils.h>  //a local library with some utility fncs


using namespace std;
PclUtils *g_pcl_utils_ptr; 


//this fnc is defined in a separate module, find_indices_of_plane_from_patch.cpp
//void find_indices_of_plane_from_patch(pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud_ptr,
//        pcl::PointCloud<pcl::PointXYZ>::Ptr patch_cloud_ptr, vector<int> &indices);

//void find_indices_box_filtered_from_patch(pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud_ptr,
//        pcl::PointCloud<pcl::PointXYZ>::Ptr patch_cloud_ptr, Eigen::Vector3f box_pt_min, Eigen::Vector3f box_pt_max,
//        vector<int> &indices);

//Eigen::Affine3f compute_plane_affine_from_patch(pcl::PointCloud<pcl::PointXYZ>::Ptr patch_cloud_ptr); 

void find_indices_box_filtered(pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud_ptr,Eigen::Vector3f box_pt_min,  
       Eigen::Vector3f box_pt_max,vector<int> &indices) {
    int npts = input_cloud_ptr->points.size();
    Eigen::Vector3f pt;
    indices.clear();
    cout<<"box min: "<<box_pt_min.transpose()<<endl;
    cout<<"box max: "<<box_pt_max.transpose()<<endl;
    for (int i = 0; i < npts; ++i) {
        pt = input_cloud_ptr->points[i].getVector3fMap();
        //cout<<"pt: "<<pt.transpose()<<endl;
        //check if in the box:
        if ((pt[0]>box_pt_min[0])&&(pt[0]<box_pt_max[0])&&(pt[1]>box_pt_min[1])&&(pt[1]<box_pt_max[1])&&(pt[2]>box_pt_min[2])&&(pt[2]<box_pt_max[2])) { 
            //passed box-crop test; include this point
               indices.push_back(i);
        }
    }
    int n_extracted = indices.size();
    cout << " number of points within box = " << n_extracted << endl;            
}



int main(int argc, char** argv) {
    ros::init(argc, argv, "plane_finder"); //node name
	ros::NodeHandle nh;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pclKinect_clr_ptr(new pcl::PointCloud<pcl::PointXYZRGB>); //pointer for color version of pointcloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr plane_pts_ptr(new pcl::PointCloud<pcl::PointXYZRGB>); //pointer for pointcloud of planar points found
    pcl::PointCloud<pcl::PointXYZ>::Ptr selected_pts_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>); //ptr to selected pts from Rvis tool
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_kinect_ptr(new pcl::PointCloud<pcl::PointXYZRGB>); //ptr to hold filtered Kinect image
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr box_filtered_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>); //ptr to hold filtered Kinect image
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    vector<int> indices;
    Eigen::Affine3f A_plane_wrt_camera;

    //load a PCD file using pcl::io function; alternatively, could subscribe to Kinect messages    
    string fname;
    cout << "enter pcd file name: "; //prompt to enter file name
    cin >> fname;
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (fname, *pclKinect_clr_ptr) == -1) //* load the file
    {
        ROS_ERROR("Couldn't read file \n");
        return (-1);
    }
    //PCD file does not seem to record the reference frame;  set frame_id manually
    pclKinect_clr_ptr->header.frame_id = "camera_depth_optical_frame";
    ROS_INFO("view frame camera_depth_optical_frame on topics pcd, planar_pts and downsampled_pcd");

    PclUtils pclUtils(&nh); //instantiate a PclUtils object--a local library w/ some handy fncs
    g_pcl_utils_ptr = &pclUtils; // make this object shared globally, so above fnc can use it too

    cout << " select a patch of points to find corresponding plane..." << endl; //prompt user action
    //loop to test for new selected-points inputs and compute and display corresponding planar fits 
    //while(!pclUtils.got_selected_points()) {
     //   pubCloud.publish(ros_cloud);
       // ros::spinOnce(); //pclUtils needs some spin cycles to invoke callbacks for new selected points
        //ros::Duration(0.3).sleep();
    //}
    //pclUtils.get_copy_selected_points(selected_pts_cloud_ptr); //get a copy of the selected points
    //cout << "got new patch with number of selected pts = " << selected_pts_cloud_ptr->points.size() << endl;        
        
    //A_plane_wrt_camera = compute_plane_affine_from_patch(selected_pts_cloud_ptr);  
    //transform cloud to plane coords:
    //pcl::transformPointCloud(*pclKinect_clr_ptr, *transformed_cloud_ptr, A_plane_wrt_camera.inverse());

    
    
    Eigen::Vector3f box_pt_min,box_pt_max;
    box_pt_min<< -10, -10.330088, 0.691054;
    box_pt_max<< 10.853099, 10.130088, 0.891054;
    int dim;

    

   cout<<"enter dim (0,1,2): ";
   cin>>dim;
   if ((dim>=0)&&(dim<3)) {
	cout<<"enter min val: ";
	cin>>box_pt_min(dim);
	cout<<"enter max val: ";
	cin>>box_pt_max(dim); }
   else { ROS_WARN("index out of range"); }

	//find pts coplanar w/ selected patch, using PCL methods in above-defined function
	//"indices" will get filled with indices of points that are approx co-planar with the selected patch
	// can extract indices from original cloud, or from voxel-filtered (down-sampled) cloud
	//find_indices_of_plane_from_patch(pclKinect_clr_ptr, selected_pts_cloud_ptr, indices);x,y,z = 0.753099, -0.230088, 0.791054

	//find_indices_of_plane_from_patch(downsampled_kinect_ptr, selected_pts_cloud_ptr, indices);
	find_indices_box_filtered(transformed_cloud_ptr,box_pt_min,  box_pt_max,indices);
	pcl::copyPointCloud(*pclKinect_clr_ptr, indices, *box_filtered_cloud_ptr); //extract these pts into new cloud
	//the new cloud is a set of points from original cloud, coplanar with selected patch; display the result
	pcl::io::savePCDFile("filteredCloud.pcd", *box_filtered_cloud_ptr, true);

        //pubCloud.publish(ros_cloud); // will not need to keep republishing if display setting is persistent
        //pubPlane.publish(ros_planar_cloud); // display the set of points computed to be coplanar w/ selection
        //pubDnSamp.publish(downsampled_cloud); //can directly publish a pcl::PointCloud2!!
        //pubBoxFilt.publish(ros_box_filtered_cloud);
        //ros::spinOnce(); //pclUtils needs some spin cycles to invoke callbacks for new selected points
        //ros::Duration(0.3).sleep();


    return 0;
}
