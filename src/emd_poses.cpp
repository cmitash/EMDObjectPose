#include <iostream>
#include <fstream>
#include <string>
#include <ros/ros.h>
#include <stdlib.h>
#include <time.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/filters/radius_outlier_removal.h>

#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <sensor_msgs/PointCloud2.h>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudRGB;

std::string repo_path = "/home/chaitanya/EMDObjectPose";

static void getAffineTransform(tf::StampedTransform& TFtransform, Eigen::Affine3f& Afftransform) {
    Afftransform.translation() << (float)TFtransform.getOrigin()[0], 
                                  (float)TFtransform.getOrigin()[1], 
                                  (float)TFtransform.getOrigin()[2];

    Afftransform.rotate (Eigen::Quaternionf((float)TFtransform.getRotation().getW(),
                        (float)TFtransform.getRotation()[0], 
                        (float)TFtransform.getRotation()[1],
                        (float)TFtransform.getRotation()[2]));
}

int bins = 10;
int histSize[] = {bins, bins, bins};
float xranges[] = { -0.1, 0.1 };
float yranges[] = { -0.1, 0.1 };
float zranges[] = { -0.1, 0.1 };
int channels[] = {0, 1, 2};
const float* ranges[] = { xranges, yranges, zranges};

static void getEMDDistance(cv::Mat sig1, PointCloud::Ptr pcl_2, float& emd){
  cv::MatND hist_2;
  int num_rows = pcl_2->points.size();
  cv::Mat xyzPts_2(num_rows, 1, CV_32FC3);
  for(int ii=0; ii<num_rows; ii++){
    xyzPts_2.at<cv::Vec3f>(ii,0)[0] = pcl_2->points[ii].x;
    xyzPts_2.at<cv::Vec3f>(ii,0)[1] = pcl_2->points[ii].y;
    xyzPts_2.at<cv::Vec3f>(ii,0)[2] = pcl_2->points[ii].z;
  }
  cv::calcHist( &xyzPts_2, 1, channels, cv::Mat(), hist_2, 3, histSize, ranges, true, false);

  //make signature
  int sigSize = bins*bins*bins;
  cv::Mat sig2(sigSize, 4, CV_32FC1);

  //fill value into signature
  for(int x=0; x<bins; x++) {
    for(int y=0; y<bins; ++y) {
      for(int z=0; z<bins; ++z) {
        float binval = hist_2.at<float>(x,y,z);
        sig2.at<float>( x*bins*bins + y*bins + z, 0) = binval;
        sig2.at<float>( x*bins*bins + y*bins + z, 1) = x;
        sig2.at<float>( x*bins*bins + y*bins + z, 2) = y;
        sig2.at<float>( x*bins*bins + y*bins + z, 3) = z;
      }
    }
  }

   emd = cv::EMD(sig1, sig2, CV_DIST_L2); //emd 0 is best matching. 
}

int main( int ac, char* av[] ) {
    srand (time(NULL));
    std::string node_name(3, 'x');
    node_name[0] = "absdefghijklmnopzrstuvwzyz"[rand() % 26];
    node_name[1] = "absdefghijklmnopzrstuvwzyz"[rand() % 26];
    node_name[2] = "absdefghijklmnopzrstuvwzyz"[rand() % 26];

    std::cout << node_name <<std::endl;

    ros::init(ac, av, node_name);
    ros::NodeHandle main_node_handle;

    std::string modelName;
    tf::StampedTransform model_tf;
    PointCloud::Ptr modelCloud;

    double cent_x, cent_y, cent_z;
    double roll, pitch, yaw;

    PointCloud::Ptr combinedCloud(new PointCloud);

    // load model cloud
    modelName = std::string(av[1]);
    modelCloud = PointCloud::Ptr(new PointCloud);
    pcl::io::loadPCDFile (repo_path + "/models/" + modelName + ".pcd", *modelCloud);

    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud (modelCloud);
    sor.setLeafSize (0.01f, 0.01f, 0.01f);
    sor.filter (*modelCloud);

    ofstream fileout;
    fileout.open ((repo_path + "/" + modelName + ".txt").c_str(), std::ofstream::out | std::ofstream::app);
    
    std::vector<PointCloud::Ptr> transformed_clouds;
    PointCloud::Ptr transformed_cloud (new PointCloud);

    int num_rows = modelCloud->points.size();
    std::cout << num_rows <<std::endl;
    cv::Mat xyzPts_1(num_rows, 1, CV_32FC3);
    for(int ii=0; ii<num_rows; ii++){
      xyzPts_1.at<cv::Vec3f>(ii,0)[0] = modelCloud->points[ii].x;
      xyzPts_1.at<cv::Vec3f>(ii,0)[1] = modelCloud->points[ii].y;
      xyzPts_1.at<cv::Vec3f>(ii,0)[2] = modelCloud->points[ii].z;
    }
    cv::MatND hist_1;
    cv::calcHist( &xyzPts_1, 1, channels, cv::Mat(), hist_1, 3, histSize, ranges, true, false);
    int sigSize = bins*bins*bins;
    cv::Mat sig1(sigSize, 4, CV_32FC1);
    for(int x=0; x<bins; x++) {
      for(int y=0; y<bins; ++y) {
        for(int z=0; z<bins; ++z) {
          float binval = hist_1.at<float>(x,y,z);
          sig1.at<float>( x*bins*bins + y*bins + z, 0) = binval;
          sig1.at<float>( x*bins*bins + y*bins + z, 1) = x;
          sig1.at<float>( x*bins*bins + y*bins + z, 2) = y;
          sig1.at<float>( x*bins*bins + y*bins + z, 3) = z;
        }
      }
    }

    for(cent_x=-0.02; cent_x<=0.02; cent_x=cent_x+0.01){
      for(cent_y=-0.02; cent_y<=0.02; cent_y=cent_y+0.01){
        for(cent_z=-0.02; cent_z<=0.02; cent_x=cent_z+0.01){
          clock_t begin_time = clock();
          for(roll=0; roll<360; roll=roll+15){
            for(pitch=0; pitch<360; pitch=pitch+15){
              for(yaw=0; yaw<360; yaw=yaw+15){
                float emd;
                model_tf.setOrigin( tf::Vector3(cent_x, cent_y, cent_z) );
                model_tf.setRotation( tf::createQuaternionFromRPY(roll*M_PI/180,pitch*M_PI/180,yaw*M_PI/180) );
                Eigen::Affine3f transform_1 = Eigen::Affine3f::Identity();
                getAffineTransform(model_tf, transform_1);
                pcl::transformPointCloud (*modelCloud, *transformed_cloud, transform_1);
                *combinedCloud += *transformed_cloud;
                getEMDDistance(sig1, transformed_cloud, emd);
                fileout << emd << std::endl;
              }
            }
          }
          std::cout << "time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
        }
      }
    }

    fileout.close();
    ros::shutdown();
    return 0;
}
