#pragma once

#include <ros/ros.h>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "utility/utility.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>

const double FOCAL_LENGTH = 460.0;
const int WINDOW_SIZE = 10;
const int NUM_OF_CAM = 1;
const int NUM_OF_F = 1000; // number of landmarks in the whole window i.e. total number of features observed in 10 frames

extern double THRUST_X_Y_N; //std deviation of collective rotor thrust in body x and y axis
extern double THRUST_Z_N; //std deviation of collective rotor thrust in body z axis
extern double F_EXT_NORM_WEIGHT;
extern double SCALE_THRUST_INPUT;
extern int USE_VIMO;
extern int PROCESS_AT_CONTROL_INPUT_RATE;

//#define UNIT_SPHERE_ERROR

extern double INIT_DEPTH;
extern double MIN_PARALLAX;
extern int ESTIMATE_EXTRINSIC;

extern Eigen::Vector3d ACC_N;
extern double ACC_W;
extern double GYR_N, GYR_W;

extern std::vector<Eigen::Matrix3d> RIC; // rotation of camera wrt IMU
extern std::vector<Eigen::Vector3d> TIC; // translation of camera wrt IMU
extern Eigen::Vector3d G;

extern double BIAS_ACC_THRESHOLD;
extern double BIAS_GYR_THRESHOLD;
extern double SOLVER_TIME;
extern int NUM_ITERATIONS;
extern std::string EX_CALIB_RESULT_PATH;
extern std::string VINS_RESULT_PATH;

extern std::string IMU_TOPIC;
extern std::string CONTROL_TOPIC;

extern double TD;
extern double TR;
extern int ESTIMATE_TD;
extern int ROLLING_SHUTTER;
extern double ROW, COL; //image height, width

void readParameters(ros::NodeHandle &n);

enum SIZE_PARAMETERIZATION
{
    SIZE_POSITION = 3,
    SIZE_ATTITUDE = 4, // quaternion
    SIZE_SPEED = 3, 
    SIZE_BIAS = 6, // 3 accel bias, 3 gyro bias
    SIZE_FEATURE = 1, // 1/Zc for each landmark
    SIZE_FORCES = 3
};

enum StateOrder
{
    O_P = 0,
    O_R = 3, // minimal state
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};