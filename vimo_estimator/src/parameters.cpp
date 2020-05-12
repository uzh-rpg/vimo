#include "parameters.h"

double THRUST_Z_N;
double THRUST_X_Y_N;
double F_EXT_NORM_WEIGHT;
double SCALE_THRUST_INPUT;
int USE_VIMO;
int PROCESS_AT_CONTROL_INPUT_RATE; 

double INIT_DEPTH;
double MIN_PARALLAX;
Eigen::Vector3d ACC_N;
double ACC_W;
double GYR_N, GYR_W; 

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;

std::string IMU_TOPIC;
std::string CONTROL_TOPIC;
double ROW, COL;
double TD, TR; // TR is rolling shutter

template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

void readParameters(ros::NodeHandle &n)
{
    std::string config_file;
    config_file = readParam<std::string>(n, "config_file");
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    fsSettings["imu_topic"] >> IMU_TOPIC;
    fsSettings["control_topic"] >> CONTROL_TOPIC;

    SOLVER_TIME = fsSettings["max_solver_time"];
    NUM_ITERATIONS = fsSettings["max_num_iterations"];
    MIN_PARALLAX = fsSettings["keyframe_parallax"];
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

    USE_VIMO = fsSettings["use_vimo"];
    if (USE_VIMO == 1)
    {
        ROS_INFO("Launching VIMO!");
    }

    PROCESS_AT_CONTROL_INPUT_RATE = fsSettings["process_at_control_input_rate"];
    if (PROCESS_AT_CONTROL_INPUT_RATE == 1)
    {
        ROS_INFO("Will run preintegration at control input frequency instead of IMU rate.");
    }

    std::string OUTPUT_PATH;
    fsSettings["output_path"] >> OUTPUT_PATH;

    if (USE_VIMO)
    {
        VINS_RESULT_PATH = OUTPUT_PATH + "/vimo_result.csv";
    }
    else
    {
        VINS_RESULT_PATH = OUTPUT_PATH + "/vins_result.csv";
    }
    
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    if (!fout || fout.bad() || fout.fail())
        std::cout << "VINS_RESULT_PATH not opened! Check if the path exists." << std::endl;
    fout.close();

    THRUST_Z_N = fsSettings["control_thrust_z_n"];
    THRUST_X_Y_N = fsSettings["control_thrust_x_y_n"];
    F_EXT_NORM_WEIGHT = fsSettings["fext_norm_weight"];
    SCALE_THRUST_INPUT = fsSettings["scale_thrust_input"];

    ACC_W = fsSettings["acc_w"];
    GYR_N = fsSettings["gyr_n"];
    GYR_W = fsSettings["gyr_w"];
    G.z() = fsSettings["g_norm"];
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    ROS_INFO("ROW: %f COL: %f ", ROW, COL);

    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
    if (ESTIMATE_EXTRINSIC == 2)
    {
        ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
        RIC.push_back(Eigen::Matrix3d::Identity());
        TIC.push_back(Eigen::Vector3d::Zero());
        EX_CALIB_RESULT_PATH = OUTPUT_PATH + "extrinsic_parameter.csv";

    }
    else 
    {
        if ( ESTIMATE_EXTRINSIC == 1)
        {
            ROS_WARN("Optimize extrinsic param around initial guess!");
            EX_CALIB_RESULT_PATH = OUTPUT_PATH + "extrinsic_parameter.csv";
        }
        if (ESTIMATE_EXTRINSIC == 0)
            ROS_WARN(" fix extrinsic param ");

        cv::Mat cv_R, cv_T, cv_acc_n;
        fsSettings["extrinsicRotation"] >> cv_R;
        fsSettings["extrinsicTranslation"] >> cv_T;
        fsSettings["acc_n"] >> cv_acc_n;
        Eigen::Matrix3d eigen_R;
        Eigen::Vector3d eigen_T;
        
        cv::cv2eigen(cv_R, eigen_R);
        cv::cv2eigen(cv_T, eigen_T);
        cv::cv2eigen(cv_acc_n, ACC_N);
        Eigen::Quaterniond Q(eigen_R);
        eigen_R = Q.normalized();
        RIC.push_back(eigen_R);
        TIC.push_back(eigen_T);
        
        ROS_INFO_STREAM("Extrinsic_R : " << std::endl << RIC[0]);
        ROS_INFO_STREAM("Extrinsic_T : " << std::endl << TIC[0].transpose());
    } 

    INIT_DEPTH = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;

    TD = fsSettings["td"];
    ESTIMATE_TD = fsSettings["estimate_td"];
    if (ESTIMATE_TD)
        ROS_INFO_STREAM("Unsynchronized sensors, online estimate time offset, initial td: " << TD);
    else
        ROS_INFO_STREAM("Synchronized sensors, fix time offset: " << TD);

    ROLLING_SHUTTER = fsSettings["rolling_shutter"];
    if (ROLLING_SHUTTER)
    {
        TR = fsSettings["rolling_shutter_tr"];
        ROS_INFO_STREAM("rolling shutter camera, read out time per line: " << TR);
    }
    else
    {
        TR = 0;
    }
    
    fsSettings.release();
}
