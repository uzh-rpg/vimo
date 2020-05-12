#pragma once
#include <ros/assert.h>
#include <iostream>
#include <eigen3/Eigen/Dense>

#include "../utility/utility.h"
#include "../parameters.h"
#include "integration_base.h"

#include <ceres/ceres.h>

class IMUFactor : public ceres::SizedCostFunction<15, 3, 4, 3, 6, 3, 4, 3, 6>
{
  public:
    IMUFactor() = delete;
    IMUFactor(IntegrationBase* _pre_integration):pre_integration(_pre_integration)
    {
    }
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {

        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[1][3], parameters[1][0], parameters[1][1], parameters[1][2]);

        Eigen::Vector3d Vi(parameters[2][0], parameters[2][1], parameters[2][2]);
        Eigen::Vector3d Bai(parameters[3][0], parameters[3][1], parameters[3][2]);
        Eigen::Vector3d Bgi(parameters[3][3], parameters[3][4], parameters[3][5]);

        Eigen::Vector3d Pj(parameters[4][0], parameters[4][1], parameters[4][2]);
        Eigen::Quaterniond Qj(parameters[5][3], parameters[5][0], parameters[5][1], parameters[5][2]);

        Eigen::Vector3d Vj(parameters[6][0], parameters[6][1], parameters[6][2]);
        Eigen::Vector3d Baj(parameters[7][0], parameters[7][1], parameters[7][2]);
        Eigen::Vector3d Bgj(parameters[7][3], parameters[7][4], parameters[7][5]);

#if 0
        if ((Bai - pre_integration->linearized_ba).norm() > 0.10 ||
            (Bgi - pre_integration->linearized_bg).norm() > 0.01)
        {
            ROS_WARN_STREAM("repropagating!: " << (Bai - pre_integration->linearized_ba).norm() << " " << (Bgi - pre_integration->linearized_bg).norm());
            pre_integration->repropagate(Bai, Bgi);
        }
#endif

        Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);
        residual = pre_integration->evaluate(Pi, Qi, Vi, Bai, Bgi,
                                            Pj, Qj, Vj, Baj, Bgj);

        Eigen::Matrix<double, 15, 15> sqrt_info = Eigen::LLT<Eigen::Matrix<double, 15, 15>>(pre_integration->covariance.inverse()).matrixL().transpose();
        //sqrt_info.setIdentity();

        residual = sqrt_info * residual;

        if (jacobians)
        {
            double sum_dt = pre_integration->sum_dt;
            Eigen::Matrix3d dp_dba = pre_integration->jacobian.template block<3, 3>(O_P, O_BA);
            Eigen::Matrix3d dp_dbg = pre_integration->jacobian.template block<3, 3>(O_P, O_BG);

            Eigen::Matrix3d dq_dbg = pre_integration->jacobian.template block<3, 3>(O_R, O_BG);

            Eigen::Matrix3d dv_dba = pre_integration->jacobian.template block<3, 3>(O_V, O_BA);
            Eigen::Matrix3d dv_dbg = pre_integration->jacobian.template block<3, 3>(O_V, O_BG);

            if (pre_integration->jacobian.maxCoeff() > 1e8 || pre_integration->jacobian.minCoeff() < -1e8)
            {
                ROS_WARN("numerical unstable in IMU preintegration");
                //std::cout << pre_integration->jacobian << std::endl;
                //ROS_BREAK();
            }

            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> jacobian_position_i(jacobians[0]);
                jacobian_position_i.setZero();

                jacobian_position_i.block<3, 3>(O_P, O_P) = -Qi.inverse().toRotationMatrix();

                jacobian_position_i = sqrt_info * jacobian_position_i;

                if (jacobian_position_i.maxCoeff() > 1e8 || jacobian_position_i.minCoeff() < -1e8)
                {
                    ROS_WARN("numerical unstable in jacobian of IMU residual wrt position_i");
                    //std::cout << sqrt_info << std::endl;
                    //ROS_BREAK();
                }
            }
            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 4, Eigen::RowMajor>> jacobian_attitude_i(jacobians[1]);
                jacobian_attitude_i.setZero();

                jacobian_attitude_i.block<3, 3>(O_P, O_R-O_R) = Utility::skewSymmetric(Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt));

                Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
                jacobian_attitude_i.block<3, 3>(O_R, O_R-O_R) = -(Utility::Qleft(Qj.inverse() * Qi) * Utility::Qright(corrected_delta_q)).bottomRightCorner<3, 3>();

                jacobian_attitude_i.block<3, 3>(O_V, O_R-O_R) = Utility::skewSymmetric(Qi.inverse() * (G * sum_dt + Vj - Vi));

                jacobian_attitude_i = sqrt_info * jacobian_attitude_i;

                if (jacobian_attitude_i.maxCoeff() > 1e8 || jacobian_attitude_i.minCoeff() < -1e8)
                {
                    ROS_WARN("numerical unstable in jacobian of IMU residual wrt attitude_i");
                    //std::cout << sqrt_info << std::endl;
                    //ROS_BREAK();
                }
            }
            if (jacobians[2])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> jacobian_speed_i(jacobians[2]);
                jacobian_speed_i.setZero();
                jacobian_speed_i.block<3, 3>(O_P, O_V - O_V) = -Qi.inverse().toRotationMatrix() * sum_dt;
                
                jacobian_speed_i.block<3, 3>(O_V, O_V - O_V) = -Qi.inverse().toRotationMatrix();
               
                jacobian_speed_i = sqrt_info * jacobian_speed_i;

                //ROS_ASSERT(fabs(jacobian_speed_i.maxCoeff()) < 1e8);
                //ROS_ASSERT(fabs(jacobian_speed_i.minCoeff()) < 1e8);
            }
            if (jacobians[3]) // derivative of residual wrt the parameter block bias i
            {
                Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> jacobian_bias_i(jacobians[3]);
                jacobian_bias_i.setZero();
                jacobian_bias_i.block<3, 3>(O_P, O_BA - O_BA) = -dp_dba;
                jacobian_bias_i.block<3, 3>(O_P, O_BG - O_BA) = -dp_dbg;

#if 0
            jacobian_bias_i.block<3, 3>(O_R, O_BG - O_BA) = -dq_dbg;
#else
                //Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
                //jacobian_bias_i.block<3, 3>(O_R, O_BG - O_BA) = -Utility::Qleft(Qj.inverse() * Qi * corrected_delta_q).bottomRightCorner<3, 3>() * dq_dbg;
                jacobian_bias_i.block<3, 3>(O_R, O_BG - O_BA) = -Utility::Qleft(Qj.inverse() * Qi * pre_integration->delta_q).bottomRightCorner<3, 3>() * dq_dbg;
#endif

                jacobian_bias_i.block<3, 3>(O_V, O_BA - O_BA) = -dv_dba;
                jacobian_bias_i.block<3, 3>(O_V, O_BG - O_BA) = -dv_dbg;

                jacobian_bias_i.block<3, 3>(O_BA, O_BA - O_BA) = -Eigen::Matrix3d::Identity();

                jacobian_bias_i.block<3, 3>(O_BG, O_BG - O_BA) = -Eigen::Matrix3d::Identity();

                jacobian_bias_i = sqrt_info * jacobian_bias_i;

                //ROS_ASSERT(fabs(jacobian_bias_i.maxCoeff()) < 1e8);
                //ROS_ASSERT(fabs(jacobian_bias_i.minCoeff()) < 1e8);
            }
            if (jacobians[4])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> jacobian_position_j(jacobians[4]);
                jacobian_position_j.setZero();

                jacobian_position_j.block<3, 3>(O_P, O_P) = Qi.inverse().toRotationMatrix();

                jacobian_position_j = sqrt_info * jacobian_position_j;

                //ROS_ASSERT(fabs(jacobian_position_j.maxCoeff()) < 1e8);
                //ROS_ASSERT(fabs(jacobian_position_j.minCoeff()) < 1e8);
            }
            if (jacobians[5])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 4, Eigen::RowMajor>> jacobian_attitude_j(jacobians[5]);
                jacobian_attitude_j.setZero();
#if 0
            jacobian_attitude_j.block<3, 3>(O_R, O_R-O_R) = Eigen::Matrix3d::Identity();
#else
                Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
                jacobian_attitude_j.block<3, 3>(O_R, O_R-O_R) = Utility::Qleft(corrected_delta_q.inverse() * Qi.inverse() * Qj).bottomRightCorner<3, 3>();
#endif

                jacobian_attitude_j = sqrt_info * jacobian_attitude_j;

                ROS_DEBUG_STREAM_ONCE("IMU jacobian_attitude_j after:" << jacobian_attitude_j);

                //ROS_ASSERT(fabs(jacobian_attitude_j.maxCoeff()) < 1e8);
                //ROS_ASSERT(fabs(jacobian_attitude_j.minCoeff()) < 1e8);
            }
            if (jacobians[6])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> jacobian_speed_j(jacobians[6]);
                jacobian_speed_j.setZero();

                jacobian_speed_j.block<3, 3>(O_V, O_V - O_V) = Qi.inverse().toRotationMatrix();

                jacobian_speed_j = sqrt_info * jacobian_speed_j;

                //ROS_ASSERT(fabs(jacobian_speed_j.maxCoeff()) < 1e8);
                //ROS_ASSERT(fabs(jacobian_speed_j.minCoeff()) < 1e8);
            }
            if (jacobians[7])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> jacobian_bias_j(jacobians[7]);
                jacobian_bias_j.setZero();

                jacobian_bias_j.block<3, 3>(O_BA, O_BA - O_BA) = Eigen::Matrix3d::Identity();

                jacobian_bias_j.block<3, 3>(O_BG, O_BG - O_BA) = Eigen::Matrix3d::Identity();

                jacobian_bias_j = sqrt_info * jacobian_bias_j;

                //ROS_ASSERT(fabs(jacobian_bias_j.maxCoeff()) < 1e8);
                //ROS_ASSERT(fabs(jacobian_bias_j.minCoeff()) < 1e8);
            }
        }

        return true;
    }

    IntegrationBase* pre_integration;

};

