#pragma once
#include <ros/assert.h>
#include <iostream>
#include <eigen3/Eigen/Dense>

#include "../utility/utility.h"
#include "../parameters.h"
#include "integration_base.h"

#include <ceres/ceres.h>


class ModelFactor : public ceres::SizedCostFunction<9, 3, 4, 3, 3, 3, 3> //position i, attitude i,  speed i, fext i, position j, speed j
{
  public:
    ModelFactor() = delete;
    ModelFactor(IntegrationBase* _pre_integration):pre_integration(_pre_integration)
    {
    }
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {

        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[1][3], parameters[1][0], parameters[1][1], parameters[1][2]);

        Eigen::Vector3d Vi(parameters[2][0], parameters[2][1], parameters[2][2]);

        Eigen::Vector3d Fexti(parameters[3][0], parameters[3][1], parameters[3][2]);

        Eigen::Vector3d Pj(parameters[4][0], parameters[4][1], parameters[4][2]);

        Eigen::Vector3d Vj(parameters[5][0], parameters[5][1], parameters[5][2]);


        Eigen::Map<Eigen::Matrix<double, 9, 1>> residual(residuals);
        residual = pre_integration->evaluate_model(Pi, Qi, Vi, Fexti,
                                            Pj, Vj);

        
        Eigen::Matrix<double, 6, 6> covariance_model_pv;
        covariance_model_pv.setIdentity();
        covariance_model_pv.block<3, 3>(0, 0) = pre_integration->covariance_model.block<3, 3>(O_P, O_P);
        covariance_model_pv.block<3, 3>(0, 3) = pre_integration->covariance_model.block<3, 3>(O_P, O_V);
        covariance_model_pv.block<3, 3>(3, 0) = pre_integration->covariance_model.block<3, 3>(O_V, O_P);
        covariance_model_pv.block<3, 3>(3, 3) = pre_integration->covariance_model.block<3, 3>(O_V, O_V);
        
        Eigen::Matrix<double, 9, 9> info_matrix;
        info_matrix.setZero();
        info_matrix.block<6, 6>(0,0) = covariance_model_pv.inverse();
        info_matrix.block<3, 3>(6,6) = F_EXT_NORM_WEIGHT * Eigen::Matrix3d::Identity();

        Eigen::Matrix<double, 9, 9> sqrt_info;
        sqrt_info = Eigen::LLT<Eigen::Matrix<double, 9, 9>>(info_matrix).matrixL().transpose();

        residual = sqrt_info * residual;

        if (jacobians)
        {
            double sum_dt = pre_integration->sum_dt;

            if (jacobians[0]) // derivative of residual wrt the first parameter block i.e. 3D position_i
            {
                Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> jacobian_position_i(jacobians[0]);
                jacobian_position_i.setZero();

                jacobian_position_i.block<3, 3>(O_P, O_P) = -Qi.inverse().toRotationMatrix();

                jacobian_position_i = sqrt_info * jacobian_position_i;

                if (jacobian_position_i.maxCoeff() > 1e8 || jacobian_position_i.minCoeff() < -1e8)
                {
                    ROS_WARN("numerical unstable in jacobian of model residual wrt position_i");
                    //std::cout << sqrt_info << std::endl;
                    //ROS_BREAK();
                }
                ROS_DEBUG_STREAM_ONCE("Model jacobian_position_i after:" << jacobian_position_i);
            }
            if (jacobians[1]) // derivative of residual wrt parameter block i.e. 4D attitude_i
            {
                Eigen::Map<Eigen::Matrix<double, 9, 4, Eigen::RowMajor>> jacobian_attitude_i(jacobians[1]);
                jacobian_attitude_i.setZero();

                jacobian_attitude_i.block<3, 3>(O_P, O_R-O_R) = Utility::skewSymmetric(Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt));

                jacobian_attitude_i.block<3, 3>(O_V-3, O_R-O_R) = Utility::skewSymmetric(Qi.inverse() * (G  * sum_dt + Vj - Vi));
              
                jacobian_attitude_i = sqrt_info * jacobian_attitude_i;

                if (jacobian_attitude_i.maxCoeff() > 1e8 || jacobian_attitude_i.minCoeff() < -1e8)
                {
                    ROS_WARN("numerical unstable in jacobian of model residual wrt attitude_i");
                    //std::cout << sqrt_info << std::endl;
                    //ROS_BREAK();
                }
            }
            if (jacobians[2]) // derivative of residual wrt parameter block i.e. 3D speed i
            {
                Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> jacobian_speed_i(jacobians[2]);
                jacobian_speed_i.setZero();
                
                jacobian_speed_i.block<3, 3>(O_P, O_V - O_V) = -Qi.inverse().toRotationMatrix() * sum_dt;

                jacobian_speed_i.block<3, 3>(O_V-3, O_V - O_V) = -Qi.inverse().toRotationMatrix();

                jacobian_speed_i = sqrt_info * jacobian_speed_i;

                if (jacobian_speed_i.maxCoeff() > 1e8 || jacobian_speed_i.minCoeff() < -1e8)
                {
                    ROS_WARN("numerical unstable in jacobian of model residual wrt speed_i");
                    //std::cout << sqrt_info << std::endl;
                    //ROS_BREAK();
                }

                //ROS_ASSERT(fabs(jacobian_speed_i.maxCoeff()) < 1e8);
                //ROS_ASSERT(fabs(jacobian_speed_i.minCoeff()) < 1e8);
            }
            if (jacobians[3]) // derivative of residual wrt  3D external force
            {
                Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> jacobian_fext_i(jacobians[3]);
                jacobian_fext_i.setZero();

                jacobian_fext_i.block<3, 3>(O_P, 0) = - 0.5 * sum_dt * sum_dt * Eigen::Matrix3d::Identity();
                jacobian_fext_i.block<3, 3>(O_V-3, 0) = - sum_dt * Eigen::Matrix3d::Identity();
                jacobian_fext_i.block<3, 3>(6, 0) = Eigen::Matrix3d::Identity();

                jacobian_fext_i = sqrt_info * jacobian_fext_i;

                if (jacobian_fext_i.maxCoeff() > 1e8 || jacobian_fext_i.minCoeff() < -1e8)
                {
                    ROS_WARN("numerical unstable in jacobian of model residual wrt fext_i");
                    //std::cout << sqrt_info << std::endl;
                    //ROS_BREAK();
                }

                //ROS_ASSERT(fabs(jacobian_fext_i.maxCoeff()) < 1e8);
                //ROS_ASSERT(fabs(jacobian_fext_i.minCoeff()) < 1e8);
            }
            if (jacobians[4]) // derivative of residual wrt  3D position j
            {
                Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> jacobian_position_j(jacobians[4]);
                jacobian_position_j.setZero();

                jacobian_position_j.block<3, 3>(O_P, O_P) = Qi.inverse().toRotationMatrix();

                jacobian_position_j = sqrt_info * jacobian_position_j;
                ROS_DEBUG_STREAM_ONCE("Model jacobian_position_j after:" << jacobian_position_j);

                if (jacobian_position_j.maxCoeff() > 1e8 || jacobian_position_j.minCoeff() < -1e8)
                {
                    ROS_WARN("numerical unstable in jacobian of model residual wrt position_j");
                    //std::cout << sqrt_info << std::endl;
                    //ROS_BREAK();
                }

                //ROS_ASSERT(fabs(jacobian_position_j.maxCoeff()) < 1e8);
                //ROS_ASSERT(fabs(jacobian_position_j.minCoeff()) < 1e8);
            }
            if (jacobians[5])
            {
                Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> jacobian_speed_j(jacobians[5]);
                jacobian_speed_j.setZero();

                jacobian_speed_j.block<3, 3>(O_V-3, O_V - O_V) = Qi.inverse().toRotationMatrix();

                jacobian_speed_j = sqrt_info * jacobian_speed_j;

                if (jacobian_speed_j.maxCoeff() > 1e8 || jacobian_speed_j.minCoeff() < -1e8)
                {
                    ROS_WARN("numerical unstable in jacobian of model residual wrt speed_j");
                    //std::cout << sqrt_info << std::endl;
                    //ROS_BREAK();
                }

                //ROS_ASSERT(fabs(jacobian_speed_j.maxCoeff()) < 1e8);
                //ROS_ASSERT(fabs(jacobian_speed_j.minCoeff()) < 1e8);
            }
        }

        return true;
    }

    IntegrationBase* pre_integration;

};

