#include "projection_td_factor.h"

Eigen::Matrix2d ProjectionTdFactor::sqrt_info;
double ProjectionTdFactor::sum_t;

ProjectionTdFactor::ProjectionTdFactor(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j, 
                                       const Eigen::Vector2d &_velocity_i, const Eigen::Vector2d &_velocity_j,
                                       const double _td_i, const double _td_j, const double _row_i, const double _row_j) : 
                                       pts_i(_pts_i), pts_j(_pts_j), 
                                       td_i(_td_i), td_j(_td_j)
{
    velocity_i.x() = _velocity_i.x();
    velocity_i.y() = _velocity_i.y();
    velocity_i.z() = 0;
    velocity_j.x() = _velocity_j.x();
    velocity_j.y() = _velocity_j.y();
    velocity_j.z() = 0;
    row_i = _row_i - ROW / 2;
    row_j = _row_j - ROW / 2;

#ifdef UNIT_SPHERE_ERROR
    Eigen::Vector3d b1, b2;
    Eigen::Vector3d a = pts_j.normalized();
    Eigen::Vector3d tmp(0, 0, 1);
    if(a == tmp)
        tmp << 1, 0, 0;
    b1 = (tmp - a * (a.transpose() * tmp)).normalized();
    b2 = a.cross(b1);
    tangent_base.block<1, 3>(0, 0) = b1.transpose();
    tangent_base.block<1, 3>(1, 0) = b2.transpose();
#endif
};

bool ProjectionTdFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    TicToc tic_toc;
    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Qi(parameters[1][3], parameters[1][0], parameters[1][1], parameters[1][2]);

    Eigen::Vector3d Pj(parameters[2][0], parameters[2][1], parameters[2][2]);
    Eigen::Quaterniond Qj(parameters[3][3], parameters[3][0], parameters[3][1], parameters[3][2]);

    Eigen::Vector3d tic(parameters[4][0], parameters[4][1], parameters[4][2]);
    Eigen::Quaterniond qic(parameters[5][3], parameters[5][0], parameters[5][1], parameters[5][2]);

    double inv_dep_i = parameters[6][0];

    double td = parameters[7][0];

    Eigen::Vector3d pts_i_td, pts_j_td;
    pts_i_td = pts_i - (td - td_i + TR / ROW * row_i) * velocity_i;
    pts_j_td = pts_j - (td - td_j + TR / ROW * row_j) * velocity_j;
    Eigen::Vector3d pts_camera_i = pts_i_td / inv_dep_i;
    Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
    Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
    Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
    Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);
    Eigen::Map<Eigen::Vector2d> residual(residuals);

#ifdef UNIT_SPHERE_ERROR 
    residual =  tangent_base * (pts_camera_j.normalized() - pts_j_td.normalized());
#else
    double dep_j = pts_camera_j.z();
    residual = (pts_camera_j / dep_j).head<2>() - pts_j_td.head<2>();
#endif

    residual = sqrt_info * residual;

    if (jacobians)
    {
        Eigen::Matrix3d Ri = Qi.toRotationMatrix();
        Eigen::Matrix3d Rj = Qj.toRotationMatrix();
        Eigen::Matrix3d ric = qic.toRotationMatrix();
        Eigen::Matrix<double, 2, 3> reduce(2, 3);
#ifdef UNIT_SPHERE_ERROR
        double norm = pts_camera_j.norm();
        Eigen::Matrix3d norm_jaco;
        double x1, x2, x3;
        x1 = pts_camera_j(0);
        x2 = pts_camera_j(1);
        x3 = pts_camera_j(2);
        norm_jaco << 1.0 / norm - x1 * x1 / pow(norm, 3), - x1 * x2 / pow(norm, 3),            - x1 * x3 / pow(norm, 3),
                     - x1 * x2 / pow(norm, 3),            1.0 / norm - x2 * x2 / pow(norm, 3), - x2 * x3 / pow(norm, 3),
                     - x1 * x3 / pow(norm, 3),            - x2 * x3 / pow(norm, 3),            1.0 / norm - x3 * x3 / pow(norm, 3);
        reduce = tangent_base * norm_jaco;
#else
        reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j),
            0, 1. / dep_j, -pts_camera_j(1) / (dep_j * dep_j);
#endif
        reduce = sqrt_info * reduce;

        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jacobian_position_i(jacobians[0]);

            Eigen::Matrix<double, 3, 3> jaco_i;
            jaco_i = ric.transpose() * Rj.transpose();
            
            jacobian_position_i = reduce * jaco_i;
            
        }

        if (jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> jacobian_attitude_i(jacobians[1]);

            Eigen::Matrix<double, 3, 3> jaco_i;
            jaco_i = ric.transpose() * Rj.transpose() * Ri * -Utility::skewSymmetric(pts_imu_i);

            jacobian_attitude_i.leftCols<3>() = reduce * jaco_i;
            jacobian_attitude_i.rightCols<1>().setZero();
        }

        if (jacobians[2])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jacobian_position_j(jacobians[2]);

            Eigen::Matrix<double, 3, 3> jaco_j;
            jaco_j = ric.transpose() * -Rj.transpose();

            jacobian_position_j = reduce * jaco_j;
            
        }

         if (jacobians[3])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> jacobian_attitude_j(jacobians[3]);

            Eigen::Matrix<double, 3, 3> jaco_j;
            jaco_j = ric.transpose() * Utility::skewSymmetric(pts_imu_j);

            jacobian_attitude_j.leftCols<3>() = reduce * jaco_j;
            jacobian_attitude_j.rightCols<1>().setZero();
        }

        if (jacobians[4])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jacobian_ex_position(jacobians[4]);
            Eigen::Matrix<double, 3, 3> jaco_ex;
            jaco_ex = ric.transpose() * (Rj.transpose() * Ri - Eigen::Matrix3d::Identity());
            
            jacobian_ex_position = reduce * jaco_ex;
            
        }
        if (jacobians[5])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> jacobian_ex_attitude(jacobians[5]);
            Eigen::Matrix<double, 3, 3> jaco_ex;
            
            Eigen::Matrix3d tmp_r = ric.transpose() * Rj.transpose() * Ri * ric;
            jaco_ex = -tmp_r * Utility::skewSymmetric(pts_camera_i) + Utility::skewSymmetric(tmp_r * pts_camera_i) +
                                     Utility::skewSymmetric(ric.transpose() * (Rj.transpose() * (Ri * tic + Pi - Pj) - tic));
            jacobian_ex_attitude.leftCols<3>() = reduce * jaco_ex;
            jacobian_ex_attitude.rightCols<1>().setZero();
        }

        if (jacobians[6])
        {
            Eigen::Map<Eigen::Vector2d> jacobian_feature(jacobians[6]);
            jacobian_feature = reduce * ric.transpose() * Rj.transpose() * Ri * ric * pts_i_td * -1.0 / (inv_dep_i * inv_dep_i);
        }
        if (jacobians[7])
        {
            Eigen::Map<Eigen::Vector2d> jacobian_td(jacobians[7]);
            jacobian_td = reduce * ric.transpose() * Rj.transpose() * Ri * ric * velocity_i / inv_dep_i * -1.0  +
                          sqrt_info * velocity_j.head(2);
        }
    }
    sum_t += tic_toc.toc();

    return true;
}