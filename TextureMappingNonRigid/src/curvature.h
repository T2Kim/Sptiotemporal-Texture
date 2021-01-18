#ifndef CURVATURES_H
#define CURVATURES_H
#include <Eigen/Core>
#include "TriangleAndVertex.h"
// Approximate principal curvature values and directions locally by considering
// the two-ring neighborhood of each vertex in the mesh (`V`,`F`).
//
// Inputs:
//   V  #V by 3 list of mesh vertex positions
//   F  #F by 3 list of mesh face indices into V
// Outputs:
//   D1  #V by 3 list of first principal (unit) directions 
//   D2  #V by 3 list of second principal (unit) directions 
//   K1  #V by 3 list of first principal values (maximum curvature)
//   K2  #V by 3 list of second principal values (minimum curvature)
//
void get_curvature(vector<hostVertex>& Vert, vector<hostTriangle>& Tri, int idx, vector<float>& result);
void get_curvature_descriptors(vector<hostVertex>& Vert, vector<hostTriangle>& Tri, int idx, Eigen::VectorXd& out_K1, Eigen::VectorXd& out_K2, Eigen::VectorXd& out_theta);

void curvatures(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    Eigen::MatrixXd& D1,
    Eigen::MatrixXd& D2,
    Eigen::VectorXd& K1,
    Eigen::VectorXd& K2);


void curvatures_normal(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    Eigen::MatrixXd& D1,
    Eigen::MatrixXd& D2,
    Eigen::VectorXd& K1,
    Eigen::VectorXd& K2,
    Eigen::MatrixXd& N);
#endif