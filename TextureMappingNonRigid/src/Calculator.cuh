#pragma once

#ifndef CALCULATOR_H
#define CALCULATOR_H


#include "any.hpp"
#include "TriangleAndVertex.h"
#include "Mapper4D.h"
#include "Selector.h"

using namespace TexMap;

void calNodeweight(vector<int>& valid2all, vector<hostTriangle>& hTs, vector<set<uint>>& n_hops_tri_union, vector<set<uint>>& n_hops_ver_union, vector<vector<int>>& candidateLabels_ori, vector<vector<float3>>& stretchvec, vector<vector<float>>& curvaturevec, int nImage, float weight_c, float weight_s, float weight_q, vector<vector<vector<float>>>& result);

void calNodeweight2(vector<int>& valid2all, vector<hostTriangle>& hTs, vector<set<uint>>& n_hops_tri_union, vector<set<uint>>& n_hops_ver_union, vector<vector<int>>& candidateLabels_ori, vector<vector<float3>>& stretchvec, vector<vector<float3>>& curvaturevec, int nImage, float weight_c, float weight_s, float weight_q, vector<vector<vector<float>>>& result);

void precalFacediff(vector<int>& valid2all, vector<hostVertex>& hVs, vector<hostTriangle>& hTs, vector<vector<int>>& candidateLabels_ori, vector<uchar>& images, int nImage, int w, int h, float weight, vector<vector<float>>& result);

void precalFacediff_all(vector<hostVertex>& hVs, vector<hostTriangle>& hTs, vector<vector<int>>& candidateLabels_ori, vector<uchar>& images, int nImage, int w, int h, vector<vector<float>>& result);

void calWidediff(vector<slEdge>& edge, vector<int>& valid2all, vector<vector<int>>& candidateLabels_ori, vector<set<uint>> face_union, vector<vector<float>>& face_diff, int nImage, float weight, vector<vector<vector<float>>>& result);

void calWindowdiff(vector<int>& valid2all, vector<vector<int>>& candidateLabels_ori, vector<vector<float>>& face_diff, int nImage, int n_half_window, float weight, vector<vector<vector<float>>>& result);

void calEdgediff(vector<slEdge>& edge, vector<int>& valid2all, vector<hostVertex>& hVs, vector<hostTriangle>& hTs, vector<vector<int>>& candidateLabels_ori, vector<uchar>& images, int nImage, int w, int h, float weight, vector<vector<vector<float>>>& result);

void calFacediff(vector<int>& valid2all, vector<hostVertex>& hVs, vector<hostTriangle>& hTs, vector<vector<int>>& candidateLabels_ori, vector<uchar>& images, int nImage, int w, int h, float weight, vector<vector<vector<float>>>& result);

#endif CALCULATOR_H 