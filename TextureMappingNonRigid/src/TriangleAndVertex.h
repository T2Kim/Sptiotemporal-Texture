#pragma once

#ifndef TAV_H
#define TAV_H
// -------------------- OpenMesh
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_functions.h>
#include <helper_cuda.h> 
#include <helper_math.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <config_utils.h>


struct hostTriangle {
	int		_Vertices[3];
	float2  _Tex_BIGPIC[3];
	float3	_Normal;
	std::vector<float> _Area;

	std::vector<int> _Img;
	std::vector<bool> _Bound;
	std::vector<bool> _Label;
	std::vector<float> _Img_Weight;

	int selected_img = -1;

	hostTriangle() {
		_Vertices[0] = -1; _Vertices[1] = -1; _Vertices[2] = -1;

		_Img.clear();
		_Label.clear();
		_Bound.clear();
		_Img_Weight.clear();
	}
	int getImgIdx(int targetImg) {
		for (int i = 0; i < _Img.size(); i++) {
			if (targetImg == _Img[i]) {
				return i;
			}
		}
		return -1;
	}
	float getImgWeight(int targetImg) {
		for (int i = 0; i < _Img.size(); i++) {
			if (targetImg == _Img[i]) {
				return _Img_Weight[i];
			}
		}
		return -1.0;
	}
	void write(ofstream* file) {
		file->write((char*)&_Vertices[0], sizeof(int) * 3);
		file->write((char*)&_Tex_BIGPIC[0], sizeof(float2) * 3);
		file->write((char*)&_Normal, sizeof(float3));

		int _Area_sz = _Area.size();
		file->write((char*)&_Area_sz, sizeof(int));
		if(_Area_sz > 0)
			file->write((char*)&_Area[0], sizeof(float) * _Area_sz);

		int _Img_sz = _Img.size();
		file->write((char*)&_Img_sz, sizeof(int));
		if (_Img_sz > 0)
			file->write((char*)&_Img[0], sizeof(int) * _Img_sz);

		int _Bound_sz = _Bound.size();
		file->write((char*)&_Bound_sz, sizeof(int));
		if (_Bound_sz > 0)
			file->write((char*)&_Bound[0], sizeof(bool) * _Bound_sz);

		int _Label_sz = _Label.size();
		file->write((char*)&_Label_sz, sizeof(int));
		if (_Label_sz > 0)
			file->write((char*)&_Label[0], sizeof(bool) * _Label_sz);

		int _Img_Weight_sz = _Img_Weight.size();
		file->write((char*)&_Img_Weight_sz, sizeof(int));
		if (_Img_Weight_sz > 0)
			file->write((char*)&_Img_Weight[0], sizeof(float) * _Img_Weight_sz);
		return;
	}
	void read(ifstream* file) {
		file->read((char*)&_Vertices[0], sizeof(int) * 3);
		file->read((char*)&_Tex_BIGPIC[0], sizeof(float2) * 3);
		file->read((char*)&_Normal, sizeof(float3));

		int _Area_sz, _Img_sz, _Bound_sz, _Label_sz, _Img_Weight_sz;

		file->read((char*)&_Area_sz, sizeof(int));
		if (_Area_sz > 0) {
			_Area.resize(_Area_sz);
			file->read((char*)&_Area[0], sizeof(float) * _Area_sz);
		}

		file->read((char*)&_Img_sz, sizeof(int));
		if (_Img_sz > 0) {
			_Img.resize(_Img_sz);
			file->read((char*)&_Img[0], sizeof(int) * _Img_sz);
		}
		file->read((char*)&_Bound_sz, sizeof(int));
		if (_Bound_sz > 0) {
			_Bound.resize(_Bound_sz);
			for (int i = 0; i < _Bound_sz; i++) {
				bool a;
				file->read((char*)&a, 1);
				_Bound[i] = a;
			}
		}

		file->read((char*)&_Label_sz, sizeof(int));
		if (_Label_sz > 0) {
			_Label.resize(_Label_sz);
			for (int i = 0; i < _Label_sz; i++) {
				bool a;
				file->read((char*)&a, 1);
				_Label[i] = a;
			}
		}

		file->read((char*)&_Img_Weight_sz, sizeof(int));
		if (_Img_Weight_sz > 0) {
			_Img_Weight.resize(_Img_Weight_sz);
			file->read((char*)&_Img_Weight[0], sizeof(float) * _Img_Weight_sz);
		}
	}
};
struct hostVertex {

	//x, y, z
	float3	_Pos;
	float3	_Col;

	std::vector<float3> _Pos_time;
	std::vector<float3> _Norm_time;

	std::vector<int> _Triangles;
	std::vector<int> _Img;
	std::vector<float2> _Img_Tex;
	std::vector<float2> _Img_Tex_ori;

	hostVertex() {
		_Pos.x = 0.0f; _Pos.y = 0.0f; _Pos.z = 0.0f;
		_Col.x = 0.0f; _Col.y = 0.0f; _Col.z = 0.0f;

		_Triangles.clear();
		_Img.clear();
		_Img_Tex.clear();
	}
	int getImgIdx(int targetImg) {
		for (int i = 0; i < _Img.size(); i++) {
			if (targetImg == _Img[i]) {
				return i;
			}
		}
		return -1;
	}
	void roll_back() {
		_Img_Tex.clear();
		_Img_Tex.assign(_Img_Tex_ori.begin(), _Img_Tex_ori.end());
	}
	void write(ofstream* file) {
		file->write((char*)&_Pos, sizeof(float3));
		file->write((char*)&_Col, sizeof(float3));

		int _Pos_time_sz = _Pos_time.size();
		file->write((char*)&_Pos_time_sz, sizeof(int));
		if (_Pos_time_sz > 0)
			file->write((char*)&_Pos_time[0], sizeof(float3) * _Pos_time_sz);

		int _Norm_time_sz = _Norm_time.size();
		file->write((char*)&_Norm_time_sz, sizeof(int));
		if (_Norm_time_sz > 0)
			file->write((char*)&_Norm_time[0], sizeof(float3) * _Norm_time_sz);

		int _Triangles_sz = _Triangles.size();
		file->write((char*)&_Triangles_sz, sizeof(int));
		if (_Triangles_sz > 0)
			file->write((char*)&_Triangles[0], sizeof(int) * _Triangles_sz);

		int _Img_sz = _Img.size();
		file->write((char*)&_Img_sz, sizeof(int));
		if (_Img_sz > 0)
			file->write((char*)&_Img[0], sizeof(int) * _Img_sz);

		int _Img_Tex_sz = _Img_Tex.size();
		file->write((char*)&_Img_Tex_sz, sizeof(int));
		if (_Img_Tex_sz > 0)
			file->write((char*)&_Img_Tex[0], sizeof(float2) * _Img_Tex_sz);

		int _Img_Tex_ori_sz = _Img_Tex_ori.size();
		file->write((char*)&_Img_Tex_ori_sz, sizeof(int));
		if (_Img_Tex_ori_sz > 0)
			file->write((char*)&_Img_Tex_ori[0], sizeof(float2) * _Img_Tex_ori_sz);
		return;
	}
	void read(ifstream* file) {
		file->read((char*)&_Pos, sizeof(float3));
		file->read((char*)&_Col, sizeof(float3));

		int _Pos_time_sz = _Pos_time.size();
		file->read((char*)&_Pos_time_sz, sizeof(int));
		if (_Pos_time_sz > 0) {
			_Pos_time.resize(_Pos_time_sz);
			file->read((char*)&_Pos_time[0], sizeof(float3) * _Pos_time_sz);
		}
		int _Norm_time_sz = _Norm_time.size();
		file->read((char*)&_Norm_time_sz, sizeof(int));
		if (_Norm_time_sz > 0) {
			_Norm_time.resize(_Norm_time_sz);
			file->read((char*)&_Norm_time[0], sizeof(float3) * _Norm_time_sz);
		}
		int _Triangles_sz = _Triangles.size();
		file->read((char*)&_Triangles_sz, sizeof(int));
		if (_Triangles_sz > 0) {
			_Triangles.resize(_Triangles_sz);
			file->read((char*)&_Triangles[0], sizeof(int) * _Triangles_sz);
		}
		int _Img_sz = _Img.size();
		file->read((char*)&_Img_sz, sizeof(int));
		if (_Img_sz > 0) {
			_Img.resize(_Img_sz);
			file->read((char*)&_Img[0], sizeof(int) * _Img_sz);
		}
		int _Img_Tex_sz = _Img_Tex.size();
		file->read((char*)&_Img_Tex_sz, sizeof(int));
		if (_Img_Tex_sz > 0) {
			_Img_Tex.resize(_Img_Tex_sz);
			file->read((char*)&_Img_Tex[0], sizeof(float2) * _Img_Tex_sz);
		}
		int _Img_Tex_ori_sz = _Img_Tex_ori.size();
		file->read((char*)&_Img_Tex_ori_sz, sizeof(int));
		if (_Img_Tex_ori_sz > 0) {
			_Img_Tex_ori.resize(_Img_Tex_ori_sz);
			file->read((char*)&_Img_Tex_ori[0], sizeof(float2) * _Img_Tex_ori_sz);
		}
		return;
	}
};
struct deviceTriangle {
	int		_Vertices[3];

	int _Img_Num;

	size_t _imgOffset;

	//sample intensity

	__device__ __host__ deviceTriangle() {
		_Vertices[0] = -1; _Vertices[1] = -1; _Vertices[2] = -1;
		_Img_Num = 0;
		_imgOffset = 0;
	}
	__host__ void init(hostTriangle &hT, size_t imgOffset) {
		int tmp_Img_Num = hT._Img.size();
		checkCudaErrors(cudaMemcpy(&_Img_Num, &tmp_Img_Num, sizeof(int), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(_Vertices, hT._Vertices, sizeof(int) * 3, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(&_imgOffset, &imgOffset, sizeof(size_t), cudaMemcpyHostToDevice));
	}
	__device__ __host__ bool isCoveredBy(int targetImg, int *_Img) {
		for (int i = 0; i < _Img_Num; i++) {
			if (targetImg == _Img[i])
				return true;
		}
		return false;
	}
	__device__ __host__ int getImgIdx(int targetImg, int* _Img) {
		for (int i = 0; i < _Img_Num; i++) {
			if (targetImg == _Img[i]) {
				return i;
			}
		}
		return -1;
	}
	__device__ __host__ float weightedBy(int targetImg, int *_Img, float *_Img_Weight) {
		for (int i = 0; i < _Img_Num; i++) {
			if (targetImg == _Img[i])
				return _Img_Weight[i];
		}
		return 0.0;
	}
};
struct deviceVertex {

	int _Triangles_Num;
	int _Img_Num;

	size_t _imgOffset;
	size_t _triOffset;
	size_t _edgeOffset;

	__device__ __host__ deviceVertex() {
		_Triangles_Num = 0;
		_Img_Num = 0;
		_imgOffset = 0;
		_triOffset = 0;
		_edgeOffset = 0;
	}
	__host__ void init(hostVertex &hV, size_t imgOffset, size_t triOffset, size_t edgeOffset) {
		int tmp_Triangles_Num = hV._Triangles.size();
		int tmp_Img_Num = hV._Img.size();
		checkCudaErrors(cudaMemcpy(&_Triangles_Num, &tmp_Triangles_Num, sizeof(int), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(&_Img_Num, &tmp_Img_Num, sizeof(int), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(&_imgOffset, &imgOffset, sizeof(size_t), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(&_triOffset, &triOffset, sizeof(size_t), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(&_edgeOffset, &edgeOffset, sizeof(size_t), cudaMemcpyHostToDevice));
	}
	__device__ __host__ int getImgIdx(int targetImg, int *_Img) {
		for (int i = 0; i < _Img_Num; i++) {
			if (targetImg == _Img[i]) {
				return i;
			}
		}
		return -1;
	}
};

class deviceMemHolderT {
	thrust::device_vector<unsigned char> _SamCol_t;
	thrust::device_vector<float> _Weight_t;
	thrust::device_vector<int> _Img_t;
	thrust::device_vector<bool> _Bound_t;
	thrust::device_vector<bool> _Label_t;

public:
	unsigned char *_SamCol;
	int *_Img;
	float *_Weight;
	bool *_Bound;
	bool *_Label;
	__host__ void push_back(hostTriangle &hT);

	__host__ void update_bound(vector<bool>& boundary, vector<bool>& label);

	__host__ void update_bound(bool value);

	__host__ void clear() {
		_SamCol_t.clear();
		_Img_t.clear();
		_Weight_t.clear();
		_Bound_t.clear();
		_Label_t.clear();
		_SamCol = NULL;
		_Img = NULL;
		_Weight = NULL;
		_Bound = NULL;
		_Label = NULL;
	}

	__host__ void ready() {
		_SamCol = thrust::raw_pointer_cast(&_SamCol_t[0]);
		_Img = thrust::raw_pointer_cast(&_Img_t[0]);
		_Weight = thrust::raw_pointer_cast(&_Weight_t[0]);
		_Bound = thrust::raw_pointer_cast(&_Bound_t[0]);
		_Label = thrust::raw_pointer_cast(&_Label_t[0]);
	}
};

class deviceMemHolderV {
	thrust::device_vector<int> _Triangles_t;
	thrust::device_vector<int> _Img_t;
	thrust::device_vector<float2> _Img_Tex_t;
	thrust::device_vector<float2> _Edge_Init_t;

public:
	int *_Triangles;
	int *_Img;
	float2 *_Img_Tex;
	float2 *_Edge_Init;
	__host__ void push_back(hostVertex &hV);

	__host__ void update_texcoord(vector<float2>& img_tex);

	__host__ void clear() {
		_Triangles_t.clear();
		_Img_t.clear();
		_Img_Tex_t.clear();
		_Edge_Init_t.clear();
		_Triangles = NULL;
		_Img = NULL;
		_Img_Tex = NULL;
		_Edge_Init = NULL;
	}

	__host__ void ready() {
		_Triangles = thrust::raw_pointer_cast(&_Triangles_t[0]);
		_Img = thrust::raw_pointer_cast(&_Img_t[0]);
		_Img_Tex = thrust::raw_pointer_cast(&_Img_Tex_t[0]);
		_Edge_Init = thrust::raw_pointer_cast(&_Edge_Init_t[0]);
	}
};

struct mDummyImage {
	int _Img;
	float2	_Img_Tex;
	float _Weight;
	bool operator<(const mDummyImage& a) const
	{
		return _Weight > a._Weight;
	}
	mDummyImage() {
		_Weight = 0;
	}
};
// ¼Õº¼°Í
/*
static void mWriteTriangle(mTriangle &_T, std::ofstream &fout) {


	fout.write((char*)&(_T._Vertices[0]), sizeof(int));
	fout.write((char*)&(_T._Vertices[1]), sizeof(int));
	fout.write((char*)&(_T._Vertices[2]), sizeof(int));

	for (int i = 0; i < 3; i++) {
		fout.write((char*)&(_T._Tex_BIGPIC[i].x), sizeof(float));
		fout.write((char*)&(_T._Tex_BIGPIC[i].y), sizeof(float));
	}


	fout.write((char*)&(_T._Normal.x), sizeof(float));
	fout.write((char*)&(_T._Normal.y), sizeof(float));
	fout.write((char*)&(_T._Normal.z), sizeof(float));

	fout.write((char*)&(_T._Img_Num), sizeof(int));

	for (int i = 0; i < MAXIMG; i++) {
		fout.write((char*)&(_T._Img[i]), sizeof(int));
		fout.write((char*)&(_T._Img_Weight[i]), sizeof(float));
	}
}

static void mWriteVertex(mVertex &_V, std::ofstream &fout) {
	fout.write((char*)&(_V._Triangles_Num), sizeof(int));

	for (int i = 0; i < MAXTRI; i++) {
		fout.write((char*)&(_V._Triangles[i]), sizeof(int));
	}

	fout.write((char*)&(_V._Pos.x), sizeof(float));
	fout.write((char*)&(_V._Pos.y), sizeof(float));
	fout.write((char*)&(_V._Pos.z), sizeof(float));
	fout.write((char*)&(_V._Col.x), sizeof(float));
	fout.write((char*)&(_V._Col.y), sizeof(float));
	fout.write((char*)&(_V._Col.z), sizeof(float));

	fout.write((char*)&(_V._Img_Num), sizeof(int));

	for (int i = 0; i < MAXIMG; i++) {
		fout.write((char*)&(_V._Img[i]), sizeof(int));
		fout.write((char*)&(_V._Img_Tex[i].x), sizeof(float));
		fout.write((char*)&(_V._Img_Tex[i].y), sizeof(float));
	}
}
*/

#endif TAV_H