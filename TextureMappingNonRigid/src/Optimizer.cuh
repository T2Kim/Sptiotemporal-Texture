#pragma once

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include <helper_functions.h>
//#include <helper_cuda.h> 
//#include <helper_math.h>
//#include <thrust/device_vector.h>
//#include <thrust/host_vector.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <fstream>
#include <string>
#include "Selector.h"
#include "Mapper4D.h"


#include <glm/glm.hpp>

#define cudaCheckError() {                                          \
	 cudaError_t e=cudaGetLastError();                                 \
	 if(e!=cudaSuccess) {                                              \
	   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
	   exit(0); \
	  }                                                         \
	}
//// �ٲܰ�
/*
struct Vertex_Save
{
	int _Triangles_Num;
	int _Triangles[MAXTRI];
	float3 _Pos;
	float3 _Col;
	int _Img_Num;
	int _Img[MAXIMG];
	float2 _Img_Tex[MAXIMG];
};
struct Triangle_Save
{
	int _Vertices[3];
	float2 _Tex_BIGPIC[3];
	float3 _Normal;
	int _Img_Num;
	int _Img[MAXIMG];
	float _Img_Weight[MAXIMG];
};
struct Vertex_Load {
	int _Triangles_Num;
	int _Triangles[MAXTRI];

	float3	_Pos;
	float3	_Col;

	int		_Img_Num;
	int _Img[MAXIMG];
	float2 _Img_Tex[MAXIMG];

	__host__ __device__ Vertex_Load() {
		_Pos.x = 0.0f; _Pos.y = 0.0f; _Pos.z = 0.0f;

		_Triangles_Num = 0;
		_Img_Num = 0;
		for (int i = 0; i < MAXTRI; i++) {
			_Triangles[i] = -1;
			_Img[i] = -1;
		}
	}
	__device__ __host__ int getImgIdx(int targetImg) {
		for (int i = 0; i < _Img_Num; i++) {
			if (targetImg == _Img[i]) {
				return i;
			}
		}
		return -1;
	}
};
struct Triangle_Load {
	int		_Vertices[3];
	float2  _Tex_BIGPIC[3];
	float3	_Normal;
	//sample intensity
	uchar	_SamCol[SAMNUM];

	float hue;

	int		_Img_Num;
	int		_Img[MAXIMG];
	float	_Img_Weight[MAXIMG];

	__device__ __host__ Triangle_Load() {
		_Vertices[0] = -1; _Vertices[1] = -1; _Vertices[2] = -1;

		_Img_Num = 0;
		for (int i = 0; i < MAXIMG; i++) _Img[i] = -1;
	}
	__device__ __host__ bool isCoveredBy(int targetImg) {
		for (int i = 0; i < _Img_Num; i++) {
			if (targetImg == _Img[i])
				return true;
		}
		return false;
	}
};
*/

namespace TexMap{
	class Optimizer {
	public:

		uchar *hCImages;
		Mapper4D *mapper4D_ptr;
		Selector *selector_ptr;
		
		Optimizer() {}
		Optimizer(string opt_mode) {this->opt_mode = opt_mode; }
		~Optimizer();

		void SetMode(string opt_mode) {this->opt_mode = opt_mode; }

		void Initialize(int layer, bool use_keyframe = true, bool resue = false);
		void Clear();
		void Reinit(int layer, int timestamp);
		int Update(int iteration, int layer, bool layerup = true);
		void Update_toAtlas(int iteration, int layer);
		StopWatchInterface* sample_host(int width, int height, int nVertex, int nTriangle, float *hEnergy, int currentIter);
		StopWatchInterface* sample_host_sel(int width, int height, int nVertex, int nTriangle, float *hEnergy, int currentIter);
		StopWatchInterface* sample_host_toAtlas(int width, int height, int nVertex, int nTriangle, float* hEnergy);
		//void WriteModel(string outFilePath);
		//void GetModel(Triangle_Load *tempT, Vertex_Load *tempV);
		void PrepareRend_UVAtlas();
		vector<bool> PrepareRend_UVAtlas_video(int targetIdx);
		vector<bool> PrepareRend_UVAtlas_video_mask(int targetIdx);
		void GetAtlasInfoi_UVAtlas(std::vector<float> *_uv, std::vector<float> *_uvImg, std::vector<int> *_triIdx, int idx);
		void GetAtlasInfoi_UVAtlas_mask(std::vector<float> *_uv, std::vector<float> *_uvImg, std::vector<int> *_triIdx, int idx);
		void GetNumber(uint *nT, uint *nV);
		void GetNumber(uint *nI, uint *w, uint *h);

		void SetSelector(Selector* selector);

		void LoadModel4D(Mapper4D* mapper4D, bool use_keyframe = true);

		void Model4DLoadandMultiUpdate(Mapper4D * mapper4D, string streamPath);
		void Model4DLoadandSelectUpdate(Mapper4D* mapper4D);
		void Model4DLoadandMultiUpdate(Mapper4D* mapper4D);
		void Model4DLoadandMultiUpdate_key(Mapper4D* mapper4D);
		void Model4DLoadandMultiUpdate_all(Mapper4D* mapper4D);
		void Model4DLoadandMultiRefine_all(Mapper4D* mapper4D);
		void Model4DLoadandMultiUpdate_key_refine(Mapper4D* mapper4D);
		void MultiUpdate_toAtlas();
		void Model4DLoadandSelectEachUpdate(Mapper4D* mapper4D);
		
		string opt_mode;

	private:

		std::vector<float2>* atlas_coord;
		std::vector<float3>* img_coord;
		std::vector<float3>* img_coord_mask;
		std::vector<int>* atlas_tri_idx;
		std::vector<float3>* All_img_coord;
		uint all_imgNum;

		StopWatchInterface *timer = NULL;

		dim3 gridSize;
		dim3 blockSize;

		int nVertex;
		int nTriangle;
		int nImage;
		int preIter = 0;

		//cv::Mat4b *rgbImages;
		//cv::Mat1w *depthImages;
		cv::Mat4b *meshImages;

		vector<hostVertex> hVertices;
		vector<hostTriangle> hTriangles;

		vector<vector<hostVertex>> time_hVertices;

		uchar *hImages;
		uchar *hImages_mask;
		//uchar4 *hCImages4;
		uchar *hAtlas;
		short *hUg;
		short *hVg;
		std::vector<std::vector<cv::Mat>> hUg_vec;
		std::vector<std::vector<cv::Mat>> hVg_vec;
		std::vector<std::vector<cv::Mat>> blur_vec;
		
		vector<vector<unsigned char>> h_samcol;
		float *hEnergy;

		deviceVertex *ddVertices;
		deviceTriangle *ddTriangles;
		deviceMemHolderV dmhVertices;
		deviceMemHolderT dmhTriangles;
		uchar *dImages;
		uchar *dImages_original;
		uchar *dImages_mask;
		//uchar *dCImages;
		short *dUg;
		short *dVg;
		float *dEnergy;

		vector<vector<float2>> propVec;

		void initCuda(int width, int height, int nVertex, int nTriangle, int nImage)
		{
			checkCudaErrors(cudaMalloc((void**)&dImages, sizeof(uchar)*height*width*nImage));
			//checkCudaErrors(cudaMalloc((void**)&dCImages, sizeof(uchar)*height*width*nImage * 3));
			checkCudaErrors(cudaMalloc((void**)&dUg, sizeof(short)*height*width*nImage));
			checkCudaErrors(cudaMalloc((void**)&dVg, sizeof(short)*height*width*nImage));
			checkCudaErrors(cudaMalloc((void**)&ddVertices, sizeof(deviceVertex)*nVertex));
			checkCudaErrors(cudaMalloc((void**)&ddTriangles, sizeof(deviceTriangle)*nTriangle));
			checkCudaErrors(cudaMalloc((void**)&dEnergy, sizeof(float)*nVertex));
			sdkCreateTimer(&timer);
		}
		void freeCuda() {
			sdkDeleteTimer(&timer);
			checkCudaErrors(cudaFree(dImages));
			//checkCudaErrors(cudaFree(dCImages));
			checkCudaErrors(cudaFree(dUg));
			checkCudaErrors(cudaFree(dVg));
			checkCudaErrors(cudaFree(ddVertices));
			checkCudaErrors(cudaFree(ddTriangles));
			checkCudaErrors(cudaFree(dEnergy));
		}

		void initCuda_onetime(int width, int height, int nImage)
		{
			checkCudaErrors(cudaMalloc((void**)&dImages_original, sizeof(uchar) * height * width * nImage));
			checkCudaErrors(cudaMalloc((void**)&dImages_mask, sizeof(uchar) * height * width * nImage));
		}

		void freeCuda_onetime() {
			checkCudaErrors(cudaFree(dImages_original));
			checkCudaErrors(cudaFree(dImages_mask));
		}
		//// �ٲܰ�
		/*
		void WriteTriangle(Triangle_Save &_T, std::ofstream &fout) {


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
		void WriteVertex(Vertex_Save &_V, std::ofstream &fout) {
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


};

}

#endif OPTIMIZER_H 