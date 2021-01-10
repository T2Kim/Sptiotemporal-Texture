#include "Optimizer.cuh"

using namespace TexMap;

__device__ float sampling_weights[3][SAMNUM] = { 0.5, 0.25, 0.25, 0.8, 0.1, 0.1, 0.5, 0.0, 0.5, 0.2, 0.6, 0.2, 0.1, 0.5, 0.4,
											     0.25, 0.5, 0.25, 0.1, 0.8, 0.1, 0.5, 0.5, 0.0, 0.2, 0.2, 0.6, 0.5, 0.4, 0.1,
											     0.25, 0.25, 0.5, 0.1, 0.1, 0.8, 0.0, 0.5, 0.5, 0.6, 0.2, 0.2, 0.4, 0.1, 0.5 };
__device__ float EDGE_WEIGHT = (0.4 * 255);
__device__ float EDGE_WEIGHT_TO_ATLAS = (0.2 * 255);
	
void deviceMemHolderT::push_back(hostTriangle &hT) {
	int i = _Img_t.size();
	int w = _Weight_t.size();
	int b = _Bound_t.size();
	int l = _Label_t.size();
	_Img_t.resize(_Img_t.size() + hT._Img.size());
	_Weight_t.resize(_Weight_t.size() + hT._Img_Weight.size());
	_Bound_t.resize(_Bound_t.size() + hT._Bound.size());
	_Label_t.resize(_Label_t.size() + hT._Label.size());
	thrust::copy(hT._Img.begin(), hT._Img.end(), _Img_t.begin() + i);
	thrust::copy(hT._Img_Weight.begin(), hT._Img_Weight.end(), _Weight_t.begin() + w);
	thrust::copy(hT._Bound.begin(), hT._Bound.end(), _Bound_t.begin() + b);
	thrust::copy(hT._Label.begin(), hT._Label.end(), _Label_t.begin() + l);
	//for (auto imIdx : hT._Img) _Img_t[i++] = imIdx;


	//for (auto imIdx : hT._Img) _Img_t.push_back(imIdx);
	_SamCol_t.resize(_SamCol_t.size() + SAMNUM);
}

void deviceMemHolderT::update_bound(vector<bool>& boundary, vector<bool>& label) {
	_Bound = NULL;
	_Bound_t.clear();
	_Bound_t.resize(boundary.size());
	thrust::copy(boundary.begin(), boundary.end(), _Bound_t.begin());
	_Bound = thrust::raw_pointer_cast(&_Bound_t[0]);

	_Label = NULL;
	_Label_t.clear();
	_Label_t.resize(label.size());
	thrust::copy(label.begin(), label.end(), _Label_t.begin());
	_Label = thrust::raw_pointer_cast(&_Label_t[0]);
}

void deviceMemHolderT::update_bound(bool value) {
	int b_size = _Bound_t.size();
	_Bound = NULL;
	_Bound_t.clear();
	_Bound_t.resize(b_size, value);
	_Bound = thrust::raw_pointer_cast(&_Bound_t[0]);

	int l_size = _Label_t.size();
	_Label = NULL;
	_Label_t.clear();
	_Label_t.resize(l_size, value);
	_Label = thrust::raw_pointer_cast(&_Label_t[0]);
}

void deviceMemHolderV::push_back(hostVertex &hV) {
	int t = _Triangles_t.size();
	int i = _Img_t.size();
	int it = _Img_Tex_t.size();
	_Triangles_t.resize(_Triangles_t.size() + hV._Triangles.size());
	_Img_t.resize(_Img_t.size() + hV._Img.size());
	_Img_Tex_t.resize(_Img_Tex_t.size() + hV._Img_Tex.size());
	thrust::copy(hV._Triangles.begin(), hV._Triangles.end(), _Triangles_t.begin() + t);
	thrust::copy(hV._Img.begin(), hV._Img.end(), _Img_t.begin() + i);
	thrust::copy(hV._Img_Tex.begin(), hV._Img_Tex.end(), _Img_Tex_t.begin() + it);

	_Edge_Init_t.resize(_Edge_Init_t.size() + hV._Triangles.size() * hV._Img.size());
}

void deviceMemHolderV::update_texcoord(vector<float2>& img_tex) {
		_Img_Tex = NULL;
		_Img_Tex_t.clear();
		_Img_Tex_t.resize(img_tex.size());
		thrust::copy(img_tex.begin(), img_tex.end(), _Img_Tex_t.begin());
		_Img_Tex = thrust::raw_pointer_cast(&_Img_Tex_t[0]);
	}

inline __device__ __host__ float2 sampling(float2 a, float2 b, float2 c, float alpha, float beta, float gamma)
{
	return alpha*a + beta*b + gamma*c;
}

__device__ __host__ void calcJTJ(float* out, float *J, float rows)
{
	/*
	M is samples x 2 (jacobian matrix)
	M^T * M
	*/
	out[0] = 0;
	out[1] = 0;
	out[2] = 0;
	out[3] = 0;
	for (int i = 0; i < rows * 2; i += 2) {
		out[0] += J[i] * J[i];
		out[1] += J[i] * J[i + 1];
		out[3] += J[i + 1] * J[i + 1];
	}
	out[2] = out[1];
}

__device__ __host__ void calcJTF(float* out, float *J, float *F, float rows)
{
	out[0] = 0;
	out[1] = 0;
	for (int i = 0; i < rows; i++) {
		out[0] += J[i * 2] * F[i];
		out[1] += J[i * 2 + 1] * F[i];
	}
}

__device__ __host__ void calcInv2x2(float *out, float *in)
{
	float mult = 1.0f / (in[0] * in[3] - in[1] * in[2]);
	out[0] = in[3] * mult;
	out[1] = -in[1] * mult;
	out[2] = -in[2] * mult;
	out[3] = in[0] * mult;
}

__device__ __host__ float linearInterpolate(uchar a, uchar b, uchar c, uchar d, float ddx, float ddy) {
	return(a * (1 - ddx) * (1 - ddy) + b * (1 + ddx) * (1 - ddy) + c * (1 - ddx) * (1 + ddy) + d * (1 + ddx) * (1 + ddy)) / 4.0;
}

__device__ __host__ float linearInterpolate(short a, short b, short c, short d, float ddx, float ddy) {
	return(a * (1 - ddx) * (1 - ddy) + b * (1 + ddx) * (1 - ddy) + c * (1 - ddx) * (1 + ddy) + d * (1 + ddx) * (1 + ddy)) / 4.0;
}

__device__ __host__ float gaussianInterpolate(float a, float b, float c, float d, float ddx, float ddy) {
	return expf(-(a*ddx*ddx + (b + c)*ddx*ddy + d*ddy*ddy));
}

__global__ void update_initial_edge(deviceVertex* vertices, deviceTriangle* triangles, int* dmh_ImgV, float2 *dmh_Img_Tex, float2 *dmh_Edge_Init, int* dmh_ImgT, int *dmh_Tri, unsigned char *images, int w, int h, int nVertex)
{
	int vidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (vidx >= nVertex) return;
	deviceVertex vCurrent = vertices[vidx];
	deviceVertex v1, v2, v3;
	int tmpImgNum = vCurrent._Img_Num;
	for (int i = 0; i < tmpImgNum; i++) {
		int imgidx = dmh_ImgV[vCurrent._imgOffset + i];
		int tmpTriNum = vCurrent._Triangles_Num;
		for (int j = 0; j < tmpTriNum; j++) {
			deviceTriangle tCurrent = triangles[dmh_Tri[vCurrent._triOffset + j]];
			if (tCurrent.isCoveredBy(imgidx, &dmh_ImgT[tCurrent._imgOffset])) {
				int imgidx1, imgidx2, imgidx3;
				float2 uv1, uv2, uv3;

				// get other vertices
				if (tCurrent._Vertices[0] == vidx) {
					v1 = vCurrent;
					v2 = vertices[tCurrent._Vertices[1]];
					v3 = vertices[tCurrent._Vertices[2]];
					imgidx1 = v1.getImgIdx(imgidx, &dmh_ImgV[v1._imgOffset]);
					imgidx2 = v2.getImgIdx(imgidx, &dmh_ImgV[v2._imgOffset]);
					imgidx3 = v3.getImgIdx(imgidx, &dmh_ImgV[v3._imgOffset]);
					uv1 = dmh_Img_Tex[v1._imgOffset + imgidx1];
					uv2 = dmh_Img_Tex[v2._imgOffset + imgidx2];
					uv3 = dmh_Img_Tex[v3._imgOffset + imgidx3];
					float len1 = sqrt(dot(uv1 - uv2, uv1 - uv2));
					float len2 = sqrt(dot(uv1 - uv3, uv1 - uv3));
					dmh_Edge_Init[vCurrent._edgeOffset + i * tmpTriNum + j] = make_float2(len1, len2);
				}
				else if (tCurrent._Vertices[1] == vidx) {
					v1 = vertices[tCurrent._Vertices[0]];
					v2 = vCurrent;
					v3 = vertices[tCurrent._Vertices[2]];
					imgidx1 = v1.getImgIdx(imgidx, &dmh_ImgV[v1._imgOffset]);
					imgidx2 = v2.getImgIdx(imgidx, &dmh_ImgV[v2._imgOffset]);
					imgidx3 = v3.getImgIdx(imgidx, &dmh_ImgV[v3._imgOffset]);
					uv1 = dmh_Img_Tex[v1._imgOffset + imgidx1];
					uv2 = dmh_Img_Tex[v2._imgOffset + imgidx2];
					uv3 = dmh_Img_Tex[v3._imgOffset + imgidx3];
					float len1 = sqrt(dot(uv2 - uv1, uv2 - uv1));
					float len2 = sqrt(dot(uv2 - uv3, uv2 - uv3));
					dmh_Edge_Init[vCurrent._edgeOffset + i * tmpTriNum + j] = make_float2(len1, len2);
				}
				else if (tCurrent._Vertices[2] == vidx) {
					v1 = vertices[tCurrent._Vertices[0]];
					v2 = vertices[tCurrent._Vertices[1]];
					v3 = vCurrent;
					imgidx1 = v1.getImgIdx(imgidx, &dmh_ImgV[v1._imgOffset]);
					imgidx2 = v2.getImgIdx(imgidx, &dmh_ImgV[v2._imgOffset]);
					imgidx3 = v3.getImgIdx(imgidx, &dmh_ImgV[v3._imgOffset]);
					uv1 = dmh_Img_Tex[v1._imgOffset + imgidx1];
					uv2 = dmh_Img_Tex[v2._imgOffset + imgidx2];
					uv3 = dmh_Img_Tex[v3._imgOffset + imgidx3];
					float len1 = sqrt(dot(uv3 - uv1, uv3 - uv1));
					float len2 = sqrt(dot(uv3 - uv2, uv3 - uv2));
					dmh_Edge_Init[vCurrent._edgeOffset + i * tmpTriNum + j] = make_float2(len1, len2);
				}
				else {
					return;
				}
			}
		}
	}
}

__global__ void update_texture_coordinate(deviceVertex* vertices, deviceTriangle* triangles, int* dmh_ImgV, float2 *dmh_Img_Tex, float2 *dmh_Edge_Init, int* dmh_ImgT, int *dmh_Tri, unsigned char *dmh_samcol, unsigned char *images, short *ug, short * vg, int w, int h, int nVertex)
{
	int vidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (vidx >= nVertex) return;
	deviceVertex vCurrent = vertices[vidx];
	int tmpImgNum = vCurrent._Img_Num;
	int tmpTriNum = vCurrent._Triangles_Num;
	float F[(SAMNUM + 2)*MAXTRI]; // sampling points, edge regularize, position regularize
	float J[(SAMNUM * 2 + 4)*MAXTRI];
	float2 samples[SAMNUM];

	int size = w*h;

	deviceTriangle triangles_current;
	deviceVertex vertices_current[3];

	for (int i = 0; i < tmpImgNum; i++) {
		int imgidx = dmh_ImgV[vCurrent._imgOffset + i];

		int currentSample = 0;
		for (int j = 0; j < tmpTriNum && j < MAXTRI; j++) {
			triangles_current = triangles[dmh_Tri[vCurrent._triOffset + j]];
			vertices_current[0] = vertices[triangles_current._Vertices[0]];
			vertices_current[1] = vertices[triangles_current._Vertices[1]];
			vertices_current[2] = vertices[triangles_current._Vertices[2]];

			float *sampling_weights_current;
			if (triangles_current.isCoveredBy(imgidx, &dmh_ImgT[triangles_current._imgOffset])) {
				float2 uv1, uv2, uv3;
				float2 edge_length;
				edge_length = dmh_Edge_Init[vCurrent._edgeOffset + i * tmpTriNum + j];
				// get other vertices
				if (triangles_current._Vertices[0] == vidx) {
					sampling_weights_current = sampling_weights[0];
					uv1 = dmh_Img_Tex[vCurrent._imgOffset + i];
					uv2 = dmh_Img_Tex[vertices_current[1]._imgOffset + vertices_current[1].getImgIdx(imgidx, &dmh_ImgV[vertices_current[1]._imgOffset])]; 
					uv3 = dmh_Img_Tex[vertices_current[2]._imgOffset + vertices_current[2].getImgIdx(imgidx, &dmh_ImgV[vertices_current[2]._imgOffset])]; 
					float2 diff1 = uv1 - uv2;
					float2 diff2 = uv1 - uv3;
					float dist = sqrt(dot(diff1, diff1));
					F[currentSample] = (dist - edge_length.x) * EDGE_WEIGHT;
					J[currentSample * 2] = (diff1).x / (dist + 0.0001) * EDGE_WEIGHT;
					J[currentSample * 2 + 1] = (diff1).y / (dist + 0.0001) * EDGE_WEIGHT;
					currentSample++;
					dist = sqrt(dot(diff2, diff2));
					F[currentSample] = (dist - edge_length.y)*EDGE_WEIGHT;
					J[currentSample * 2] = (diff2).x / (dist + 0.0001)*EDGE_WEIGHT;
					J[currentSample * 2 + 1] = (diff2).y / (dist + 0.0001)*EDGE_WEIGHT;
					currentSample++;
				}
				else if (triangles_current._Vertices[1] == vidx) {
					sampling_weights_current = sampling_weights[1];
					uv1 = dmh_Img_Tex[vertices_current[0]._imgOffset + vertices_current[0].getImgIdx(imgidx, &dmh_ImgV[vertices_current[0]._imgOffset])];
					uv2 = dmh_Img_Tex[vCurrent._imgOffset + i];
					uv3 = dmh_Img_Tex[vertices_current[2]._imgOffset + vertices_current[2].getImgIdx(imgidx, &dmh_ImgV[vertices_current[2]._imgOffset])];
					float2 diff1 = uv2 - uv1;
					float2 diff2 = uv2 - uv3;
					float dist = sqrt(dot(diff1, diff1));
					F[currentSample] = (dist - edge_length.x)*EDGE_WEIGHT;
					J[currentSample * 2] = (diff1).x / (dist + 0.0001)*EDGE_WEIGHT;
					J[currentSample * 2 + 1] = (diff1).y / (dist + 0.0001)*EDGE_WEIGHT;
					currentSample++;
					dist = sqrt(dot(diff2, diff2));
					F[currentSample] = (dist - edge_length.y)*EDGE_WEIGHT;
					J[currentSample * 2] = (diff2).x / (dist + 0.0001)*EDGE_WEIGHT;
					J[currentSample * 2 + 1] = (diff2).y / (dist + 0.0001)*EDGE_WEIGHT;
					currentSample++;
				}
				else {
					sampling_weights_current = sampling_weights[2];
					uv1 = dmh_Img_Tex[vertices_current[0]._imgOffset + vertices_current[0].getImgIdx(imgidx, &dmh_ImgV[vertices_current[0]._imgOffset])];
					uv2 = dmh_Img_Tex[vertices_current[1]._imgOffset + vertices_current[1].getImgIdx(imgidx, &dmh_ImgV[vertices_current[1]._imgOffset])];
					uv3 = dmh_Img_Tex[vCurrent._imgOffset + i];
					float2 diff1 = uv3 - uv1;
					float2 diff2 = uv3 - uv2;
					float dist = sqrt(dot(diff1, diff1));
					F[currentSample] = (dist - edge_length.x)*EDGE_WEIGHT;
					J[currentSample * 2] = (diff1).x / (dist + 0.0001)*EDGE_WEIGHT;
					J[currentSample * 2 + 1] = (diff1).y / (dist + 0.0001)*EDGE_WEIGHT;
					currentSample++;
					dist = sqrt(dot(diff2, diff2));
					F[currentSample] = (dist - edge_length.y)*EDGE_WEIGHT;
					J[currentSample * 2] = (diff2).x / (dist + 0.0001)*EDGE_WEIGHT;
					J[currentSample * 2 + 1] = (diff2).y / (dist + 0.0001)*EDGE_WEIGHT;
					currentSample++;
				}

				// sampling the uv coordinates
				for (int s = 0; s < SAMNUM; s++) {
					samples[s] = sampling(uv1, uv2, uv3, sampling_weights[0][s], sampling_weights[1][s], sampling_weights[2][s]);
				}

				// append the matrix F and J
				for (int s = 0; s < SAMNUM; s++) {
					// get intensity from the image
					int2 pivot = make_int2(samples[s].x + 0.5, samples[s].y + 0.5);

					float intensity = linearInterpolate(images[imgidx * size + (pivot.y - 1) * w + pivot.x - 1],
														images[imgidx * size + (pivot.y - 1) * w + pivot.x],
														images[imgidx * size + (pivot.y) * w + pivot.x - 1],
														images[imgidx * size + (pivot.y) * w + pivot.x],
														samples[s].x - pivot.x, samples[s].y - pivot.y);
					
					//float intensity = images[imgidx*size + (int)samples[s].y*w + (int)samples[s].x];
					float proxy = dmh_samcol[dmh_Tri[vCurrent._triOffset + j] * SAMNUM + s];
					F[currentSample] = (intensity - proxy);
					/*J[currentSample * 2] = (float)ug[imgidx*size + samples[s].y*w + samples[s].x] * sampling_weights_current[s];
					J[currentSample * 2 + 1] = (float)vg[imgidx*size + samples[s].y*w + samples[s].x] * sampling_weights_current[s];*/
					J[currentSample * 2] = linearInterpolate(ug[imgidx * size + (pivot.y - 1) * w + pivot.x - 1],
						ug[imgidx * size + (pivot.y - 1) * w + pivot.x],
						ug[imgidx * size + (pivot.y) * w + pivot.x - 1],
						ug[imgidx * size + (pivot.y) * w + pivot.x],
						samples[s].x - pivot.x, samples[s].y - pivot.y) *sampling_weights_current[s];
					J[currentSample * 2+1] = linearInterpolate(vg[imgidx * size + (pivot.y - 1) * w + pivot.x - 1],
						vg[imgidx * size + (pivot.y - 1) * w + pivot.x],
						vg[imgidx * size + (pivot.y) * w + pivot.x - 1],
						vg[imgidx * size + (pivot.y) * w + pivot.x],
						samples[s].x - pivot.x, samples[s].y - pivot.y) *sampling_weights_current[s];
					currentSample++;
				}
			}
		}
		if (currentSample > 0) {
			float JTJ[4]; float JTF[2];
			float JTJinv[4];
			calcJTJ(JTJ, J, currentSample);
			calcJTF(JTF, J, F, currentSample);
			if (abs(JTJ[0] * JTJ[3] - JTJ[1] * JTJ[2]) < 0.0001) {
				JTJ[0] = JTJ[0] + 0.0001;
				JTJ[3] = JTJ[3] + 0.0001;
			}
			calcInv2x2(JTJinv, JTJ);
			float2 step = make_float2(JTJinv[0] * JTF[0] + JTJinv[1] * JTF[1], JTJinv[2] * JTF[0] + JTJinv[3] * JTF[1]);
			if (isnan(step.x) || isnan(step.y))
				step = make_float2(0, 0);
			step = clamp(step, -5.0, 5.0);
			dmh_Img_Tex[vertices[vidx]._imgOffset + i] = make_float2(clamp(dmh_Img_Tex[vertices[vidx]._imgOffset + i].x - step.x, 0.0, w - 1.0), clamp(dmh_Img_Tex[vertices[vidx]._imgOffset + i].y - step.y, 0.0, h - 1.0));
		}
	}
}

__global__ void calc_energy(float *energy, deviceVertex* vertices, deviceTriangle* triangles, int* dmh_ImgV, float2 *dmh_Img_Tex, int* dmh_ImgT, int *dmh_Tri, unsigned char *dmh_samcol, unsigned char *images, short *ug, short * vg, int w, int h, int nVertex)
{
	int vidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (vidx >= nVertex) return;
	deviceVertex vCurrent = vertices[vidx];
	int tmpImgNum = vCurrent._Img_Num;
	int tmpTriNum = vCurrent._Triangles_Num;
	//float *F = (float*)malloc(SAMNUM * tmpTriNum);
	//float *F = new float[(SAMNUM) * tmpTriNum];
	float F[SAMNUM*MAXTRI]; // sampling points, edge regularize, position regularize
	int2 samples[SAMNUM];

	float vertex_energy = 0;
	int size = w*h;
	int imgcounted = 0;

	deviceTriangle triangles_current;
	deviceVertex vertices_current[3];

	for (int i = 0; i < tmpImgNum ; i++) {
		int imgidx = dmh_ImgV[vCurrent._imgOffset + i];
		int currentSample = 0;
		for (int j = 0; j < tmpTriNum && j < MAXTRI; j++) {
			triangles_current = triangles[dmh_Tri[vCurrent._triOffset + j]];
			vertices_current[0] = vertices[triangles_current._Vertices[0]];
			vertices_current[1] = vertices[triangles_current._Vertices[1]];
			vertices_current[2] = vertices[triangles_current._Vertices[2]];

			float *sampling_weights_current;
			if (triangles_current.isCoveredBy(imgidx, &dmh_ImgT[triangles_current._imgOffset])) {
				int imgidx1, imgidx2, imgidx3;
				float2 uv1, uv2, uv3;
				// get other vertices
				if (triangles_current._Vertices[0] == vidx) {
					sampling_weights_current = sampling_weights[0];
					uv1 = dmh_Img_Tex[vCurrent._imgOffset + i];
					uv2 = dmh_Img_Tex[vertices_current[1]._imgOffset + vertices_current[1].getImgIdx(imgidx, &dmh_ImgV[vertices_current[1]._imgOffset])];
					uv3 = dmh_Img_Tex[vertices_current[2]._imgOffset + vertices_current[2].getImgIdx(imgidx, &dmh_ImgV[vertices_current[2]._imgOffset])];
				}
				else if (triangles_current._Vertices[1] == vidx) {
					sampling_weights_current = sampling_weights[1];
					uv1 = dmh_Img_Tex[vertices_current[0]._imgOffset + vertices_current[0].getImgIdx(imgidx, &dmh_ImgV[vertices_current[0]._imgOffset])];
					uv2 = dmh_Img_Tex[vCurrent._imgOffset + i];
					uv3 = dmh_Img_Tex[vertices_current[2]._imgOffset + vertices_current[2].getImgIdx(imgidx, &dmh_ImgV[vertices_current[2]._imgOffset])];
				}
				else if (triangles_current._Vertices[2] == vidx) {
					sampling_weights_current = sampling_weights[2];
					uv1 = dmh_Img_Tex[vertices_current[0]._imgOffset + vertices_current[0].getImgIdx(imgidx, &dmh_ImgV[vertices_current[0]._imgOffset])];
					uv2 = dmh_Img_Tex[vertices_current[1]._imgOffset + vertices_current[1].getImgIdx(imgidx, &dmh_ImgV[vertices_current[1]._imgOffset])];
					uv3 = dmh_Img_Tex[vCurrent._imgOffset + i];
				}
				else {
					continue;
				}

				// sampling the uv coordinates
				for (int s = 0; s < SAMNUM; s++) {
					samples[s] = make_int2(sampling(uv1, uv2, uv3, sampling_weights[0][s], sampling_weights[1][s], sampling_weights[2][s]));
					samples[s].x = min(w - 1, max(0, samples[s].x));
					samples[s].y = min(h - 1, max(0, samples[s].y));
				}

				// append the matrix F and J
				for (int s = 0; s < SAMNUM; s++) {
					int2 pivot = make_int2(samples[s].x + 0.5, samples[s].y + 0.5);

					float intensity = linearInterpolate(images[imgidx * size + (pivot.y - 1) * w + pivot.x - 1],
						images[imgidx * size + (pivot.y - 1) * w + pivot.x],
						images[imgidx * size + (pivot.y) * w + pivot.x - 1],
						images[imgidx * size + (pivot.y) * w + pivot.x],
						samples[s].x - pivot.x, samples[s].y - pivot.y);
					// get intensity from the image
					//float intensity = images[imgidx*size + samples[s].y*w + samples[s].x];
					float proxy = dmh_samcol[dmh_Tri[vCurrent._triOffset + j] * SAMNUM + s];
					F[currentSample] = (intensity - proxy) / 255.0f;
					currentSample++;
				}
			}
		}
		if (currentSample > 0) {
			for (int k = 0; k < currentSample; k++)
				vertex_energy += F[k] * F[k];
			imgcounted++;
		}
	}

	energy[vidx] = vertex_energy;
}

__global__ void update_proxy_color(deviceVertex* vertices, deviceTriangle* triangles, int* dmh_ImgV, float2 *dmh_Img_Tex, int* dmh_ImgT, unsigned char *dmh_samcol, float *dmh_weight, unsigned char *images, int w, int h, int nTriangle)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (tidx >= nTriangle) return;
	deviceTriangle tCurrent = triangles[tidx];
	int tmpImgNum = tCurrent._Img_Num;
	deviceVertex *v1 = &vertices[tCurrent._Vertices[0]];
	deviceVertex *v2 = &vertices[tCurrent._Vertices[1]];
	deviceVertex *v3 = &vertices[tCurrent._Vertices[2]];
	int size = w * h;
	float samcol[SAMNUM];
	for (int i = 0; i < SAMNUM; i++) { samcol[i] = 0; }
	for (int i = 0; i < tmpImgNum; i++) { // for each image projects to this triangle
		int imgidx = dmh_ImgT[tCurrent._imgOffset + i];

		int imgidx1 = v1->getImgIdx(imgidx, &dmh_ImgV[v1->_imgOffset]);
		int imgidx2 = v2->getImgIdx(imgidx, &dmh_ImgV[v2->_imgOffset]);
		int imgidx3 = v3->getImgIdx(imgidx, &dmh_ImgV[v3->_imgOffset]);
		float2 uv1 = dmh_Img_Tex[v1->_imgOffset + imgidx1];
		float2 uv2 = dmh_Img_Tex[v2->_imgOffset + imgidx2];
		float2 uv3 = dmh_Img_Tex[v3->_imgOffset + imgidx3];

		float2 samples[SAMNUM];
		// sampling the uv coordinates
		for (int s = 0; s < SAMNUM; s++)
			samples[s] = sampling(uv1, uv2, uv3, sampling_weights[0][s], sampling_weights[1][s], sampling_weights[2][s]);
		for (int s = 0; s < SAMNUM; s++) {
			// get intensity from the image
			
			//samcol[s] += images[imgidx*w*h + samples[s].y*w + samples[s].x];// *tCurrent._Img_Weight[i];
			int2 pivot = make_int2(samples[s].x + 0.5, samples[s].y + 0.5);
			samcol[s] += linearInterpolate(images[imgidx * size + (pivot.y - 1) * w + pivot.x - 1],
											images[imgidx * size + (pivot.y - 1) * w + pivot.x],
											images[imgidx * size + (pivot.y) * w + pivot.x - 1],
											images[imgidx * size + (pivot.y) * w + pivot.x],
											samples[s].x - pivot.x, samples[s].y - pivot.y) * tCurrent.weightedBy(imgidx, &dmh_ImgT[tCurrent._imgOffset], &dmh_weight[tCurrent._imgOffset]);;
		}
	}
	for (int s = 0; s < SAMNUM; s++)
		//dmh_samcol[tidx * SAMNUM + s] = uchar(clamp(samcol[s] / tmpImgNum, 0.f, 255.f));
		dmh_samcol[tidx * SAMNUM + s] = uchar(clamp(samcol[s], 0.f, 255.f));
}

__global__ void update_texture_coordinate_sel(deviceVertex* vertices, deviceTriangle* triangles, int* dmh_ImgV, float2* dmh_Img_Tex, float2* dmh_Edge_Init, int* dmh_ImgT, bool* dmh_selT, int* dmh_Tri, unsigned char* dmh_samcol, unsigned char* images, short* ug, short* vg, int w, int h, int nVertex)
{
	int vidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (vidx >= nVertex) return;
	deviceVertex vCurrent = vertices[vidx];
	int tmpImgNum = vCurrent._Img_Num;
	int tmpTriNum = vCurrent._Triangles_Num;
	float F[(SAMNUM + 2) * MAXTRI]; // sampling points, edge regularize, position regularize
	float J[(SAMNUM * 2 + 4) * MAXTRI];
	float2 samples[SAMNUM];

	int size = w * h;

	deviceTriangle triangles_current;
	deviceVertex vertices_current[3];

	for (int i = 0; i < tmpImgNum; i++) {
		int imgidx = dmh_ImgV[vCurrent._imgOffset + i];

		int currentSample = 0;
		for (int j = 0; j < tmpTriNum && j < MAXTRI; j++) {
			triangles_current = triangles[dmh_Tri[vCurrent._triOffset + j]];
			vertices_current[0] = vertices[triangles_current._Vertices[0]];
			vertices_current[1] = vertices[triangles_current._Vertices[1]];
			vertices_current[2] = vertices[triangles_current._Vertices[2]];

			float* sampling_weights_current;
			if (triangles_current.isCoveredBy(imgidx, &dmh_ImgT[triangles_current._imgOffset])) {

				float2 uv1, uv2, uv3;
				float2 edge_length;
				edge_length = dmh_Edge_Init[vCurrent._edgeOffset + i * tmpTriNum + j];
				// get other vertices
				if (triangles_current._Vertices[0] == vidx) {
					sampling_weights_current = sampling_weights[0];
					uv1 = dmh_Img_Tex[vCurrent._imgOffset + i];
					uv2 = dmh_Img_Tex[vertices_current[1]._imgOffset + vertices_current[1].getImgIdx(imgidx, &dmh_ImgV[vertices_current[1]._imgOffset])];
					uv3 = dmh_Img_Tex[vertices_current[2]._imgOffset + vertices_current[2].getImgIdx(imgidx, &dmh_ImgV[vertices_current[2]._imgOffset])];
					float2 diff1 = uv1 - uv2;
					float2 diff2 = uv1 - uv3;
					float dist = sqrt(dot(diff1, diff1));
					F[currentSample] = (dist - edge_length.x) * EDGE_WEIGHT;
					J[currentSample * 2] = (diff1).x / (dist + 0.0001) * EDGE_WEIGHT;
					J[currentSample * 2 + 1] = (diff1).y / (dist + 0.0001) * EDGE_WEIGHT;
					currentSample++;
					dist = sqrt(dot(diff2, diff2));
					F[currentSample] = (dist - edge_length.y) * EDGE_WEIGHT;
					J[currentSample * 2] = (diff2).x / (dist + 0.0001) * EDGE_WEIGHT;
					J[currentSample * 2 + 1] = (diff2).y / (dist + 0.0001) * EDGE_WEIGHT;
					currentSample++;
				}
				else if (triangles_current._Vertices[1] == vidx) {
					sampling_weights_current = sampling_weights[1];
					uv1 = dmh_Img_Tex[vertices_current[0]._imgOffset + vertices_current[0].getImgIdx(imgidx, &dmh_ImgV[vertices_current[0]._imgOffset])];
					uv2 = dmh_Img_Tex[vCurrent._imgOffset + i];
					uv3 = dmh_Img_Tex[vertices_current[2]._imgOffset + vertices_current[2].getImgIdx(imgidx, &dmh_ImgV[vertices_current[2]._imgOffset])];
					float2 diff1 = uv2 - uv1;
					float2 diff2 = uv2 - uv3;
					float dist = sqrt(dot(diff1, diff1));
					F[currentSample] = (dist - edge_length.x) * EDGE_WEIGHT;
					J[currentSample * 2] = (diff1).x / (dist + 0.0001) * EDGE_WEIGHT;
					J[currentSample * 2 + 1] = (diff1).y / (dist + 0.0001) * EDGE_WEIGHT;
					currentSample++;
					dist = sqrt(dot(diff2, diff2));
					F[currentSample] = (dist - edge_length.y) * EDGE_WEIGHT;
					J[currentSample * 2] = (diff2).x / (dist + 0.0001) * EDGE_WEIGHT;
					J[currentSample * 2 + 1] = (diff2).y / (dist + 0.0001) * EDGE_WEIGHT;
					currentSample++;
				}
				else {
					sampling_weights_current = sampling_weights[2];
					uv1 = dmh_Img_Tex[vertices_current[0]._imgOffset + vertices_current[0].getImgIdx(imgidx, &dmh_ImgV[vertices_current[0]._imgOffset])];
					uv2 = dmh_Img_Tex[vertices_current[1]._imgOffset + vertices_current[1].getImgIdx(imgidx, &dmh_ImgV[vertices_current[1]._imgOffset])];
					uv3 = dmh_Img_Tex[vCurrent._imgOffset + i];
					float2 diff1 = uv3 - uv1;
					float2 diff2 = uv3 - uv2;
					float dist = sqrt(dot(diff1, diff1));
					F[currentSample] = (dist - edge_length.x) * EDGE_WEIGHT;
					J[currentSample * 2] = (diff1).x / (dist + 0.0001) * EDGE_WEIGHT;
					J[currentSample * 2 + 1] = (diff1).y / (dist + 0.0001) * EDGE_WEIGHT;
					currentSample++;
					dist = sqrt(dot(diff2, diff2));
					F[currentSample] = (dist - edge_length.y) * EDGE_WEIGHT;
					J[currentSample * 2] = (diff2).x / (dist + 0.0001) * EDGE_WEIGHT;
					J[currentSample * 2 + 1] = (diff2).y / (dist + 0.0001) * EDGE_WEIGHT;
					currentSample++;
				}


				if (!dmh_selT[triangles_current._imgOffset + triangles_current.getImgIdx(imgidx, &dmh_ImgT[triangles_current._imgOffset])])
					continue;

				// sampling the uv coordinates
				for (int s = 0; s < SAMNUM; s++) {
					samples[s] = sampling(uv1, uv2, uv3, sampling_weights[0][s], sampling_weights[1][s], sampling_weights[2][s]);
				}

				// append the matrix F and J
				for (int s = 0; s < SAMNUM; s++) {
					// get intensity from the image
					int2 pivot = make_int2(samples[s].x + 0.5, samples[s].y + 0.5);

					float intensity = linearInterpolate(images[imgidx * size + (pivot.y - 1) * w + pivot.x - 1],
						images[imgidx * size + (pivot.y - 1) * w + pivot.x],
						images[imgidx * size + (pivot.y) * w + pivot.x - 1],
						images[imgidx * size + (pivot.y) * w + pivot.x],
						samples[s].x - pivot.x, samples[s].y - pivot.y);

					//float intensity = images[imgidx*size + (int)samples[s].y*w + (int)samples[s].x];
					float proxy = dmh_samcol[dmh_Tri[vCurrent._triOffset + j] * SAMNUM + s];
					F[currentSample] = (intensity - proxy);
					/*J[currentSample * 2] = (float)ug[imgidx*size + samples[s].y*w + samples[s].x] * sampling_weights_current[s];
					J[currentSample * 2 + 1] = (float)vg[imgidx*size + samples[s].y*w + samples[s].x] * sampling_weights_current[s];*/
					J[currentSample * 2] = linearInterpolate(ug[imgidx * size + (pivot.y - 1) * w + pivot.x - 1],
						ug[imgidx * size + (pivot.y - 1) * w + pivot.x],
						ug[imgidx * size + (pivot.y) * w + pivot.x - 1],
						ug[imgidx * size + (pivot.y) * w + pivot.x],
						samples[s].x - pivot.x, samples[s].y - pivot.y) * sampling_weights_current[s];
					J[currentSample * 2 + 1] = linearInterpolate(vg[imgidx * size + (pivot.y - 1) * w + pivot.x - 1],
						vg[imgidx * size + (pivot.y - 1) * w + pivot.x],
						vg[imgidx * size + (pivot.y) * w + pivot.x - 1],
						vg[imgidx * size + (pivot.y) * w + pivot.x],
						samples[s].x - pivot.x, samples[s].y - pivot.y) * sampling_weights_current[s];
					currentSample++;
				}
			}
		}
		if (currentSample > 0) {
			float JTJ[4]; float JTF[2];
			float JTJinv[4];
			calcJTJ(JTJ, J, currentSample);
			calcJTF(JTF, J, F, currentSample);
			if (abs(JTJ[0] * JTJ[3] - JTJ[1] * JTJ[2]) < 0.0001) {
				JTJ[0] = JTJ[0] + 0.0001;
				JTJ[3] = JTJ[3] + 0.0001;
			}
			calcInv2x2(JTJinv, JTJ);
			float2 step = make_float2(JTJinv[0] * JTF[0] + JTJinv[1] * JTF[1], JTJinv[2] * JTF[0] + JTJinv[3] * JTF[1]);
			if (isnan(step.x) || isnan(step.y))
				step = make_float2(0, 0);
			step = clamp(step, -5.0, 5.0);
			dmh_Img_Tex[vertices[vidx]._imgOffset + i] = make_float2(clamp(dmh_Img_Tex[vertices[vidx]._imgOffset + i].x - step.x, 0.0, w - 1.0), clamp(dmh_Img_Tex[vertices[vidx]._imgOffset + i].y - step.y, 0.0, h - 1.0));
		}
	}
}

__global__ void update_texture_coordinate_sel2(deviceVertex* vertices, deviceTriangle* triangles, int* dmh_ImgV, float2* dmh_Img_Tex, float2* dmh_Edge_Init, int* dmh_ImgT, bool* dmh_selT, bool* dmh_selT2, int* dmh_Tri, unsigned char* dmh_samcol, unsigned char* images, short* ug, short* vg, int w, int h, int nVertex)
{
	int vidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (vidx >= nVertex) return;
	deviceVertex vCurrent = vertices[vidx];
	int tmpImgNum = vCurrent._Img_Num;
	int tmpTriNum = vCurrent._Triangles_Num;
	float F[(SAMNUM + 2) * MAXTRI]; // sampling points, edge regularize, position regularize
	float J[(SAMNUM * 2 + 4) * MAXTRI];
	float2 samples[SAMNUM];

	int size = w * h;

	deviceTriangle triangles_current;
	deviceVertex vertices_current[3];

	for (int i = 0; i < tmpImgNum; i++) {
		int imgidx = dmh_ImgV[vCurrent._imgOffset + i];

		int currentSample = 0;
		for (int j = 0; j < tmpTriNum && j < MAXTRI; j++) {
			triangles_current = triangles[dmh_Tri[vCurrent._triOffset + j]];
			vertices_current[0] = vertices[triangles_current._Vertices[0]];
			vertices_current[1] = vertices[triangles_current._Vertices[1]];
			vertices_current[2] = vertices[triangles_current._Vertices[2]];
			float* sampling_weights_current;
			if (triangles_current.isCoveredBy(imgidx, &dmh_ImgT[triangles_current._imgOffset])) {
				if (!dmh_selT2[triangles_current._imgOffset + triangles_current.getImgIdx(imgidx, &dmh_ImgT[triangles_current._imgOffset])])
					continue;

				float2 uv1, uv2, uv3;
				float2 edge_length;
				edge_length = dmh_Edge_Init[vCurrent._edgeOffset + i * tmpTriNum + j];
				// get other vertices
				if (triangles_current._Vertices[0] == vidx) {
					sampling_weights_current = sampling_weights[0];
					uv1 = dmh_Img_Tex[vCurrent._imgOffset + i];
					uv2 = dmh_Img_Tex[vertices_current[1]._imgOffset + vertices_current[1].getImgIdx(imgidx, &dmh_ImgV[vertices_current[1]._imgOffset])];
					uv3 = dmh_Img_Tex[vertices_current[2]._imgOffset + vertices_current[2].getImgIdx(imgidx, &dmh_ImgV[vertices_current[2]._imgOffset])];
					float2 diff1 = uv1 - uv2;
					float2 diff2 = uv1 - uv3;
					float dist = sqrt(dot(diff1, diff1));
					F[currentSample] = (dist - edge_length.x) * EDGE_WEIGHT;
					J[currentSample * 2] = (diff1).x / (dist + 0.0001) * EDGE_WEIGHT;
					J[currentSample * 2 + 1] = (diff1).y / (dist + 0.0001) * EDGE_WEIGHT;
					currentSample++;
					dist = sqrt(dot(diff2, diff2));
					F[currentSample] = (dist - edge_length.y) * EDGE_WEIGHT;
					J[currentSample * 2] = (diff2).x / (dist + 0.0001) * EDGE_WEIGHT;
					J[currentSample * 2 + 1] = (diff2).y / (dist + 0.0001) * EDGE_WEIGHT;
					currentSample++;
				}
				else if (triangles_current._Vertices[1] == vidx) {
					sampling_weights_current = sampling_weights[1];
					uv1 = dmh_Img_Tex[vertices_current[0]._imgOffset + vertices_current[0].getImgIdx(imgidx, &dmh_ImgV[vertices_current[0]._imgOffset])];
					uv2 = dmh_Img_Tex[vCurrent._imgOffset + i];
					uv3 = dmh_Img_Tex[vertices_current[2]._imgOffset + vertices_current[2].getImgIdx(imgidx, &dmh_ImgV[vertices_current[2]._imgOffset])];
					float2 diff1 = uv2 - uv1;
					float2 diff2 = uv2 - uv3;
					float dist = sqrt(dot(diff1, diff1));
					F[currentSample] = (dist - edge_length.x) * EDGE_WEIGHT;
					J[currentSample * 2] = (diff1).x / (dist + 0.0001) * EDGE_WEIGHT;
					J[currentSample * 2 + 1] = (diff1).y / (dist + 0.0001) * EDGE_WEIGHT;
					currentSample++;
					dist = sqrt(dot(diff2, diff2));
					F[currentSample] = (dist - edge_length.y) * EDGE_WEIGHT;
					J[currentSample * 2] = (diff2).x / (dist + 0.0001) * EDGE_WEIGHT;
					J[currentSample * 2 + 1] = (diff2).y / (dist + 0.0001) * EDGE_WEIGHT;
					currentSample++;
				}
				else {
					sampling_weights_current = sampling_weights[2];
					uv1 = dmh_Img_Tex[vertices_current[0]._imgOffset + vertices_current[0].getImgIdx(imgidx, &dmh_ImgV[vertices_current[0]._imgOffset])];
					uv2 = dmh_Img_Tex[vertices_current[1]._imgOffset + vertices_current[1].getImgIdx(imgidx, &dmh_ImgV[vertices_current[1]._imgOffset])];
					uv3 = dmh_Img_Tex[vCurrent._imgOffset + i];
					float2 diff1 = uv3 - uv1;
					float2 diff2 = uv3 - uv2;
					float dist = sqrt(dot(diff1, diff1));
					F[currentSample] = (dist - edge_length.x) * EDGE_WEIGHT;
					J[currentSample * 2] = (diff1).x / (dist + 0.0001) * EDGE_WEIGHT;
					J[currentSample * 2 + 1] = (diff1).y / (dist + 0.0001) * EDGE_WEIGHT;
					currentSample++;
					dist = sqrt(dot(diff2, diff2));
					F[currentSample] = (dist - edge_length.y) * EDGE_WEIGHT;
					J[currentSample * 2] = (diff2).x / (dist + 0.0001) * EDGE_WEIGHT;
					J[currentSample * 2 + 1] = (diff2).y / (dist + 0.0001) * EDGE_WEIGHT;
					currentSample++;
				}


				if (!dmh_selT[triangles_current._imgOffset + triangles_current.getImgIdx(imgidx, &dmh_ImgT[triangles_current._imgOffset])])
					continue;

				// sampling the uv coordinates
				for (int s = 0; s < SAMNUM; s++) {
					samples[s] = sampling(uv1, uv2, uv3, sampling_weights[0][s], sampling_weights[1][s], sampling_weights[2][s]);
				}

				// append the matrix F and J
				for (int s = 0; s < SAMNUM; s++) {
					// get intensity from the image
					int2 pivot = make_int2(samples[s].x + 0.5, samples[s].y + 0.5);

					float intensity = linearInterpolate(images[imgidx * size + (pivot.y - 1) * w + pivot.x - 1],
						images[imgidx * size + (pivot.y - 1) * w + pivot.x],
						images[imgidx * size + (pivot.y) * w + pivot.x - 1],
						images[imgidx * size + (pivot.y) * w + pivot.x],
						samples[s].x - pivot.x, samples[s].y - pivot.y);

					//float intensity = images[imgidx*size + (int)samples[s].y*w + (int)samples[s].x];
					float proxy = dmh_samcol[dmh_Tri[vCurrent._triOffset + j] * SAMNUM + s];
					F[currentSample] = (intensity - proxy);
					/*J[currentSample * 2] = (float)ug[imgidx*size + samples[s].y*w + samples[s].x] * sampling_weights_current[s];
					J[currentSample * 2 + 1] = (float)vg[imgidx*size + samples[s].y*w + samples[s].x] * sampling_weights_current[s];*/
					J[currentSample * 2] = linearInterpolate(ug[imgidx * size + (pivot.y - 1) * w + pivot.x - 1],
						ug[imgidx * size + (pivot.y - 1) * w + pivot.x],
						ug[imgidx * size + (pivot.y) * w + pivot.x - 1],
						ug[imgidx * size + (pivot.y) * w + pivot.x],
						samples[s].x - pivot.x, samples[s].y - pivot.y) * sampling_weights_current[s];
					J[currentSample * 2 + 1] = linearInterpolate(vg[imgidx * size + (pivot.y - 1) * w + pivot.x - 1],
						vg[imgidx * size + (pivot.y - 1) * w + pivot.x],
						vg[imgidx * size + (pivot.y) * w + pivot.x - 1],
						vg[imgidx * size + (pivot.y) * w + pivot.x],
						samples[s].x - pivot.x, samples[s].y - pivot.y) * sampling_weights_current[s];
					currentSample++;
				}
			}
		}
		if (currentSample > 0) {
			float JTJ[4]; float JTF[2];
			float JTJinv[4];
			calcJTJ(JTJ, J, currentSample);
			calcJTF(JTF, J, F, currentSample);
			if (abs(JTJ[0] * JTJ[3] - JTJ[1] * JTJ[2]) < 0.0001) {
				JTJ[0] = JTJ[0] + 0.0001;
				JTJ[3] = JTJ[3] + 0.0001;
			}
			calcInv2x2(JTJinv, JTJ);
			float2 step = make_float2(JTJinv[0] * JTF[0] + JTJinv[1] * JTF[1], JTJinv[2] * JTF[0] + JTJinv[3] * JTF[1]);
			if (isnan(step.x) || isnan(step.y))
				step = make_float2(0, 0);
			// 여기여기
			step = clamp(step, -5.0, 5.0);
			dmh_Img_Tex[vertices[vidx]._imgOffset + i] = make_float2(clamp(dmh_Img_Tex[vertices[vidx]._imgOffset + i].x - step.x, 0.0, w - 1.0), clamp(dmh_Img_Tex[vertices[vidx]._imgOffset + i].y - step.y, 0.0, h - 1.0));
		}
	}
}

__global__ void update_texture_coordinate_cal_energy(float* energy, deviceVertex* vertices, deviceTriangle* triangles, int* dmh_ImgV, float2* dmh_Img_Tex, float2* dmh_Edge_Init, int* dmh_ImgT, bool* dmh_selT, bool* dmh_selT2, int* dmh_Tri, unsigned char* dmh_samcol, unsigned char* images, short* ug, short* vg, int w, int h, int nVertex)
{
	int vidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (vidx >= nVertex) return;
	deviceVertex vCurrent = vertices[vidx];
	int tmpImgNum = vCurrent._Img_Num;
	int tmpTriNum = vCurrent._Triangles_Num;
	float F[(SAMNUM + 2) * MAXTRI]; // sampling points, edge regularize, position regularize
	float J[(SAMNUM * 2 + 4) * MAXTRI];
	float2 samples[SAMNUM];
	float vertex_energy = 0;

	int size = w * h;

	deviceTriangle triangles_current;
	deviceVertex vertices_current[3];

	for (int i = 0; i < tmpImgNum; i++) {
		int imgidx = dmh_ImgV[vCurrent._imgOffset + i];

		int currentSample = 0;
		for (int j = 0; j < tmpTriNum && j < MAXTRI; j++) {
			triangles_current = triangles[dmh_Tri[vCurrent._triOffset + j]];
			vertices_current[0] = vertices[triangles_current._Vertices[0]];
			vertices_current[1] = vertices[triangles_current._Vertices[1]];
			vertices_current[2] = vertices[triangles_current._Vertices[2]];
			float* sampling_weights_current;
			if (triangles_current.isCoveredBy(imgidx, &dmh_ImgT[triangles_current._imgOffset])) {
				if (!dmh_selT2[triangles_current._imgOffset + triangles_current.getImgIdx(imgidx, &dmh_ImgT[triangles_current._imgOffset])])
					continue;

				float2 uv1, uv2, uv3;
				float2 edge_length;
				edge_length = dmh_Edge_Init[vCurrent._edgeOffset + i * tmpTriNum + j];
				// get other vertices
				if (triangles_current._Vertices[0] == vidx) {
					sampling_weights_current = sampling_weights[0];
					uv1 = dmh_Img_Tex[vCurrent._imgOffset + i];
					uv2 = dmh_Img_Tex[vertices_current[1]._imgOffset + vertices_current[1].getImgIdx(imgidx, &dmh_ImgV[vertices_current[1]._imgOffset])];
					uv3 = dmh_Img_Tex[vertices_current[2]._imgOffset + vertices_current[2].getImgIdx(imgidx, &dmh_ImgV[vertices_current[2]._imgOffset])];
					float2 diff1 = uv1 - uv2;
					float2 diff2 = uv1 - uv3;
					float dist = sqrt(dot(diff1, diff1));
					F[currentSample] = (dist - edge_length.x) * EDGE_WEIGHT_TO_ATLAS;
					J[currentSample * 2] = (diff1).x / (dist + 0.0001) * EDGE_WEIGHT_TO_ATLAS;
					J[currentSample * 2 + 1] = (diff1).y / (dist + 0.0001) * EDGE_WEIGHT_TO_ATLAS;
					currentSample++;
					dist = sqrt(dot(diff2, diff2));
					F[currentSample] = (dist - edge_length.y) * EDGE_WEIGHT_TO_ATLAS;
					J[currentSample * 2] = (diff2).x / (dist + 0.0001) * EDGE_WEIGHT_TO_ATLAS;
					J[currentSample * 2 + 1] = (diff2).y / (dist + 0.0001) * EDGE_WEIGHT_TO_ATLAS;
					currentSample++;
				}
				else if (triangles_current._Vertices[1] == vidx) {
					sampling_weights_current = sampling_weights[1];
					uv1 = dmh_Img_Tex[vertices_current[0]._imgOffset + vertices_current[0].getImgIdx(imgidx, &dmh_ImgV[vertices_current[0]._imgOffset])];
					uv2 = dmh_Img_Tex[vCurrent._imgOffset + i];
					uv3 = dmh_Img_Tex[vertices_current[2]._imgOffset + vertices_current[2].getImgIdx(imgidx, &dmh_ImgV[vertices_current[2]._imgOffset])];
					float2 diff1 = uv2 - uv1;
					float2 diff2 = uv2 - uv3;
					float dist = sqrt(dot(diff1, diff1));
					F[currentSample] = (dist - edge_length.x) * EDGE_WEIGHT_TO_ATLAS;
					J[currentSample * 2] = (diff1).x / (dist + 0.0001) * EDGE_WEIGHT_TO_ATLAS;
					J[currentSample * 2 + 1] = (diff1).y / (dist + 0.0001) * EDGE_WEIGHT_TO_ATLAS;
					currentSample++;
					dist = sqrt(dot(diff2, diff2));
					F[currentSample] = (dist - edge_length.y) * EDGE_WEIGHT_TO_ATLAS;
					J[currentSample * 2] = (diff2).x / (dist + 0.0001) * EDGE_WEIGHT_TO_ATLAS;
					J[currentSample * 2 + 1] = (diff2).y / (dist + 0.0001) * EDGE_WEIGHT_TO_ATLAS;
					currentSample++;
				}
				else {
					sampling_weights_current = sampling_weights[2];
					uv1 = dmh_Img_Tex[vertices_current[0]._imgOffset + vertices_current[0].getImgIdx(imgidx, &dmh_ImgV[vertices_current[0]._imgOffset])];
					uv2 = dmh_Img_Tex[vertices_current[1]._imgOffset + vertices_current[1].getImgIdx(imgidx, &dmh_ImgV[vertices_current[1]._imgOffset])];
					uv3 = dmh_Img_Tex[vCurrent._imgOffset + i];
					float2 diff1 = uv3 - uv1;
					float2 diff2 = uv3 - uv2;
					float dist = sqrt(dot(diff1, diff1));
					F[currentSample] = (dist - edge_length.x) * EDGE_WEIGHT_TO_ATLAS;
					J[currentSample * 2] = (diff1).x / (dist + 0.0001) * EDGE_WEIGHT_TO_ATLAS;
					J[currentSample * 2 + 1] = (diff1).y / (dist + 0.0001) * EDGE_WEIGHT_TO_ATLAS;
					currentSample++;
					dist = sqrt(dot(diff2, diff2));
					F[currentSample] = (dist - edge_length.y) * EDGE_WEIGHT_TO_ATLAS;
					J[currentSample * 2] = (diff2).x / (dist + 0.0001) * EDGE_WEIGHT_TO_ATLAS;
					J[currentSample * 2 + 1] = (diff2).y / (dist + 0.0001) * EDGE_WEIGHT_TO_ATLAS;
					currentSample++;
				}


				if (!dmh_selT[triangles_current._imgOffset + triangles_current.getImgIdx(imgidx, &dmh_ImgT[triangles_current._imgOffset])])
					continue;

				// sampling the uv coordinates
				for (int s = 0; s < SAMNUM; s++) {
					samples[s] = sampling(uv1, uv2, uv3, sampling_weights[0][s], sampling_weights[1][s], sampling_weights[2][s]);
				}

				// append the matrix F and J
				for (int s = 0; s < SAMNUM; s++) {
					// get intensity from the image
					int2 pivot = make_int2(samples[s].x + 0.5, samples[s].y + 0.5);

					float intensity = linearInterpolate(images[imgidx * size + (pivot.y - 1) * w + pivot.x - 1],
						images[imgidx * size + (pivot.y - 1) * w + pivot.x],
						images[imgidx * size + (pivot.y) * w + pivot.x - 1],
						images[imgidx * size + (pivot.y) * w + pivot.x],
						samples[s].x - pivot.x, samples[s].y - pivot.y);

					//float intensity = images[imgidx*size + (int)samples[s].y*w + (int)samples[s].x];
					float proxy = dmh_samcol[dmh_Tri[vCurrent._triOffset + j] * SAMNUM + s];
					F[currentSample] = (intensity - proxy);
					/*J[currentSample * 2] = (float)ug[imgidx*size + samples[s].y*w + samples[s].x] * sampling_weights_current[s];
					J[currentSample * 2 + 1] = (float)vg[imgidx*size + samples[s].y*w + samples[s].x] * sampling_weights_current[s];*/
					J[currentSample * 2] = linearInterpolate(ug[imgidx * size + (pivot.y - 1) * w + pivot.x - 1],
						ug[imgidx * size + (pivot.y - 1) * w + pivot.x],
						ug[imgidx * size + (pivot.y) * w + pivot.x - 1],
						ug[imgidx * size + (pivot.y) * w + pivot.x],
						samples[s].x - pivot.x, samples[s].y - pivot.y) * sampling_weights_current[s];
					J[currentSample * 2 + 1] = linearInterpolate(vg[imgidx * size + (pivot.y - 1) * w + pivot.x - 1],
						vg[imgidx * size + (pivot.y - 1) * w + pivot.x],
						vg[imgidx * size + (pivot.y) * w + pivot.x - 1],
						vg[imgidx * size + (pivot.y) * w + pivot.x],
						samples[s].x - pivot.x, samples[s].y - pivot.y) * sampling_weights_current[s];
					currentSample++;
				}
			}
		}
		if (currentSample > 0) {
			float JTJ[4]; float JTF[2];
			float JTJinv[4];
			calcJTJ(JTJ, J, currentSample);
			calcJTF(JTF, J, F, currentSample);
			if (abs(JTJ[0] * JTJ[3] - JTJ[1] * JTJ[2]) < 0.0001) {
				JTJ[0] = JTJ[0] + 0.0001;
				JTJ[3] = JTJ[3] + 0.0001;
			}
			calcInv2x2(JTJinv, JTJ);
			float2 step = make_float2(JTJinv[0] * JTF[0] + JTJinv[1] * JTF[1], JTJinv[2] * JTF[0] + JTJinv[3] * JTF[1]);
			if (isnan(step.x) || isnan(step.y))
				step = make_float2(0, 0);
			// 여기여기
			step = clamp(step, -5.0, 5.0);
			dmh_Img_Tex[vertices[vidx]._imgOffset + i] = make_float2(clamp(dmh_Img_Tex[vertices[vidx]._imgOffset + i].x - step.x, 0.0, w - 1.0), clamp(dmh_Img_Tex[vertices[vidx]._imgOffset + i].y - step.y, 0.0, h - 1.0));
			for (int k = 2; k < currentSample; k+=3)
				vertex_energy += F[k] * F[k] / 65025.0;
		}
	}
	energy[vidx] = vertex_energy;
}


__global__ void calc_energy_sel(float* energy, deviceVertex* vertices, deviceTriangle* triangles, int* dmh_ImgV, float2* dmh_Img_Tex, int* dmh_ImgT, bool* dmh_selT, int* dmh_Tri, unsigned char* dmh_samcol, unsigned char* images, short* ug, short* vg, int w, int h, int nVertex)
{
	int vidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (vidx >= nVertex) return;
	deviceVertex vCurrent = vertices[vidx];
	int tmpImgNum = vCurrent._Img_Num;
	int tmpTriNum = vCurrent._Triangles_Num;
	//float *F = (float*)malloc(SAMNUM * tmpTriNum);
	//float *F = new float[(SAMNUM) * tmpTriNum];
	float F[SAMNUM * MAXTRI]; // sampling points, edge regularize, position regularize
	int2 samples[SAMNUM];

	float vertex_energy = 0;
	int size = w * h;
	int imgcounted = 0;

	deviceTriangle triangles_current;
	deviceVertex vertices_current[3];

	for (int i = 0; i < tmpImgNum; i++) {
		int imgidx = dmh_ImgV[vCurrent._imgOffset + i];
		int currentSample = 0;
		for (int j = 0; j < tmpTriNum && j < MAXTRI; j++) {
			triangles_current = triangles[dmh_Tri[vCurrent._triOffset + j]];
			vertices_current[0] = vertices[triangles_current._Vertices[0]];
			vertices_current[1] = vertices[triangles_current._Vertices[1]];
			vertices_current[2] = vertices[triangles_current._Vertices[2]];

			float* sampling_weights_current;
			if (triangles_current.isCoveredBy(imgidx, &dmh_ImgT[triangles_current._imgOffset])) {

				int imgidx1, imgidx2, imgidx3;
				float2 uv1, uv2, uv3;
				// get other vertices
				if (triangles_current._Vertices[0] == vidx) {
					sampling_weights_current = sampling_weights[0];
					uv1 = dmh_Img_Tex[vCurrent._imgOffset + i];
					uv2 = dmh_Img_Tex[vertices_current[1]._imgOffset + vertices_current[1].getImgIdx(imgidx, &dmh_ImgV[vertices_current[1]._imgOffset])];
					uv3 = dmh_Img_Tex[vertices_current[2]._imgOffset + vertices_current[2].getImgIdx(imgidx, &dmh_ImgV[vertices_current[2]._imgOffset])];
				}
				else if (triangles_current._Vertices[1] == vidx) {
					sampling_weights_current = sampling_weights[1];
					uv1 = dmh_Img_Tex[vertices_current[0]._imgOffset + vertices_current[0].getImgIdx(imgidx, &dmh_ImgV[vertices_current[0]._imgOffset])];
					uv2 = dmh_Img_Tex[vCurrent._imgOffset + i];
					uv3 = dmh_Img_Tex[vertices_current[2]._imgOffset + vertices_current[2].getImgIdx(imgidx, &dmh_ImgV[vertices_current[2]._imgOffset])];
				}
				else if (triangles_current._Vertices[2] == vidx) {
					sampling_weights_current = sampling_weights[2];
					uv1 = dmh_Img_Tex[vertices_current[0]._imgOffset + vertices_current[0].getImgIdx(imgidx, &dmh_ImgV[vertices_current[0]._imgOffset])];
					uv2 = dmh_Img_Tex[vertices_current[1]._imgOffset + vertices_current[1].getImgIdx(imgidx, &dmh_ImgV[vertices_current[1]._imgOffset])];
					uv3 = dmh_Img_Tex[vCurrent._imgOffset + i];
				}
				else {
					continue;
				}


				if (!dmh_selT[triangles_current._imgOffset + triangles_current.getImgIdx(imgidx, &dmh_ImgT[triangles_current._imgOffset])] || triangles_current.getImgIdx(imgidx, &dmh_ImgT[triangles_current._imgOffset]) < 0)
					continue;

				// sampling the uv coordinates
				for (int s = 0; s < SAMNUM; s++) {
					samples[s] = make_int2(sampling(uv1, uv2, uv3, sampling_weights[0][s], sampling_weights[1][s], sampling_weights[2][s]));
					samples[s].x = min(w - 1, max(0, samples[s].x));
					samples[s].y = min(h - 1, max(0, samples[s].y));
				}

				// append the matrix F and J
				for (int s = 0; s < SAMNUM; s++) {
					int2 pivot = make_int2(samples[s].x + 0.5, samples[s].y + 0.5);

					float intensity = linearInterpolate(images[imgidx * size + (pivot.y - 1) * w + pivot.x - 1],
						images[imgidx * size + (pivot.y - 1) * w + pivot.x],
						images[imgidx * size + (pivot.y) * w + pivot.x - 1],
						images[imgidx * size + (pivot.y) * w + pivot.x],
						samples[s].x - pivot.x, samples[s].y - pivot.y);
					// get intensity from the image
					//float intensity = images[imgidx*size + samples[s].y*w + samples[s].x];
					float proxy = dmh_samcol[dmh_Tri[vCurrent._triOffset + j] * SAMNUM + s];
					F[currentSample] = (intensity - proxy) / 255.0f;
					currentSample++;
				}
			}
		}
		if (currentSample > 0) {
			for (int k = 0; k < currentSample; k++)
				vertex_energy += F[k] * F[k];
			imgcounted++;
		}
	}

	energy[vidx] = vertex_energy;
}

__global__ void calc_energy_sel2(float* energy, deviceVertex* vertices, deviceTriangle* triangles, int* dmh_ImgV, float2* dmh_Img_Tex, int* dmh_ImgT, bool* dmh_selT, bool* dmh_selT2, int* dmh_Tri, unsigned char* dmh_samcol, unsigned char* images, short* ug, short* vg, int w, int h, int nVertex)
{
	int vidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (vidx >= nVertex) return;
	deviceVertex vCurrent = vertices[vidx];
	int tmpImgNum = vCurrent._Img_Num;
	int tmpTriNum = vCurrent._Triangles_Num;
	//float *F = (float*)malloc(SAMNUM * tmpTriNum);
	//float *F = new float[(SAMNUM) * tmpTriNum];
	float F[SAMNUM * MAXTRI]; // sampling points, edge regularize, position regularize
	int2 samples[SAMNUM];

	float vertex_energy = 0;
	int size = w * h;
	int imgcounted = 0;

	deviceTriangle triangles_current;
	deviceVertex vertices_current[3];

	for (int i = 0; i < tmpImgNum; i++) {
		int imgidx = dmh_ImgV[vCurrent._imgOffset + i];
		int currentSample = 0;
		for (int j = 0; j < tmpTriNum && j < MAXTRI; j++) {
			triangles_current = triangles[dmh_Tri[vCurrent._triOffset + j]];
			vertices_current[0] = vertices[triangles_current._Vertices[0]];
			vertices_current[1] = vertices[triangles_current._Vertices[1]];
			vertices_current[2] = vertices[triangles_current._Vertices[2]];

			float* sampling_weights_current;
			if (triangles_current.isCoveredBy(imgidx, &dmh_ImgT[triangles_current._imgOffset])) {
				if (!dmh_selT2[triangles_current._imgOffset + triangles_current.getImgIdx(imgidx, &dmh_ImgT[triangles_current._imgOffset])])
					continue;

				int imgidx1, imgidx2, imgidx3;
				float2 uv1, uv2, uv3;
				// get other vertices
				if (triangles_current._Vertices[0] == vidx) {
					sampling_weights_current = sampling_weights[0];
					uv1 = dmh_Img_Tex[vCurrent._imgOffset + i];
					uv2 = dmh_Img_Tex[vertices_current[1]._imgOffset + vertices_current[1].getImgIdx(imgidx, &dmh_ImgV[vertices_current[1]._imgOffset])];
					uv3 = dmh_Img_Tex[vertices_current[2]._imgOffset + vertices_current[2].getImgIdx(imgidx, &dmh_ImgV[vertices_current[2]._imgOffset])];
				}
				else if (triangles_current._Vertices[1] == vidx) {
					sampling_weights_current = sampling_weights[1];
					uv1 = dmh_Img_Tex[vertices_current[0]._imgOffset + vertices_current[0].getImgIdx(imgidx, &dmh_ImgV[vertices_current[0]._imgOffset])];
					uv2 = dmh_Img_Tex[vCurrent._imgOffset + i];
					uv3 = dmh_Img_Tex[vertices_current[2]._imgOffset + vertices_current[2].getImgIdx(imgidx, &dmh_ImgV[vertices_current[2]._imgOffset])];
				}
				else if (triangles_current._Vertices[2] == vidx) {
					sampling_weights_current = sampling_weights[2];
					uv1 = dmh_Img_Tex[vertices_current[0]._imgOffset + vertices_current[0].getImgIdx(imgidx, &dmh_ImgV[vertices_current[0]._imgOffset])];
					uv2 = dmh_Img_Tex[vertices_current[1]._imgOffset + vertices_current[1].getImgIdx(imgidx, &dmh_ImgV[vertices_current[1]._imgOffset])];
					uv3 = dmh_Img_Tex[vCurrent._imgOffset + i];
				}
				else {
					continue;
				}


				if (!dmh_selT[triangles_current._imgOffset + triangles_current.getImgIdx(imgidx, &dmh_ImgT[triangles_current._imgOffset])] || triangles_current.getImgIdx(imgidx, &dmh_ImgT[triangles_current._imgOffset]) < 0)
					continue;

				// sampling the uv coordinates
				for (int s = 0; s < SAMNUM; s++) {
					samples[s] = make_int2(sampling(uv1, uv2, uv3, sampling_weights[0][s], sampling_weights[1][s], sampling_weights[2][s]));
					samples[s].x = min(w - 1, max(0, samples[s].x));
					samples[s].y = min(h - 1, max(0, samples[s].y));
				}

				// append the matrix F and J
				for (int s = 0; s < SAMNUM; s++) {
					int2 pivot = make_int2(samples[s].x + 0.5, samples[s].y + 0.5);

					float intensity = linearInterpolate(images[imgidx * size + (pivot.y - 1) * w + pivot.x - 1],
						images[imgidx * size + (pivot.y - 1) * w + pivot.x],
						images[imgidx * size + (pivot.y) * w + pivot.x - 1],
						images[imgidx * size + (pivot.y) * w + pivot.x],
						samples[s].x - pivot.x, samples[s].y - pivot.y);
					// get intensity from the image
					//float intensity = images[imgidx*size + samples[s].y*w + samples[s].x];
					float proxy = dmh_samcol[dmh_Tri[vCurrent._triOffset + j] * SAMNUM + s];
					F[currentSample] = (intensity - proxy) / 255.0f;
					currentSample++;
				}
			}
		}
		if (currentSample > 0) {
			for (int k = 0; k < currentSample; k++)
				vertex_energy += F[k] * F[k];
			imgcounted++;
		}
	}

	energy[vidx] = vertex_energy;
}


__global__ void update_proxy_color_sel(deviceVertex* vertices, deviceTriangle* triangles, int* dmh_ImgV, float2* dmh_Img_Tex, int* dmh_ImgT, bool* dmh_selT, unsigned char* dmh_samcol, float* dmh_weight, unsigned char* images, int w, int h, int nTriangle)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (tidx >= nTriangle) return;
	deviceTriangle tCurrent = triangles[tidx];
	int tmpImgNum = tCurrent._Img_Num;
	deviceVertex* v1 = &vertices[tCurrent._Vertices[0]];
	deviceVertex* v2 = &vertices[tCurrent._Vertices[1]];
	deviceVertex* v3 = &vertices[tCurrent._Vertices[2]];
	int size = w * h;
	float samcol[SAMNUM];
	float tmp_weight = 0.0;
	for (int i = 0; i < SAMNUM; i++) { samcol[i] = 0; }
	for (int i = 0; i < tmpImgNum; i++) { // for each image projects to this triangle
		if (!dmh_selT[tCurrent._imgOffset + i])
			continue;
		int imgidx = dmh_ImgT[tCurrent._imgOffset + i];

		int imgidx1 = v1->getImgIdx(imgidx, &dmh_ImgV[v1->_imgOffset]);
		int imgidx2 = v2->getImgIdx(imgidx, &dmh_ImgV[v2->_imgOffset]);
		int imgidx3 = v3->getImgIdx(imgidx, &dmh_ImgV[v3->_imgOffset]);
		float2 uv1 = dmh_Img_Tex[v1->_imgOffset + imgidx1];
		float2 uv2 = dmh_Img_Tex[v2->_imgOffset + imgidx2];
		float2 uv3 = dmh_Img_Tex[v3->_imgOffset + imgidx3];

		float2 samples[SAMNUM];
		// sampling the uv coordinates
		for (int s = 0; s < SAMNUM; s++)
			samples[s] = sampling(uv1, uv2, uv3, sampling_weights[0][s], sampling_weights[1][s], sampling_weights[2][s]);
		for (int s = 0; s < SAMNUM; s++) {
			// get intensity from the image

			//samcol[s] += images[imgidx*w*h + samples[s].y*w + samples[s].x];// *tCurrent._Img_Weight[i];
			int2 pivot = make_int2(samples[s].x + 0.5, samples[s].y + 0.5);
			samcol[s] += linearInterpolate(images[imgidx * size + (pivot.y - 1) * w + pivot.x - 1],
				images[imgidx * size + (pivot.y - 1) * w + pivot.x],
				images[imgidx * size + (pivot.y) * w + pivot.x - 1],
				images[imgidx * size + (pivot.y) * w + pivot.x],
				samples[s].x - pivot.x, samples[s].y - pivot.y) * tCurrent.weightedBy(imgidx, &dmh_ImgT[tCurrent._imgOffset], &dmh_weight[tCurrent._imgOffset]);
		}
		tmp_weight += tCurrent.weightedBy(imgidx, &dmh_ImgT[tCurrent._imgOffset], &dmh_weight[tCurrent._imgOffset]);
	}
	for (int s = 0; s < SAMNUM; s++)
		//dmh_samcol[tidx * SAMNUM + s] = uchar(clamp(samcol[s] / tmpImgNum, 0.f, 255.f));
		dmh_samcol[tidx * SAMNUM + s] = uchar(clamp(samcol[s] / tmp_weight, 0.f, 255.f));
}

__global__ void update_proxy_color_sel2(deviceVertex* vertices, deviceTriangle* triangles, int* dmh_ImgV, float2* dmh_Img_Tex, int* dmh_ImgT, bool* dmh_selT, bool* dmh_selT2, unsigned char* dmh_samcol, float* dmh_weight, unsigned char* images, int w, int h, int nTriangle)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (tidx >= nTriangle) return;
	deviceTriangle tCurrent = triangles[tidx];
	int tmpImgNum = tCurrent._Img_Num;
	deviceVertex* v1 = &vertices[tCurrent._Vertices[0]];
	deviceVertex* v2 = &vertices[tCurrent._Vertices[1]];
	deviceVertex* v3 = &vertices[tCurrent._Vertices[2]];
	int size = w * h;
	float samcol[SAMNUM];
	float tmp_weight = 0.0;
	for (int i = 0; i < SAMNUM; i++) { samcol[i] = 0; }
	for (int i = 0; i < tmpImgNum; i++) { // for each image projects to this triangle
		if (!dmh_selT[tCurrent._imgOffset + i])
			continue;
		int imgidx = dmh_ImgT[tCurrent._imgOffset + i];

		int imgidx1 = v1->getImgIdx(imgidx, &dmh_ImgV[v1->_imgOffset]);
		int imgidx2 = v2->getImgIdx(imgidx, &dmh_ImgV[v2->_imgOffset]);
		int imgidx3 = v3->getImgIdx(imgidx, &dmh_ImgV[v3->_imgOffset]);
		float2 uv1 = dmh_Img_Tex[v1->_imgOffset + imgidx1];
		float2 uv2 = dmh_Img_Tex[v2->_imgOffset + imgidx2];
		float2 uv3 = dmh_Img_Tex[v3->_imgOffset + imgidx3];

		float2 samples[SAMNUM];
		// sampling the uv coordinates
		for (int s = 0; s < SAMNUM; s++)
			samples[s] = sampling(uv1, uv2, uv3, sampling_weights[0][s], sampling_weights[1][s], sampling_weights[2][s]);
		for (int s = 0; s < SAMNUM; s++) {
			// get intensity from the image

			//samcol[s] += images[imgidx*w*h + samples[s].y*w + samples[s].x];// *tCurrent._Img_Weight[i];
			int2 pivot = make_int2(samples[s].x + 0.5, samples[s].y + 0.5);
			samcol[s] += linearInterpolate(images[imgidx * size + (pivot.y - 1) * w + pivot.x - 1],
				images[imgidx * size + (pivot.y - 1) * w + pivot.x],
				images[imgidx * size + (pivot.y) * w + pivot.x - 1],
				images[imgidx * size + (pivot.y) * w + pivot.x],
				samples[s].x - pivot.x, samples[s].y - pivot.y) * tCurrent.weightedBy(imgidx, &dmh_ImgT[tCurrent._imgOffset], &dmh_weight[tCurrent._imgOffset]);
		}
		tmp_weight += tCurrent.weightedBy(imgidx, &dmh_ImgT[tCurrent._imgOffset], &dmh_weight[tCurrent._imgOffset]);
	}
	for (int s = 0; s < SAMNUM; s++)
		//dmh_samcol[tidx * SAMNUM + s] = uchar(clamp(samcol[s] / tmpImgNum, 0.f, 255.f));
		dmh_samcol[tidx * SAMNUM + s] = uchar(clamp(samcol[s] / tmp_weight, 0.f, 255.f));
}

__global__ void update_proxy_color_mask(deviceVertex* vertices, deviceTriangle* triangles, int* dmh_ImgV, float2* dmh_Img_Tex, int* dmh_ImgT, bool* dmh_selT, unsigned char* dmh_samcol, float* dmh_weight, unsigned char* images, unsigned char* masks, int w, int h, int nTriangle)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (tidx >= nTriangle) return;
	deviceTriangle tCurrent = triangles[tidx];
	int tmpImgNum = tCurrent._Img_Num;
	deviceVertex* v1 = &vertices[tCurrent._Vertices[0]];
	deviceVertex* v2 = &vertices[tCurrent._Vertices[1]];
	deviceVertex* v3 = &vertices[tCurrent._Vertices[2]];
	int size = w * h;
	float samcol[SAMNUM];
	float tmp_weight[SAMNUM];
	for (int i = 0; i < SAMNUM; i++) {
		samcol[i] = 0; 
		tmp_weight[i] = 0;
	}
	for (int i = 0; i < tmpImgNum; i++) { // for each image projects to this triangle
		if (!dmh_selT[tCurrent._imgOffset + i])
			continue;
		int imgidx = dmh_ImgT[tCurrent._imgOffset + i];

		int imgidx1 = v1->getImgIdx(imgidx, &dmh_ImgV[v1->_imgOffset]);
		int imgidx2 = v2->getImgIdx(imgidx, &dmh_ImgV[v2->_imgOffset]);
		int imgidx3 = v3->getImgIdx(imgidx, &dmh_ImgV[v3->_imgOffset]);
		float2 uv1 = dmh_Img_Tex[v1->_imgOffset + imgidx1];
		float2 uv2 = dmh_Img_Tex[v2->_imgOffset + imgidx2];
		float2 uv3 = dmh_Img_Tex[v3->_imgOffset + imgidx3];

		for (int s = 0; s < SAMNUM; s++) {
			// get intensity from the image
			float2 samples = sampling(uv1, uv2, uv3, sampling_weights[0][s], sampling_weights[1][s], sampling_weights[2][s]);
			//samcol[s] += images[imgidx*w*h + samples[s].y*w + samples[s].x];// *tCurrent._Img_Weight[i];
			int2 pivot = make_int2(samples.x + 0.5, samples.y + 0.5);
			float fg = 1.0;
			if (linearInterpolate(masks[imgidx * size + (pivot.y - 1) * w + pivot.x - 1],
				masks[imgidx * size + (pivot.y - 1) * w + pivot.x],
				masks[imgidx * size + (pivot.y) * w + pivot.x - 1],
				masks[imgidx * size + (pivot.y) * w + pivot.x],
				samples.x - pivot.x, samples.y - pivot.y) < 1.0)
				fg = 0.001;
			samcol[s] += linearInterpolate(images[imgidx * size + (pivot.y - 1) * w + pivot.x - 1],
				images[imgidx * size + (pivot.y - 1) * w + pivot.x],
				images[imgidx * size + (pivot.y) * w + pivot.x - 1],
				images[imgidx * size + (pivot.y) * w + pivot.x],
				samples.x - pivot.x, samples.y - pivot.y) * tCurrent.weightedBy(imgidx, &dmh_ImgT[tCurrent._imgOffset], &dmh_weight[tCurrent._imgOffset]) * fg;
			tmp_weight[s] += tCurrent.weightedBy(imgidx, &dmh_ImgT[tCurrent._imgOffset], &dmh_weight[tCurrent._imgOffset]) * fg;
		}
	}
	for (int s = 0; s < SAMNUM; s++)
		//dmh_samcol[tidx * SAMNUM + s] = uchar(clamp(samcol[s] / tmpImgNum, 0.f, 255.f));
		dmh_samcol[tidx * SAMNUM + s] = uchar(clamp(samcol[s] / tmp_weight[s], 0.f, 255.f));
}

StopWatchInterface* Optimizer::sample_host(int width, int height, int nVertex, int nTriangle, float *hEnergy, int currentIter)
{
	StopWatchInterface *t = NULL;
	sdkCreateTimer(&t);
	sdkStartTimer(&t);
	checkCudaErrors(cudaDeviceSynchronize());

	dim3 gridSize_proxy((nTriangle + 512 - 1) / 512);
	dim3 blockSize_proxy(512);

	//printf("update_proxy_color\n");
	update_proxy_color <<<gridSize_proxy, blockSize_proxy >>>(ddVertices, ddTriangles, dmhVertices._Img, dmhVertices._Img_Tex, dmhTriangles._Img, dmhTriangles._SamCol, dmhTriangles._Weight, dImages_original, width, height, nTriangle);
	cudaCheckError();

	dim3 gridSize((nVertex + 256 - 1) / 256);
	dim3 blockSize(256);
	//printf("update_texture_coordinate\n");
	update_texture_coordinate <<< gridSize, blockSize >>>(ddVertices, ddTriangles, dmhVertices._Img, dmhVertices._Img_Tex, dmhVertices._Edge_Init, dmhTriangles._Img, dmhVertices._Triangles, dmhTriangles._SamCol, dImages, dUg, dVg, width, height, nVertex);
	checkCudaErrors(cudaDeviceSynchronize());
	cudaCheckError();
	update_texture_coordinate <<< gridSize, blockSize >>>(ddVertices, ddTriangles, dmhVertices._Img, dmhVertices._Img_Tex, dmhVertices._Edge_Init, dmhTriangles._Img, dmhVertices._Triangles, dmhTriangles._SamCol, dImages, dUg, dVg, width, height, nVertex);
	checkCudaErrors(cudaDeviceSynchronize());
	cudaCheckError();


	update_proxy_color << <gridSize, blockSize >> >(ddVertices, ddTriangles, dmhVertices._Img, dmhVertices._Img_Tex, dmhTriangles._Img, dmhTriangles._SamCol, dmhTriangles._Weight, dImages_original, width, height, nTriangle);
	cudaMemset(dEnergy, 0, sizeof(float)*nVertex);
	calc_energy << < gridSize, blockSize >> >(dEnergy, ddVertices, ddTriangles, dmhVertices._Img, dmhVertices._Img_Tex, dmhTriangles._Img, dmhVertices._Triangles, dmhTriangles._SamCol, dImages_original, dUg, dVg, width, height, nVertex);

	checkCudaErrors(cudaMemcpy(hEnergy, dEnergy, sizeof(float)*nVertex, cudaMemcpyDeviceToHost));
	float sum_energy = 0;
	for (int i = 0; i < nVertex; i++) {
		sum_energy += hEnergy[i];
		//  printf("%f ", hEnergy[i]);
	}
	//sum_energy /= nVertex;
	printf("energy: %f, ", sum_energy);

	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&t);
	return t;
}

StopWatchInterface* Optimizer::sample_host_sel(int width, int height, int nVertex, int nTriangle, float* hEnergy, int currentIter)
{
	StopWatchInterface* t = NULL;
	sdkCreateTimer(&t);
	sdkStartTimer(&t);
	checkCudaErrors(cudaDeviceSynchronize());

	dim3 gridSize_proxy((nTriangle + 512 - 1) / 512);
	dim3 blockSize_proxy(512);

	//printf("update_proxy_color\n");
	//update_proxy_color_sel<< <gridSize_proxy, blockSize_proxy >> > (ddVertices, ddTriangles, dmhVertices._Img, dmhVertices._Img_Tex, dmhTriangles._Img, dmhTriangles._Bound, dmhTriangles._SamCol, dmhTriangles._Weight, dImages_original, width, height, nTriangle);
	update_proxy_color_mask<< <gridSize_proxy, blockSize_proxy >> > (ddVertices, ddTriangles, dmhVertices._Img, dmhVertices._Img_Tex, dmhTriangles._Img, dmhTriangles._Bound, dmhTriangles._SamCol, dmhTriangles._Weight, dImages_original, dImages_mask, width, height, nTriangle);
	cudaCheckError();

	dim3 gridSize((nVertex + 256 - 1) / 256);
	dim3 blockSize(256);
	//printf("update_texture_coordinate\n");
	update_texture_coordinate_sel2 << < gridSize, blockSize >> > (ddVertices, ddTriangles, dmhVertices._Img, dmhVertices._Img_Tex, dmhVertices._Edge_Init, dmhTriangles._Img, dmhTriangles._Bound, dmhTriangles._Label, dmhVertices._Triangles, dmhTriangles._SamCol, dImages, dUg, dVg, width, height, nVertex);
	checkCudaErrors(cudaDeviceSynchronize());
	cudaCheckError();
	update_texture_coordinate_sel2 << < gridSize, blockSize >> > (ddVertices, ddTriangles, dmhVertices._Img, dmhVertices._Img_Tex, dmhVertices._Edge_Init, dmhTriangles._Img, dmhTriangles._Bound, dmhTriangles._Label, dmhVertices._Triangles, dmhTriangles._SamCol, dImages, dUg, dVg, width, height, nVertex);
	checkCudaErrors(cudaDeviceSynchronize());
	cudaCheckError();


	//update_proxy_color_sel<< <gridSize_proxy, blockSize_proxy >> > (ddVertices, ddTriangles, dmhVertices._Img, dmhVertices._Img_Tex, dmhTriangles._Img, dmhTriangles._Bound, dmhTriangles._SamCol, dmhTriangles._Weight, dImages_original, width, height, nTriangle);
	update_proxy_color_mask << <gridSize_proxy, blockSize_proxy >> > (ddVertices, ddTriangles, dmhVertices._Img, dmhVertices._Img_Tex, dmhTriangles._Img, dmhTriangles._Bound, dmhTriangles._SamCol, dmhTriangles._Weight, dImages_original, dImages_mask, width, height, nTriangle);

	cudaMemset(dEnergy, 0, sizeof(float) * nVertex);
	calc_energy_sel2 << < gridSize, blockSize >> > (dEnergy, ddVertices, ddTriangles, dmhVertices._Img, dmhVertices._Img_Tex, dmhTriangles._Img, dmhTriangles._Bound, dmhTriangles._Label, dmhVertices._Triangles, dmhTriangles._SamCol, dImages_original, dUg, dVg, width, height, nVertex);

	checkCudaErrors(cudaMemcpy(hEnergy, dEnergy, sizeof(float) * nVertex, cudaMemcpyDeviceToHost));
	float sum_energy = 0;
	for (int i = 0; i < nVertex; i++) {
		sum_energy += hEnergy[i];
		//  printf("%f ", hEnergy[i]);
	}
	//sum_energy /= nVertex;
	printf("energy: %f, ", sum_energy);

	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&t);
	return t;
}

StopWatchInterface* Optimizer::sample_host_toAtlas(int width, int height, int nVertex, int nTriangle, float* hEnergy)
{
	StopWatchInterface* t = NULL;
	sdkCreateTimer(&t);
	sdkStartTimer(&t);
	checkCudaErrors(cudaDeviceSynchronize());


	dim3 gridSize((nVertex + 256 - 1) / 256);
	dim3 blockSize(256);
	//printf("update_texture_coordinate\n");
	cudaMemset(dEnergy, 0, sizeof(float) * nVertex);
	update_texture_coordinate_cal_energy << < gridSize, blockSize >> > (dEnergy, ddVertices, ddTriangles, dmhVertices._Img, dmhVertices._Img_Tex, dmhVertices._Edge_Init, dmhTriangles._Img, dmhTriangles._Bound, dmhTriangles._Label, dmhVertices._Triangles, dmhTriangles._SamCol, dImages, dUg, dVg, width, height, nVertex);
	checkCudaErrors(cudaDeviceSynchronize());
	cudaCheckError();

	checkCudaErrors(cudaMemcpy(hEnergy, dEnergy, sizeof(float) * nVertex, cudaMemcpyDeviceToHost));
	float sum_energy = 0;
	for (int i = 0; i < nVertex; i++) {
		sum_energy += hEnergy[i];
	}
	printf("energy: %f, ", sum_energy);

	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&t);
	return t;
}


Optimizer::~Optimizer() {
	delete[] atlas_coord;
	delete[] img_coord;
	delete[] img_coord_mask;
	delete[] All_img_coord;
	freeCuda();
	freeCuda_onetime();
}
void Optimizer::Initialize(int layer, bool use_keyframe, bool resue) {
	if (layer == 0 && !resue) {
		hImages = new uchar[nImage*COIMX*COIMY];
		hImages_mask = new uchar[nImage * COIMX * COIMY];
		hUg = new short[nImage*COIMX*COIMY];
		hVg = new short[nImage*COIMX*COIMY];
		for (int i = 0; i < nImage; i++) {
			cv::Mat1b grayImage;
			cv::cvtColor(mapper4D_ptr->colorImages[mapper4D_ptr->validFrame_table[i]], grayImage, CV_BGR2GRAY);
			memcpy(&hImages_mask[i * COIMX * COIMY], mapper4D_ptr->maskImages[mapper4D_ptr->validFrame_table[i]].data, sizeof(uchar) * COIMY * COIMX);
			memcpy(&hImages[i*COIMX*COIMY], grayImage.data, sizeof(uchar)*COIMY*COIMX);
		}

		initCuda_onetime(COIMX, COIMY, nImage);
		checkCudaErrors(cudaMemcpy(dImages_original, hImages, sizeof(uchar) * COIMY * COIMX * nImage, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(dImages_mask, hImages_mask, sizeof(uchar) * COIMY * COIMX * nImage, cudaMemcpyHostToDevice));
	}
	else {
		delete[] hEnergy;
		freeCuda();
	}
	hEnergy = new float[nVertex];

	initCuda(COIMX, COIMY, nVertex, nTriangle, nImage);
	dmhVertices.clear();
	dmhTriangles.clear();

	StopWatchInterface *t = NULL;
	sdkCreateTimer(&t);

	sdkStartTimer(&t);

	size_t imgOffsetT = 0;
	for (int i = 0; i < nTriangle; i++) {
		ddTriangles[i].init(hTriangles[i], imgOffsetT);

		dmhTriangles.push_back(hTriangles[i]);

		imgOffsetT += hTriangles[i]._Img.size();
	}
	dmhTriangles.ready();
	sdkStopTimer(&t);
	float layer_time = sdkGetAverageTimerValue(&t) / 1000.0f;
	printf("//////////////////////////////////////////separate f time: %fms\n", layer_time * 1000);
	t = NULL;
	sdkCreateTimer(&t);
	sdkStartTimer(&t);

	size_t imgOffsetV = 0;
	size_t triOffsetV = 0;
	size_t edgeOffsetV = 0;
	for (int i = 0; i < nVertex; i++) {
		ddVertices[i].init(hVertices[i], imgOffsetV, triOffsetV, edgeOffsetV);
		dmhVertices.push_back(hVertices[i]);
		imgOffsetV += hVertices[i]._Img.size();
		triOffsetV += hVertices[i]._Triangles.size();
		edgeOffsetV += hVertices[i]._Triangles.size() * hVertices[i]._Img.size();
	}
	dmhVertices.ready();
	sdkStopTimer(&t);
	layer_time = sdkGetAverageTimerValue(&t) / 1000.0f;
	printf("//////////////////////////////////////////separate v time: %fms\n", layer_time * 1000);


#ifdef _DEBUG
	gridSize = dim3((nVertex + 512 - 1) / 512);
	blockSize = dim3(512);
#else
	gridSize = dim3((nVertex + 1024 - 1) / 1024);
	blockSize = dim3(1024);
#endif
	update_initial_edge << <gridSize, blockSize >> >(ddVertices, ddTriangles, dmhVertices._Img, dmhVertices._Img_Tex, dmhVertices._Edge_Init, dmhTriangles._Img, dmhVertices._Triangles, dImages, COIMX, COIMY, nVertex);
	cudaCheckError();
	printf("Initializing done...\n");
}
void Optimizer::Clear() {
	delete[] atlas_coord;
	delete[] img_coord;
	delete[] img_coord_mask;
	delete[] atlas_tri_idx;
	delete[] All_img_coord;
	delete[] hImages;
	delete[] hImages_mask;
	delete[] hUg;
	delete[] hVg;
	delete[] hEnergy;
	atlas_coord = NULL;
	img_coord = NULL;
	img_coord_mask = NULL;
	atlas_tri_idx = NULL;
	All_img_coord = NULL;

	freeCuda();
	freeCuda_onetime();
	for (int layer = 0; layer < LAYERNUM; layer++) {
		vector<cv::Mat>().swap(blur_vec[layer]);
		vector<cv::Mat>().swap(hUg_vec[layer]);
		vector<cv::Mat>().swap(hVg_vec[layer]);
	}

	for (int i = 0; i < nImage; i++) {
		vector<float2>().swap(propVec[i]);
	}

}
void Optimizer::Reinit(int layer, int timestamp) {
	mapper4D_ptr->ResetTex();
	//mapper4D_ptr->SetPropVec(propVec, layer);
	dmhTriangles.update_bound(mapper4D_ptr->Bound_vec[timestamp], mapper4D_ptr->Label_vec[timestamp]);
	mapper4D_ptr->Get_V_layer(time_hVertices[timestamp], layer);
	vector<float2> tmp_tex_vec;
	for (auto v : time_hVertices[timestamp]) {
		for (auto t : v._Img_Tex) {
			tmp_tex_vec.push_back(t);
		}
	}
	dmhVertices.update_texcoord(tmp_tex_vec);
	
#ifdef _DEBUG
	gridSize = dim3((nVertex + 512 - 1) / 512);
	blockSize = dim3(512);
#else
	gridSize = dim3((nVertex + 1024 - 1) / 1024);
	blockSize = dim3(1024);
#endif
	update_initial_edge << <gridSize, blockSize >> > (ddVertices, ddTriangles, dmhVertices._Img, dmhVertices._Img_Tex, dmhVertices._Edge_Init, dmhTriangles._Img, dmhVertices._Triangles, dImages, COIMX, COIMY, nVertex);
	cudaCheckError();
	
	delete[] hEnergy;
	hEnergy = new float[nVertex];

	printf("Initializing done...\n");
}

int Optimizer::Update(int iteration, int layer, bool layerup) {
	int layeriter = 0;
	int till_iter = 0;
	for (int i = 0; i < iteration; i++) {
		till_iter++;
		if (layerup) {
			for (int j = 0; j < nImage; j++) {
				memcpy(&hImages[j*COIMX*COIMY], blur_vec[LAYERNUM - 1 - layer][j].data, sizeof(uchar)*COIMY*COIMX);
				memcpy(&hUg[j*COIMX*COIMY], hUg_vec[LAYERNUM - 1 - layer][j].data, sizeof(short)*COIMY*COIMX);
				memcpy(&hVg[j*COIMX*COIMY], hVg_vec[LAYERNUM - 1 - layer][j].data, sizeof(short)*COIMY*COIMX);
			}
			checkCudaErrors(cudaMemcpy(dImages, hImages, sizeof(uchar)*COIMY*COIMX*nImage, cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(dUg, hUg, sizeof(short)*COIMY*COIMX*nImage, cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(dVg, hVg, sizeof(short)*COIMY*COIMX*nImage, cudaMemcpyHostToDevice));
			layerup = false;
		}

		float preEnerge = 0;
		for (int i = 0; i < nVertex; i++) {
			preEnerge += hEnergy[i];
		}
		printf("layer %d, iteration %d - ", layer, i + 1);
		//timer = sample_host(COIMX, COIMY, nVertex, nTriangle, hEnergy, i);
		timer = sample_host_sel(COIMX, COIMY, nVertex, nTriangle, hEnergy, i);
		float layer_time = sdkGetAverageTimerValue(&timer) / 1000.0f;
		printf("elapsed time: %fms\n", layer_time * 1000);

		float sum_energy = 0;
		for (int i = 0; i < nVertex; i++) {
			sum_energy += hEnergy[i];
		}
		if (abs(sum_energy - preEnerge) / sum_energy < 1e-5) {
			break;
		}

	}
	preIter += iteration;
	return iteration - till_iter;
}

void Optimizer::Update_toAtlas(int iteration, int layer) {
	for (int i = 0; i < iteration; i++) {
		if (i == 0) {
			for (int j = 0; j < nImage; j++) {
				memcpy(&hImages[j * COIMX * COIMY], blur_vec[LAYERNUM - 1 - layer][j].data, sizeof(uchar) * COIMY * COIMX);
				memcpy(&hUg[j * COIMX * COIMY], hUg_vec[LAYERNUM - 1 - layer][j].data, sizeof(short) * COIMY * COIMX);
				memcpy(&hVg[j * COIMX * COIMY], hVg_vec[LAYERNUM - 1 - layer][j].data, sizeof(short) * COIMY * COIMX);
			}
			checkCudaErrors(cudaMemcpy(dImages, hImages, sizeof(uchar) * COIMY * COIMX * nImage, cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(dUg, hUg, sizeof(short) * COIMY * COIMX * nImage, cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(dVg, hVg, sizeof(short) * COIMY * COIMX * nImage, cudaMemcpyHostToDevice));
		}

		float preEnerge = 0;
		for (int i = 0; i < nVertex; i++) {
			preEnerge += hEnergy[i];
		}
		printf("layer %d, iteration %d - ", layer, i + 1);
		timer = sample_host_toAtlas(COIMX, COIMY, nVertex, nTriangle, hEnergy);
		float layer_time = sdkGetAverageTimerValue(&timer) / 1000.0f;
		printf("elapsed time: %fms\n", layer_time * 1000);

		float sum_energy = 0;
		for (int i = 0; i < nVertex; i++) {
			sum_energy += hEnergy[i];
		}
		if (abs(sum_energy - preEnerge) / sum_energy < 1e-5) {
			break;
		}
	}
}

void Optimizer::PrepareRend_UVAtlas() {

	if (atlas_coord == NULL) {
		atlas_coord = new vector<float2>[nImage];
	}
	if (atlas_tri_idx == NULL) {
		atlas_tri_idx = new vector<int>[nImage];
	}
	if (img_coord == NULL) {
		img_coord = new vector<float3>[nImage];
	}
	for (int i = 0; i < nImage; i++) {
		atlas_coord[i].clear();
		atlas_tri_idx[i].clear();
		img_coord[i].clear();
	}
	
	int f_idx = 0;
	for (int f_idx = 0; f_idx < nTriangle; f_idx++) {
		hostVertex *hV[3];
		float2 tmp_at_coord[3] = { 0 };
		for (int i = 0; i < 3; i++)
			hV[i] = &hVertices[hTriangles[f_idx]._Vertices[i]];
		for (int i = 0; i < hTriangles[f_idx]._Img.size(); i++) {
			int vIm[3] = { -1,-1,-1 };
			int imgidx = hTriangles[f_idx]._Img[i];
			float weight = hTriangles[f_idx]._Img_Weight[i];
			/*float weight = 0.0;
			if (hTriangles[f_idx].selected_img == i)
				weight = 1.0;*/
			for (int j = 0; j < 3; j++) {
				vIm[j] = hV[j]->getImgIdx(imgidx);
				atlas_coord[imgidx].push_back(tmp_at_coord[j]);
				img_coord[imgidx].push_back(make_float3(hV[j]->_Img_Tex[vIm[j]].x, hV[j]->_Img_Tex[vIm[j]].y, weight));
			}
			atlas_tri_idx[imgidx].push_back(f_idx);
			if (vIm[0] < 0 || vIm[1] < 0 || vIm[2] < 0)	cout << "Mapping wrong!!!";
		}
	}
}
vector<bool> Optimizer::PrepareRend_UVAtlas_video(int targetIdx) {

	vector<bool> validImg(nImage, false);

	if (atlas_coord == NULL) {
		atlas_coord = new vector<float2>[nImage];
	}
	if (atlas_tri_idx == NULL) {
		atlas_tri_idx = new vector<int>[nImage];
	}
	if (img_coord == NULL) {
		img_coord = new vector<float3>[nImage];
	}
	for (int i = 0; i < nImage; i++) {
		atlas_coord[i].clear();
		atlas_tri_idx[i].clear();
		img_coord[i].clear();
	}

	int f_idx = 0;
	for (int f_idx = 0; f_idx < nTriangle; f_idx++) {
		hostVertex* hV[3];
		float2 tmp_at_coord[3] = { 0 };
		for (int i = 0; i < 3; i++)
			hV[i] = &hVertices[hTriangles[f_idx]._Vertices[i]];
			//hV[i] = &time_hVertices[targetIdx][hTriangles[f_idx]._Vertices[i]];
		for (int i = 0; i < hTriangles[f_idx]._Img.size(); i++) {
			int vIm[3] = { -1,-1,-1 };
			int imgidx = hTriangles[f_idx]._Img[i];
			//float weight = hTriangles[f_idx]._Img_Weight[i];
			float weight = 0.0;
			if (selector_ptr->getLabel(targetIdx, f_idx) == hTriangles[f_idx]._Img[i]) {
				weight = 1.0;
				validImg[hTriangles[f_idx]._Img[i]] = true;
			}
			for (int j = 0; j < 3; j++) {
				vIm[j] = hV[j]->getImgIdx(imgidx);
				atlas_coord[imgidx].push_back(tmp_at_coord[j]);
				img_coord[imgidx].push_back(make_float3(hV[j]->_Img_Tex[vIm[j]].x, hV[j]->_Img_Tex[vIm[j]].y, weight));
			}
			atlas_tri_idx[imgidx].push_back(f_idx);
			if (vIm[0] < 0 || vIm[1] < 0 || vIm[2] < 0)	cout << "Mapping wrong!!!";
		}
	}
	return validImg;
}

vector<bool> Optimizer::PrepareRend_UVAtlas_video_mask(int targetIdx) {

	vector<bool> validImg(nImage, false);

	if (atlas_coord == NULL) {
		atlas_coord = new vector<float2>[nImage];
	}
	if (atlas_tri_idx == NULL) {
		atlas_tri_idx = new vector<int>[nImage];
	}
	if (img_coord_mask == NULL) {
		img_coord_mask = new vector<float3>[nImage];
	}
	for (int i = 0; i < nImage; i++) {
		atlas_coord[i].clear();
		atlas_tri_idx[i].clear();
		img_coord_mask[i].clear();
	}

	int f_idx = 0;
	for (int f_idx = 0; f_idx < nTriangle; f_idx++) {
		hostVertex* hV[3];
		float2 tmp_at_coord[3] = { 0 };
		for (int i = 0; i < 3; i++)
			hV[i] = &hVertices[hTriangles[f_idx]._Vertices[i]];
		for (int i = 0; i < hTriangles[f_idx]._Img.size(); i++) {
			int vIm[3] = { -1,-1,-1 };
			int imgidx = hTriangles[f_idx]._Img[i];
			//float weight = hTriangles[f_idx]._Img_Weight[i];
			float weight = 2.0;
			if (selector_ptr->getLabel(targetIdx, f_idx) == hTriangles[f_idx]._Img[i]) {
				weight = imgidx + 3;
				validImg[hTriangles[f_idx]._Img[i]] = true;
			}
			for (int j = 0; j < 3; j++) {
				vIm[j] = hV[j]->getImgIdx(imgidx);
				atlas_coord[imgidx].push_back(tmp_at_coord[j]);
				img_coord_mask[imgidx].push_back(make_float3(hV[j]->_Img_Tex[vIm[j]].x, hV[j]->_Img_Tex[vIm[j]].y, weight));
			}
			atlas_tri_idx[imgidx].push_back(f_idx);
			if (vIm[0] < 0 || vIm[1] < 0 || vIm[2] < 0)	cout << "Mapping wrong!!!";
		}
	}
	return validImg;
}

void Optimizer::GetAtlasInfoi_UVAtlas(vector<float> *_uv, vector<float> *_uvImg, vector<int> *_triIdx, int idx) {
	_uv->resize(2 * atlas_coord[idx].size());
	_uvImg->resize(3 * img_coord[idx].size());
	_triIdx->resize(atlas_tri_idx[idx].size());
	memcpy(_uv->data(), atlas_coord[idx].data(), sizeof(float2)*atlas_coord[idx].size());
	memcpy(_uvImg->data(), img_coord[idx].data(), sizeof(float3)*img_coord[idx].size());
	memcpy(_triIdx->data(), atlas_tri_idx[idx].data(), sizeof(int)*atlas_tri_idx[idx].size());
}

void Optimizer::GetAtlasInfoi_UVAtlas_mask(vector<float>* _uv, vector<float>* _uvImg, vector<int>* _triIdx, int idx) {
	_uv->resize(2 * atlas_coord[idx].size());
	_uvImg->resize(3 * img_coord_mask[idx].size());
	_triIdx->resize(atlas_tri_idx[idx].size());
	memcpy(_uv->data(), atlas_coord[idx].data(), sizeof(float2) * atlas_coord[idx].size());
	memcpy(_uvImg->data(), img_coord_mask[idx].data(), sizeof(float3) * img_coord_mask[idx].size());
	memcpy(_triIdx->data(), atlas_tri_idx[idx].data(), sizeof(int) * atlas_tri_idx[idx].size());
}

void Optimizer::GetNumber(uint *nT, uint *nV) {
	*nT = nTriangle;
	*nV = nVertex;
}
void Optimizer::GetNumber(uint *nI, uint *w, uint *h) {
	*nI = nImage;
	*w = COIMX;
	*h = COIMY;
}

void TexMap::Optimizer::SetSelector(Selector* selector)
{
	this->selector_ptr = selector;
}

void TexMap::Optimizer::LoadModel4D(Mapper4D* mapper4D, bool use_keyframe) {
	mapper4D_ptr = mapper4D;
	mapper4D_ptr->GetNumInfo(&nVertex, &nTriangle, &nImage);
	mapper4D_ptr->Get_VT_layer(hVertices, hTriangles, LAYERNUM - 1);
	if (use_keyframe) {
		nImage = mapper4D_ptr->validFrame_table.size();
		mapper4D_ptr->Get_VT_layer_key(hVertices, hTriangles, LAYERNUM - 1);
	}
	mapper4D_ptr->SetValidFrame(use_keyframe);
}

void Optimizer::Model4DLoadandMultiUpdate(Mapper4D* mapper4D, string streamPath) {
	mapper4D_ptr = mapper4D;
	All_img_coord = new vector<float3>[mapper4D_ptr->colorImages.size()];
	all_imgNum = mapper4D_ptr->colorImages.size();
	nVertex = 0;
	nTriangle = 0;
	nImage = 0;

	mapper4D_ptr->GetNumInfo(&nVertex, &nTriangle, &nImage);

	//rgbImages = new cv::Mat4b[nImage];
	//depthImages = new cv::Mat1w[nImage];
	float * poses = new float[16 * nImage];

	if (FROM_FILE) {
		std::ifstream imageIn(streamPath, std::ios::in | std::ios::binary);
		for (int i = 0; i < nImage; i++) {
			//rgbImages[i].create(COIMY, COIMX);
			/*depthImages[i].create(IRIMY, IRIMX);
			imageIn.read((char*)rgbImages[i].data, sizeof(uchar) * 4 * COIMX*COIMY);
			cv::cvtColor(rgbImages[i], rgbImages[i], CV_RGBA2BGRA);
			imageIn.read((char*)depthImages[i].data, sizeof(ushort) * IRIMX*IRIMY);
			imageIn.read((char*)&poses[i * 16], sizeof(float) * 16);*/
		}
		imageIn.close();
	}
	else {

		for (int i = 0; i < nImage; i++) {
			//rgbImages[i].create(COIMY, COIMX);
			//depthImages[i].create(IRIMY, IRIMX);
			//cv::cvtColor(mapper4D_ptr->colorImages[i], rgbImages[i], CV_BGR2BGRA);
			//depthImages[i] = mapper4D_ptr->depthImages[i].clone();
			memcpy(&poses[i * 16], mapper4D_ptr->_Pose.data, sizeof(float) * 16);
		}

	}

	blur_vec.resize(LAYERNUM);
	hUg_vec.resize(LAYERNUM);
	hVg_vec.resize(LAYERNUM);

	for (int i = 0; i < nImage; i++) {
		cv::Mat3b tmpImage = mapper4D_ptr->colorImages[i].clone();
		cv::GaussianBlur(tmpImage, tmpImage, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
		for (int k = 0; k < LAYERNUM; k++) {
			cv::Mat1b grayImage;
			cv::Mat1s Ug, Vg;
			cv::cvtColor(tmpImage, grayImage, CV_BGR2GRAY);
			cv::Sobel(grayImage, Ug, CV_16S, 1, 0);
			cv::Sobel(grayImage, Vg, CV_16S, 0, 1);
			blur_vec[k].push_back(grayImage.clone());
			hUg_vec[k].push_back(Ug.clone());
			hVg_vec[k].push_back(Vg.clone());
			cv::GaussianBlur(tmpImage, tmpImage, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT);
		}
	}
	printf("Stream Loading done...\n");

	for (int i = 0; i < mapper4D_ptr->colorImages.size(); i++) {
		All_img_coord[i].resize(nTriangle * 3, { 0,0,0 });
	}

	vector<vector<float2>> propVec;
	propVec.resize(nImage);

	int iter_layer = OPT_ITER;
	int remain_pre_layer = 0;

	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);

	for (int layer = 0; layer < LAYERNUM; layer++) {
		mapper4D_ptr->GetNumInfo_layer(&nVertex, &nTriangle, layer);
		if (layer > 0) {
			/*hVertices.clear();
			hTriangles.clear();*/
			mapper4D_ptr->SetPropVec(propVec, layer);
		}
		float2 dudvInit;
		dudvInit.x = 0.0;
		dudvInit.y = 0.0;
		for (int i = 0; i < nImage; i++) {
			propVec[i].clear();
			propVec[i].resize(nVertex, dudvInit);
		}

		cout << "layer " << layer << " - Face : " << nTriangle << endl;

		mapper4D_ptr->Get_VT_layer(hVertices, hTriangles, layer);

		for (int i = 0; i < nVertex; i++) {
			for (int j = 0; j < hVertices[i]._Img.size(); j++) {
				propVec[hVertices[i]._Img[j]][i] -= hVertices[i]._Img_Tex[j];
			}
		}

		printf("Vertices #: %d\n", nVertex);
		printf("Triangles #: %d\n", nTriangle);
		printf("Images #: %d\n", nImage);
		printf("Model Loading done...\n");
		Initialize(layer);
		if (layer < LAYERNUM-1) {
			if(opt_mode == "multi")
				remain_pre_layer = Update(iter_layer, layer);
		}
		else {
			if (opt_mode == "naive") {}
			else if (opt_mode == "single")
				remain_pre_layer = Update(iter_layer * LAYERNUM, layer);
			else
				remain_pre_layer = Update(iter_layer, layer);
			// remain_pre_layer = Update(iter_layer + remain_pre_layer, layer);
		}
		
		//checkCudaErrors(cudaMemcpy(hVertices, dVertices, sizeof(Vertex)*nVertex, cudaMemcpyDeviceToHost));
		//checkCudaErrors(cudaMemcpy(hTriangles, dTriangles, sizeof(Triangle)*nTriangle, cudaMemcpyDeviceToHost));
		//checkCudaErrors();

		int tmp_imgOffset = 0;
		for (int i = 0; i < nVertex; i++) {
			checkCudaErrors(cudaMemcpy(hVertices[i]._Img_Tex.data(), &dmhVertices._Img_Tex[tmp_imgOffset], sizeof(float2)*hVertices[i]._Img.size(), cudaMemcpyDeviceToHost));
			tmp_imgOffset += hVertices[i]._Img.size();
		}

		for (int i = 0; i < nVertex; i++) {
			for (int j = 0; j < hVertices[i]._Img.size(); j++) {
				propVec[hVertices[i]._Img[j]][i] += hVertices[i]._Img_Tex[j];
			}
		}

	}
}

void Optimizer::Model4DLoadandSelectUpdate(Mapper4D* mapper4D) {
	mapper4D_ptr = mapper4D;
	All_img_coord = new vector<float3>[mapper4D_ptr->colorImages.size()];
	all_imgNum = mapper4D_ptr->colorImages.size();
	nVertex = 0;
	nTriangle = 0;
	nImage = 0;

	mapper4D_ptr->GetNumInfo(&nVertex, &nTriangle, &nImage);

	//rgbImages = new cv::Mat4b[nImage];
	//depthImages = new cv::Mat1w[nImage];
	float* poses = new float[16 * nImage];


	for (int i = 0; i < nImage; i++) {
		//rgbImages[i].create(COIMY, COIMX);
		//depthImages[i].create(IRIMY, IRIMX);
		//cv::cvtColor(mapper4D_ptr->colorImages[i], rgbImages[i], CV_BGR2BGRA);
		//depthImages[i] = mapper4D_ptr->depthImages[i].clone();
		memcpy(&poses[i * 16], mapper4D_ptr->_Pose.data, sizeof(float) * 16);
	}


	blur_vec.resize(LAYERNUM);
	hUg_vec.resize(LAYERNUM);
	hVg_vec.resize(LAYERNUM);

	for (int i = 0; i < nImage; i++) {
		cv::Mat3b tmpImage = mapper4D_ptr->colorImages[i].clone();
		cv::GaussianBlur(tmpImage, tmpImage, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
		for (int k = 0; k < LAYERNUM; k++) {
			cv::Mat1b grayImage;
			cv::Mat1s Ug, Vg;
			cv::cvtColor(tmpImage, grayImage, CV_BGR2GRAY);
			cv::Sobel(grayImage, Ug, CV_16S, 1, 0);
			cv::Sobel(grayImage, Vg, CV_16S, 0, 1);
			blur_vec[k].push_back(grayImage.clone());
			hUg_vec[k].push_back(Ug.clone());
			hVg_vec[k].push_back(Vg.clone());
			cv::GaussianBlur(tmpImage, tmpImage, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
		}
	}
	printf("Stream Loading done...\n");

	for (int i = 0; i < mapper4D_ptr->colorImages.size(); i++) {
		All_img_coord[i].resize(nTriangle * 3, { 0,0,0 });
	}

	vector<vector<float2>> propVec;
	propVec.resize(nImage);

	int iter_layer = OPT_ITER;
	int remain_pre_layer = 0;

	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);

	for (int layer = 0; layer < LAYERNUM; layer++) {
		mapper4D_ptr->GetNumInfo_layer(&nVertex, &nTriangle, layer);
		if (layer > 0) {
			/*hVertices.clear();
			hTriangles.clear();*/
			mapper4D_ptr->SetPropVec(propVec, layer);
		}
		float2 dudvInit;
		dudvInit.x = 0.0;
		dudvInit.y = 0.0;
		for (int i = 0; i < nImage; i++) {
			propVec[i].clear();
			propVec[i].resize(nVertex, dudvInit);
		}

		cout << "layer " << layer << " - Face : " << nTriangle << endl;

		mapper4D_ptr->Get_VT_layer(hVertices, hTriangles, layer);

		for (int i = 0; i < nVertex; i++) {
			for (int j = 0; j < hVertices[i]._Img.size(); j++) {
				propVec[hVertices[i]._Img[j]][i] -= hVertices[i]._Img_Tex[j];
			}
		}

		printf("Vertices #: %d\n", nVertex);
		printf("Triangles #: %d\n", nTriangle);
		printf("Images #: %d\n", nImage);
		printf("Model Loading done...\n");
		Initialize(layer);
		if (layer < LAYERNUM - 1) {
			if (opt_mode == "multi")
				remain_pre_layer = Update(iter_layer, layer);
		}
		
		else {
			if (opt_mode == "naive") {}
			else if (opt_mode == "single")
				remain_pre_layer = Update(iter_layer * LAYERNUM, layer);
			else
				remain_pre_layer = Update(iter_layer, layer);
			// remain_pre_layer = Update(iter_layer + remain_pre_layer, layer);
		}

		//checkCudaErrors(cudaMemcpy(hVertices, dVertices, sizeof(Vertex)*nVertex, cudaMemcpyDeviceToHost));
		//checkCudaErrors(cudaMemcpy(hTriangles, dTriangles, sizeof(Triangle)*nTriangle, cudaMemcpyDeviceToHost));
		//checkCudaErrors();

		int tmp_imgOffset = 0;
		for (int i = 0; i < nVertex; i++) {
			checkCudaErrors(cudaMemcpy(hVertices[i]._Img_Tex.data(), &dmhVertices._Img_Tex[tmp_imgOffset], sizeof(float2) * hVertices[i]._Img.size(), cudaMemcpyDeviceToHost));
			tmp_imgOffset += hVertices[i]._Img.size();
		}

		for (int i = 0; i < nVertex; i++) {
			for (int j = 0; j < hVertices[i]._Img.size(); j++) {
				propVec[hVertices[i]._Img[j]][i] += hVertices[i]._Img_Tex[j];
			}
		}

		//mapper4D_ptr->Push_VT_layer(hVertices, hTriangles, layer);
	}
}

void Optimizer::Model4DLoadandMultiUpdate(Mapper4D* mapper4D) {
	mapper4D_ptr = mapper4D;
	All_img_coord = new vector<float3>[mapper4D_ptr->colorImages.size()];
	all_imgNum = mapper4D_ptr->colorImages.size();
	nVertex = 0;
	nTriangle = 0;
	nImage = 0;

	mapper4D_ptr->GetNumInfo(&nVertex, &nTriangle, &nImage);

	blur_vec.resize(LAYERNUM);
	hUg_vec.resize(LAYERNUM);
	hVg_vec.resize(LAYERNUM);

	for (int i = 0; i < nImage; i++) {
		cv::Mat3b tmpImage = mapper4D_ptr->colorImages[i].clone();
		cv::GaussianBlur(tmpImage, tmpImage, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
		for (int k = 0; k < LAYERNUM; k++) {
			cv::Mat1b grayImage;
			cv::Mat1s Ug, Vg;
			cv::cvtColor(tmpImage, grayImage, CV_BGR2GRAY);
			cv::Sobel(grayImage, Ug, CV_16S, 1, 0);
			cv::Sobel(grayImage, Vg, CV_16S, 0, 1);
			blur_vec[k].push_back(grayImage.clone());
			hUg_vec[k].push_back(Ug.clone());
			hVg_vec[k].push_back(Vg.clone());
			cv::GaussianBlur(tmpImage, tmpImage, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
		}
	}
	printf("Stream Loading done...\n");

	for (int i = 0; i < mapper4D_ptr->colorImages.size(); i++) {
		All_img_coord[i].resize(nTriangle * 3, { 0,0,0 });
	}

	propVec.resize(nImage);

	int iter_layer = OPT_ITER;
	int remain_pre_layer = 0;

	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);

	for (int layer = 0; layer < LAYERNUM; layer++) {
		mapper4D_ptr->GetNumInfo_layer(&nVertex, &nTriangle, layer);
		if (layer > 0) {
			/*hVertices.clear();
			hTriangles.clear();*/
			mapper4D_ptr->SetPropVec(propVec, layer);
		}
		cout << "layer " << layer << " - Face : " << nTriangle << endl;

		mapper4D_ptr->Get_VT_layer(hVertices, hTriangles, layer);

		printf("Vertices #: %d\n", nVertex);
		printf("Triangles #: %d\n", nTriangle);
		printf("Images #: %d\n", nImage);
		printf("Model Loading done...\n");
		Initialize(layer);
		if (layer < LAYERNUM) {
			float2 dudvInit;
			dudvInit.x = 0.0;
			dudvInit.y = 0.0;
			for (int i = 0; i < nImage; i++) {
				propVec[i].clear();
				propVec[i].resize(nVertex, dudvInit);
			}


			for (int i = 0; i < nVertex; i++) {
				for (int j = 0; j < hVertices[i]._Img.size(); j++) {
					propVec[hVertices[i]._Img[j]][i] -= hVertices[i]._Img_Tex[j];
				}
			}
			dmhTriangles.update_bound(true);
			Update(iter_layer, layer);

			int tmp_imgOffset = 0;
			for (int i = 0; i < nVertex; i++) {
				checkCudaErrors(cudaMemcpy(hVertices[i]._Img_Tex.data(), &dmhVertices._Img_Tex[tmp_imgOffset], sizeof(float2) * hVertices[i]._Img.size(), cudaMemcpyDeviceToHost));
				tmp_imgOffset += hVertices[i]._Img.size();
			}

			for (int i = 0; i < nVertex; i++) {
				for (int j = 0; j < hVertices[i]._Img.size(); j++) {
					propVec[hVertices[i]._Img[j]][i] += hVertices[i]._Img_Tex[j];
				}
			}
		}
		
		mapper4D_ptr->Push_VT_layer(hVertices, hTriangles, layer);
	}
}

void Optimizer::Model4DLoadandMultiUpdate_key(Mapper4D* mapper4D) {
	mapper4D_ptr = mapper4D;
	nVertex = 0;
	nTriangle = 0;
	nImage = 0;

	mapper4D_ptr->GetNumInfo(&nVertex, &nTriangle, &nImage);
	nImage = mapper4D_ptr->keyFrame_idx.size();

	blur_vec.resize(LAYERNUM);
	hUg_vec.resize(LAYERNUM);
	hVg_vec.resize(LAYERNUM);

	for (int i = 0; i < nImage; i++) {
		cv::Mat3b tmpImage = mapper4D_ptr->colorImages[mapper4D_ptr->keyFrame_idx[i]].clone();
		cv::GaussianBlur(tmpImage, tmpImage, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
		for (int k = 0; k < LAYERNUM; k++) {
			cv::Mat1b grayImage;
			cv::Mat1s Ug, Vg;
			cv::cvtColor(tmpImage, grayImage, CV_BGR2GRAY);
			cv::Sobel(grayImage, Ug, CV_16S, 1, 0);
			cv::Sobel(grayImage, Vg, CV_16S, 0, 1);
			blur_vec[k].push_back(grayImage.clone());
			hUg_vec[k].push_back(Ug.clone());
			hVg_vec[k].push_back(Vg.clone());
			cv::GaussianBlur(tmpImage, tmpImage, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
		}
	}
	printf("Stream Loading done...\n");

	propVec.resize(nImage);

	int iter_layer = OPT_ITER;
	int remain_pre_layer = 0;

	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);

	h_samcol.resize(LAYERNUM);

	for (int layer = 0; layer < LAYERNUM; layer++) {
		mapper4D_ptr->GetNumInfo_layer(&nVertex, &nTriangle, layer);
		if (layer > 0) {
			mapper4D_ptr->SetPropVec_key(propVec, layer);
		}
		cout << "layer " << layer << " - Face : " << nTriangle << endl;

		mapper4D_ptr->Get_VT_layer_key(hVertices, hTriangles, layer);

		printf("Vertices #: %d\n", nVertex);
		printf("Triangles #: %d\n", nTriangle);
		printf("Images #: %d\n", nImage);
		printf("Model Loading done...\n");
		Initialize(layer);
		float2 dudvInit;
		dudvInit.x = 0.0;
		dudvInit.y = 0.0;
		for (int i = 0; i < nImage; i++) {
			propVec[i].clear();
			propVec[i].resize(nVertex, dudvInit);
		}


		for (int i = 0; i < nVertex; i++) {
			for (int j = 0; j < hVertices[i]._Img.size(); j++) {
				propVec[hVertices[i]._Img[j]][i] -= hVertices[i]._Img_Tex[j];
			}
		}
		dmhTriangles.update_bound(true);
		Update(iter_layer, layer);

		int tmp_imgOffset = 0;
		for (int i = 0; i < nVertex; i++) {
			checkCudaErrors(cudaMemcpy(hVertices[i]._Img_Tex.data(), &dmhVertices._Img_Tex[tmp_imgOffset], sizeof(float2) * hVertices[i]._Img.size(), cudaMemcpyDeviceToHost));
			tmp_imgOffset += hVertices[i]._Img.size();
		}
		h_samcol[layer].resize(nTriangle * SAMNUM);
		checkCudaErrors(cudaMemcpy(h_samcol[layer].data(), &dmhTriangles._SamCol[0], sizeof(unsigned char) * nTriangle * SAMNUM, cudaMemcpyDeviceToHost));

		for (int i = 0; i < nVertex; i++) {
			for (int j = 0; j < hVertices[i]._Img.size(); j++) {
				propVec[hVertices[i]._Img[j]][i] += hVertices[i]._Img_Tex[j];
			}
		}
		mapper4D_ptr->Push_VT_layer_key(hVertices, hTriangles, layer);
	}
	//mapper4D_ptr->BackgroundSubtraction(true);
	mapper4D_ptr->Get_VT_layer_key(hVertices, hTriangles, LAYERNUM - 1);
}

void Optimizer::Model4DLoadandMultiUpdate_all(Mapper4D* mapper4D) {
	mapper4D_ptr = mapper4D;
	nVertex = 0;
	nTriangle = 0;
	nImage = 0;

	mapper4D_ptr->GetNumInfo(&nVertex, &nTriangle, &nImage);

	blur_vec.resize(LAYERNUM);
	hUg_vec.resize(LAYERNUM);
	hVg_vec.resize(LAYERNUM);

	for (int i = 0; i < nImage; i++) {
		cv::Mat3b tmpImage = mapper4D_ptr->colorImages[i].clone();
		cv::GaussianBlur(tmpImage, tmpImage, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
		for (int k = 0; k < LAYERNUM; k++) {
			cv::Mat1b grayImage;
			cv::Mat1s Ug, Vg;
			cv::cvtColor(tmpImage, grayImage, CV_BGR2GRAY);
			cv::Sobel(grayImage, Ug, CV_16S, 1, 0);
			cv::Sobel(grayImage, Vg, CV_16S, 0, 1);
			blur_vec[k].push_back(grayImage.clone());
			hUg_vec[k].push_back(Ug.clone());
			hVg_vec[k].push_back(Vg.clone());
			cv::GaussianBlur(tmpImage, tmpImage, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
		}
	}
	printf("Stream Loading done...\n");

	propVec.resize(nImage);

	int iter_layer = OPT_ITER;
	int remain_pre_layer = 0;

	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);

	for (int layer = 0; layer < LAYERNUM; layer++) {
		mapper4D_ptr->GetNumInfo_layer(&nVertex, &nTriangle, layer);
		if (layer > 0) {
			mapper4D_ptr->SetPropVec(propVec, layer);
		}
		cout << "layer " << layer << " - Face : " << nTriangle << endl;

		mapper4D_ptr->Get_VT_layer(hVertices, hTriangles, layer);

		printf("Vertices #: %d\n", nVertex);
		printf("Triangles #: %d\n", nTriangle);
		printf("Images #: %d\n", nImage);
		printf("Model Loading done...\n");
		Initialize(layer);
		float2 dudvInit;
		dudvInit.x = 0.0;
		dudvInit.y = 0.0;
		for (int i = 0; i < nImage; i++) {
			propVec[i].clear();
			propVec[i].resize(nVertex, dudvInit);
		}

		for (int i = 0; i < nVertex; i++) {
			for (int j = 0; j < hVertices[i]._Img.size(); j++) {
				propVec[hVertices[i]._Img[j]][i] -= hVertices[i]._Img_Tex[j];
			}
		}
		dmhTriangles.update_bound(true);
		Update(iter_layer, layer);

		int tmp_imgOffset = 0;
		for (int i = 0; i < nVertex; i++) {
			checkCudaErrors(cudaMemcpy(hVertices[i]._Img_Tex.data(), &dmhVertices._Img_Tex[tmp_imgOffset], sizeof(float2) * hVertices[i]._Img.size(), cudaMemcpyDeviceToHost));
			tmp_imgOffset += hVertices[i]._Img.size();
		}

		for (int i = 0; i < nVertex; i++) {
			for (int j = 0; j < hVertices[i]._Img.size(); j++) {
				propVec[hVertices[i]._Img[j]][i] += hVertices[i]._Img_Tex[j];
			}
		}
		mapper4D_ptr->Push_VT_layer(hVertices, hTriangles, layer);
	}
	//mapper4D_ptr->BackgroundSubtraction(false);
	mapper4D_ptr->Get_VT_layer(hVertices, hTriangles, LAYERNUM - 1);
}

void Optimizer::Model4DLoadandMultiRefine_all(Mapper4D* mapper4D) {
	mapper4D_ptr = mapper4D;
	nVertex = 0;
	nTriangle = 0;
	nImage = 0;

	mapper4D_ptr->GetNumInfo(&nVertex, &nTriangle, &nImage);

	blur_vec.resize(1);
	hUg_vec.resize(1);
	hVg_vec.resize(1);

	for (int i = 0; i < nImage; i++) {
		cv::Mat3b tmpImage = mapper4D_ptr->colorImages[i].clone();
		cv::GaussianBlur(tmpImage, tmpImage, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
		cv::Mat1b grayImage;
		cv::Mat1s Ug, Vg;
		cv::cvtColor(tmpImage, grayImage, CV_BGR2GRAY);
		cv::Sobel(grayImage, Ug, CV_16S, 1, 0);
		cv::Sobel(grayImage, Vg, CV_16S, 0, 1);
		blur_vec[0].push_back(grayImage.clone());
		hUg_vec[0].push_back(Ug.clone());
		hVg_vec[0].push_back(Vg.clone());
	}
	printf("Stream Loading done...\n");

	propVec.resize(nImage);

	int iter_layer = OPT_ITER;
	int remain_pre_layer = 0;

	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);

	mapper4D_ptr->GetNumInfo_layer(&nVertex, &nTriangle, LAYERNUM - 1);
	mapper4D_ptr->Get_VT_layer(hVertices, hTriangles, LAYERNUM - 1);

	printf("Vertices #: %d\n", nVertex);
	printf("Triangles #: %d\n", nTriangle);
	printf("Images #: %d\n", nImage);
	printf("Model Loading done...\n");
	Initialize(0);
	Initialize(LAYERNUM - 1);
	dmhTriangles.update_bound(true);
	Update(iter_layer, LAYERNUM - 1);

	int tmp_imgOffset = 0;
	for (int i = 0; i < nVertex; i++) {
		checkCudaErrors(cudaMemcpy(hVertices[i]._Img_Tex.data(), &dmhVertices._Img_Tex[tmp_imgOffset], sizeof(float2) * hVertices[i]._Img.size(), cudaMemcpyDeviceToHost));
		tmp_imgOffset += hVertices[i]._Img.size();
	}
	mapper4D_ptr->Push_VT_layer(hVertices, hTriangles, LAYERNUM - 1);
	
	//mapper4D_ptr->BackgroundSubtraction(false);
}

void Optimizer::Model4DLoadandMultiUpdate_key_refine(Mapper4D* mapper4D) {
	mapper4D_ptr = mapper4D;
	nVertex = 0;
	nTriangle = 0;
	nImage = 0;

	mapper4D_ptr->GetNumInfo(&nVertex, &nTriangle, &nImage);
	nImage = mapper4D_ptr->keyFrame_idx.size();

	blur_vec.resize(LAYERNUM);
	hUg_vec.resize(LAYERNUM);
	hVg_vec.resize(LAYERNUM);

	for (int i = 0; i < nImage; i++) {
		cv::Mat3b tmpImage = mapper4D_ptr->colorImages[mapper4D_ptr->keyFrame_idx[i]].clone();
		cv::GaussianBlur(tmpImage, tmpImage, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
		for (int k = 0; k < LAYERNUM; k++) {
			cv::Mat1b grayImage;
			cv::Mat1s Ug, Vg;
			cv::cvtColor(tmpImage, grayImage, CV_BGR2GRAY);
			cv::Sobel(grayImage, Ug, CV_16S, 1, 0);
			cv::Sobel(grayImage, Vg, CV_16S, 0, 1);
			blur_vec[k].push_back(grayImage.clone());
			hUg_vec[k].push_back(Ug.clone());
			hVg_vec[k].push_back(Vg.clone());
			cv::GaussianBlur(tmpImage, tmpImage, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
		}
	}
	printf("Stream Loading done...\n");

	propVec.resize(nImage);

	int iter_layer = OPT_ITER;
	int remain_pre_layer = 0;

	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);


	for (int layer = 0; layer < LAYERNUM; layer++) {
		mapper4D_ptr->GetNumInfo_layer(&nVertex, &nTriangle, layer);
		if (layer > 0) {
			mapper4D_ptr->SetPropVec_key(propVec, layer);
		}
		cout << "layer " << layer << " - Face : " << nTriangle << endl;

		mapper4D_ptr->Get_VT_layer_key(hVertices, hTriangles, layer);

		printf("Vertices #: %d\n", nVertex);
		printf("Triangles #: %d\n", nTriangle);
		printf("Images #: %d\n", nImage);
		printf("Model Loading done...\n");
		Initialize(layer);
		float2 dudvInit;
		dudvInit.x = 0.0;
		dudvInit.y = 0.0;
		for (int i = 0; i < nImage; i++) {
			propVec[i].clear();
			propVec[i].resize(nVertex, dudvInit);
		}


		for (int i = 0; i < nVertex; i++) {
			for (int j = 0; j < hVertices[i]._Img.size(); j++) {
				propVec[hVertices[i]._Img[j]][i] -= hVertices[i]._Img_Tex[j];
			}
		}
		dmhTriangles.update_bound(true);
		Update(iter_layer, layer);

		int tmp_imgOffset = 0;
		for (int i = 0; i < nVertex; i++) {
			checkCudaErrors(cudaMemcpy(hVertices[i]._Img_Tex.data(), &dmhVertices._Img_Tex[tmp_imgOffset], sizeof(float2) * hVertices[i]._Img.size(), cudaMemcpyDeviceToHost));
			tmp_imgOffset += hVertices[i]._Img.size();
		}

		for (int i = 0; i < nVertex; i++) {
			for (int j = 0; j < hVertices[i]._Img.size(); j++) {
				propVec[hVertices[i]._Img[j]][i] += hVertices[i]._Img_Tex[j];
			}
		}
		//mapper4D_ptr->Push_VT_layer_key(hVertices, hTriangles, layer);
	}
	mapper4D_ptr->SetPropVec_key(propVec, -1);
	mapper4D_ptr->BackPropVec_key();

	h_samcol.resize(LAYERNUM);
	for (int layer = 0; layer < LAYERNUM; layer++) {
		mapper4D_ptr->GetNumInfo_layer(&nVertex, &nTriangle, layer);
		if (layer > 0) {
			mapper4D_ptr->SetPropVec_key(propVec, layer);
		}
		cout << "refine layer " << layer << " - Face : " << nTriangle << endl;

		mapper4D_ptr->Get_VT_layer_key(hVertices, hTriangles, layer);

		Initialize(layer, true, true);
		float2 dudvInit;
		dudvInit.x = 0.0;
		dudvInit.y = 0.0;
		for (int i = 0; i < nImage; i++) {
			propVec[i].clear();
			propVec[i].resize(nVertex, dudvInit);
		}


		for (int i = 0; i < nVertex; i++) {
			for (int j = 0; j < hVertices[i]._Img.size(); j++) {
				propVec[hVertices[i]._Img[j]][i] -= hVertices[i]._Img_Tex[j];
			}
		}
		dmhTriangles.update_bound(true);
		Update(iter_layer, layer, true);

		int tmp_imgOffset = 0;
		for (int i = 0; i < nVertex; i++) {
			checkCudaErrors(cudaMemcpy(hVertices[i]._Img_Tex.data(), &dmhVertices._Img_Tex[tmp_imgOffset], sizeof(float2) * hVertices[i]._Img.size(), cudaMemcpyDeviceToHost));
			tmp_imgOffset += hVertices[i]._Img.size();
		}

		for (int i = 0; i < nVertex; i++) {
			for (int j = 0; j < hVertices[i]._Img.size(); j++) {
				propVec[hVertices[i]._Img[j]][i] += hVertices[i]._Img_Tex[j];
			}
		}
		h_samcol[layer].resize(nTriangle * SAMNUM);
		checkCudaErrors(cudaMemcpy(h_samcol[layer].data(), &dmhTriangles._SamCol[0], sizeof(unsigned char) * nTriangle * SAMNUM, cudaMemcpyDeviceToHost));
	}

	mapper4D_ptr->BackgroundSubtraction(true);
	mapper4D_ptr->Get_VT_layer_key(hVertices, hTriangles, LAYERNUM - 1);
}

void Optimizer::MultiUpdate_toAtlas() {
	nVertex = 0;
	nTriangle = 0;
	nImage = 0;

	mapper4D_ptr->GetNumInfo(&nVertex, &nTriangle, &nImage);

	blur_vec.resize(LAYERNUM);
	hUg_vec.resize(LAYERNUM);
	hVg_vec.resize(LAYERNUM);

	for (int i = 0; i < nImage; i++) {
		cv::Mat3b tmpImage = mapper4D_ptr->colorImages[i].clone();
		cv::GaussianBlur(tmpImage, tmpImage, cv::Size(7, 7), 0, 0, cv::BORDER_DEFAULT);
		for (int k = 0; k < LAYERNUM; k++) {
			cv::Mat1b grayImage;
			cv::Mat1s Ug, Vg;
			cv::cvtColor(tmpImage, grayImage, CV_BGR2GRAY);
			cv::Sobel(grayImage, Ug, CV_16S, 1, 0);
			cv::Sobel(grayImage, Vg, CV_16S, 0, 1);
			blur_vec[k].push_back(grayImage.clone());
			hUg_vec[k].push_back(Ug.clone());
			hVg_vec[k].push_back(Vg.clone());
			cv::GaussianBlur(tmpImage, tmpImage, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
		}
	}
	printf("Stream Loading done...\n");

	propVec.resize(nImage);

	int iter_layer = OPT_ITER;
	int remain_pre_layer = 0;

	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);

	for (int layer = 0; layer < LAYERNUM; layer++) {
		mapper4D_ptr->GetNumInfo_layer(&nVertex, &nTriangle, layer);
		if (layer > 0) {
			mapper4D_ptr->SetPropVec(propVec, layer);
		}
		cout << "layer " << layer << " - Face : " << nTriangle << endl;

		mapper4D_ptr->Get_VT_layer(hVertices, hTriangles, layer);

		printf("Vertices #: %d\n", nVertex);
		printf("Triangles #: %d\n", nTriangle);
		printf("Images #: %d\n", nImage);
		printf("Model Loading done...\n");
		Initialize(layer, false);

		checkCudaErrors(cudaMemcpy(&dmhTriangles._SamCol[0], h_samcol[layer].data(), sizeof(unsigned char) * nTriangle * SAMNUM, cudaMemcpyHostToDevice));
		float2 dudvInit;
		dudvInit.x = 0.0;
		dudvInit.y = 0.0;
		for (int i = 0; i < nImage; i++) {
			propVec[i].clear();
			propVec[i].resize(nVertex, dudvInit);
		}


		for (int i = 0; i < nVertex; i++) {
			for (int j = 0; j < hVertices[i]._Img.size(); j++) {
				propVec[hVertices[i]._Img[j]][i] -= hVertices[i]._Img_Tex[j];
			}
		}
		dmhTriangles.update_bound(true);
		Update_toAtlas(iter_layer, layer);

		int tmp_imgOffset = 0;
		for (int i = 0; i < nVertex; i++) {
			checkCudaErrors(cudaMemcpy(hVertices[i]._Img_Tex.data(), &dmhVertices._Img_Tex[tmp_imgOffset], sizeof(float2) * hVertices[i]._Img.size(), cudaMemcpyDeviceToHost));
			tmp_imgOffset += hVertices[i]._Img.size();
		}

		for (int i = 0; i < nVertex; i++) {
			for (int j = 0; j < hVertices[i]._Img.size(); j++) {
				propVec[hVertices[i]._Img[j]][i] += hVertices[i]._Img_Tex[j];
			}
		}
		mapper4D_ptr->Push_VT_layer(hVertices, hTriangles, layer);
	}
	mapper4D_ptr->BackgroundSubtraction(false);
	mapper4D_ptr->Get_VT_layer(hVertices, hTriangles, LAYERNUM - 1);
}

void Optimizer::Model4DLoadandSelectEachUpdate(Mapper4D* mapper4D) {
	mapper4D_ptr = mapper4D;
	All_img_coord = new vector<float3>[mapper4D_ptr->colorImages.size()];
	all_imgNum = mapper4D_ptr->colorImages.size();
	nVertex = 0;
	nTriangle = 0;
	nImage = 0;

	mapper4D_ptr->GetNumInfo(&nVertex, &nTriangle, &nImage);

	//rgbImages = new cv::Mat4b[nImage];
	//depthImages = new cv::Mat1w[nImage];
	float* poses = new float[16 * nImage];

	time_hVertices.resize(nImage);

	for (int i = 0; i < nImage; i++) {
		//rgbImages[i].create(COIMY, COIMX);
		//depthImages[i].create(IRIMY, IRIMX);
		//cv::cvtColor(mapper4D_ptr->colorImages[i], rgbImages[i], CV_BGR2BGRA);
		//depthImages[i] = mapper4D_ptr->depthImages[i].clone();
		memcpy(&poses[i * 16], mapper4D_ptr->_Pose.data, sizeof(float) * 16);
	}


	blur_vec.resize(LAYERNUM);
	hUg_vec.resize(LAYERNUM);
	hVg_vec.resize(LAYERNUM);

	for (int i = 0; i < nImage; i++) {
		cv::Mat3b tmpImage = mapper4D_ptr->colorImages[i].clone();
		cv::GaussianBlur(tmpImage, tmpImage, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
		for (int k = 0; k < LAYERNUM; k++) {
			cv::Mat1b grayImage;
			cv::Mat1s Ug, Vg;
			cv::cvtColor(tmpImage, grayImage, CV_BGR2GRAY);
			cv::Sobel(grayImage, Ug, CV_16S, 1, 0);
			cv::Sobel(grayImage, Vg, CV_16S, 0, 1);
			blur_vec[k].push_back(grayImage.clone());
			hUg_vec[k].push_back(Ug.clone());
			hVg_vec[k].push_back(Vg.clone());
			cv::GaussianBlur(tmpImage, tmpImage, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
		}
	}
	printf("Stream Loading done...\n");

	for (int i = 0; i < mapper4D_ptr->colorImages.size(); i++) {
		All_img_coord[i].resize(nTriangle * 3, { 0,0,0 });
	}

	propVec.resize(nImage);
	
	int iter_layer = OPT_ITER;
	int remain_pre_layer = 0;

	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);

	for (int layer = 0; layer < LAYERNUM; layer++) {
		mapper4D_ptr->GetNumInfo_layer(&nVertex, &nTriangle, layer);
		if (layer > 0) {
			/*hVertices.clear();
			hTriangles.clear();*/
			mapper4D_ptr->SetPropVec(propVec, layer);
		}
		cout << "layer " << layer << " - Face : " << nTriangle << endl;

		mapper4D_ptr->Get_VT_layer(hVertices, hTriangles, layer);
		
		printf("Vertices #: %d\n", nVertex);
		printf("Triangles #: %d\n", nTriangle);
		printf("Images #: %d\n", nImage);
		printf("Model Loading done...\n");
		Initialize(layer);
		if (layer < LAYERNUM) {
			float2 dudvInit;
			dudvInit.x = 0.0;
			dudvInit.y = 0.0;
			for (int i = 0; i < nImage; i++) {
				propVec[i].clear();
				propVec[i].resize(nVertex, dudvInit);
			}


			for (int i = 0; i < nVertex; i++) {
				for (int j = 0; j < hVertices[i]._Img.size(); j++) {
					propVec[hVertices[i]._Img[j]][i] -= hVertices[i]._Img_Tex[j];
				}
			}
			//dmhTriangles.update_bound(mapper4D_ptr->Bound_vec[0][0], mapper4D_ptr->Label_vec[0][0]);
			dmhTriangles.update_bound(true);
			Update(iter_layer, layer);

			int tmp_imgOffset = 0;
			for (int i = 0; i < nVertex; i++) {
				checkCudaErrors(cudaMemcpy(hVertices[i]._Img_Tex.data(), &dmhVertices._Img_Tex[tmp_imgOffset], sizeof(float2) * hVertices[i]._Img.size(), cudaMemcpyDeviceToHost));
				tmp_imgOffset += hVertices[i]._Img.size();
			}

			for (int i = 0; i < nVertex; i++) {
				for (int j = 0; j < hVertices[i]._Img.size(); j++) {
					propVec[hVertices[i]._Img[j]][i] += hVertices[i]._Img_Tex[j];
				}
			}
		}
		else {
			for (int t = 0; t < nImage; t++) {
				cout << t << " frame..." << endl;
				Reinit(layer, t);


				Update(iter_layer, layer);

				int tmp_imgOffset = 0;
				for (int i = 0; i < nVertex; i++) {
					checkCudaErrors(cudaMemcpy(hVertices[i]._Img_Tex.data(), &dmhVertices._Img_Tex[tmp_imgOffset], sizeof(float2) * hVertices[i]._Img.size(), cudaMemcpyDeviceToHost));
					tmp_imgOffset += hVertices[i]._Img.size();
				}

				mapper4D_ptr->Get_VT_layer(time_hVertices[t], hTriangles, layer);
			}
		}

		mapper4D_ptr->Push_VT_layer(hVertices, hTriangles, layer);
	}
}