#include "Calculator.cuh"

__global__ void calNodeImweight(int srcIm, int* d_NodeImg, int* d_node_face_size, int* d_node_face_offset, int* d_node_face_arr, int* d_node_vert_size, int* d_node_vert_offset, int* d_node_vert_arr, float3* d_stretchvec, float* d_curvaturevec, float* d_nodeImg_weight, float weight_c, float weight_s, int nImage, int nTriangle, int nVert, int num_node) {
	int nidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (nidx >= num_node) return;

	int face_size = d_node_face_size[nidx];
	int face_offset = d_node_face_offset[nidx];
	int vert_size = d_node_vert_size[nidx];
	int vert_offset = d_node_vert_offset[nidx];

	for (int tarIm = 0; tarIm < nImage; tarIm++) {
		if (d_NodeImg[num_node * tarIm + nidx] < 0) continue;
		float sim_cost_curvature = 0.0;
		float sim_cost_curvature_sum1 = 0.0;
		float sim_cost_curvature_sum2 = 0.0;
		float sim_cost_stretch = 0.0;
		for (int fi = 0; fi < face_size; fi++) {
			sim_cost_stretch += dot(d_stretchvec[srcIm * nTriangle + d_node_face_arr[face_offset + fi]], d_stretchvec[tarIm * nTriangle + d_node_face_arr[face_offset + fi]]) /
				(length(d_stretchvec[srcIm * nTriangle + d_node_face_arr[face_offset + fi]]) * length(d_stretchvec[tarIm * nTriangle + d_node_face_arr[face_offset + fi]]));
			//sim_cost_stretch += abs(length(d_stretchvec[srcIm * nTriangle + d_node_face_arr[face_offset + fi]]) - length(d_stretchvec[tarIm * nTriangle + d_node_face_arr[face_offset + fi]]));
		}
		for (int vi = 0; vi < vert_size; vi++) {
			/*sim_cost_curvature += dot(d_curvaturevec[srcIm * nVert + d_node_vert_arr[vert_offset + vi]], d_curvaturevec[tarIm * nVert + d_node_vert_arr[vert_offset + vi]]) / 
				(length(d_curvaturevec[srcIm * nVert + d_node_vert_arr[vert_offset + vi]]) * length(d_curvaturevec[tarIm * nVert + d_node_vert_arr[vert_offset + vi]]));*/
			//sim_cost_curvature += abs(length(d_curvaturevec[srcIm * nVert + d_node_vert_arr[vert_offset + vi]]) - length(d_curvaturevec[tarIm * nVert + d_node_vert_arr[vert_offset + vi]]));
			sim_cost_curvature += d_curvaturevec[srcIm * nVert + d_node_vert_arr[vert_offset + vi]] * d_curvaturevec[tarIm * nVert + d_node_vert_arr[vert_offset + vi]];
			sim_cost_curvature_sum1 += d_curvaturevec[srcIm * nVert + d_node_vert_arr[vert_offset + vi]] * d_curvaturevec[srcIm * nVert + d_node_vert_arr[vert_offset + vi]];
			sim_cost_curvature_sum2 += d_curvaturevec[tarIm * nVert + d_node_vert_arr[vert_offset + vi]] * d_curvaturevec[tarIm * nVert + d_node_vert_arr[vert_offset + vi]];
		}
		sim_cost_stretch /= face_size;
		//sim_cost_curvature /= vert_size;
		sim_cost_curvature /= powf(sim_cost_curvature_sum1, 0.5) * powf(sim_cost_curvature_sum2, 0.5);
		d_nodeImg_weight[num_node * tarIm + nidx] = weight_c * exp(-sim_cost_curvature) + weight_s * exp(-sim_cost_stretch);
	}
}

__global__ void calNodeImweight2(int srcIm, int* d_NodeImg, int* d_node_face_size, int* d_node_face_offset, int* d_node_face_arr, int* d_node_vert_size, int* d_node_vert_offset, int* d_node_vert_arr, float3* d_stretchvec, float3* d_curvaturevec, float* d_nodeImg_weight, float weight_c, float weight_s, int nImage, int nTriangle, int nVert, int num_node) {
	int nidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (nidx >= num_node) return;

	int face_size = d_node_face_size[nidx];
	int face_offset = d_node_face_offset[nidx];
	int vert_size = d_node_vert_size[nidx];
	int vert_offset = d_node_vert_offset[nidx];

	for (int tarIm = 0; tarIm < nImage; tarIm++) {
		if (d_NodeImg[num_node * tarIm + nidx] < 0) continue;
		float sim_cost_curvature = 0.0;
		float sim_cost_stretch = 0.0;
		for (int fi = 0; fi < face_size; fi++) {
			sim_cost_stretch += dot(d_stretchvec[srcIm * nTriangle + d_node_face_arr[face_offset + fi]], d_stretchvec[tarIm * nTriangle + d_node_face_arr[face_offset + fi]]) /
				(length(d_stretchvec[srcIm * nTriangle + d_node_face_arr[face_offset + fi]]) * length(d_stretchvec[tarIm * nTriangle + d_node_face_arr[face_offset + fi]]));
			//sim_cost_stretch += abs(length(d_stretchvec[srcIm * nTriangle + d_node_face_arr[face_offset + fi]]) - length(d_stretchvec[tarIm * nTriangle + d_node_face_arr[face_offset + fi]]));
		}
		for (int vi = 0; vi < vert_size; vi++) {
			float sim_cost_curvature_tmp = 0.0;
			sim_cost_curvature_tmp += 1 / expf(abs(d_curvaturevec[srcIm * nVert + d_node_vert_arr[vert_offset + vi]].x - d_curvaturevec[tarIm * nVert + d_node_vert_arr[vert_offset + vi]].x));
			sim_cost_curvature_tmp += 1 / expf(abs(d_curvaturevec[srcIm * nVert + d_node_vert_arr[vert_offset + vi]].y - d_curvaturevec[tarIm * nVert + d_node_vert_arr[vert_offset + vi]].y));
			sim_cost_curvature_tmp *= abs(cos(d_curvaturevec[srcIm * nVert + d_node_vert_arr[vert_offset + vi]].z - d_curvaturevec[tarIm * nVert + d_node_vert_arr[vert_offset + vi]].z));
			sim_cost_curvature_tmp *= 0.5;
			sim_cost_curvature += sim_cost_curvature_tmp;
		}
		sim_cost_stretch /= face_size;
		sim_cost_curvature /= vert_size;
		d_nodeImg_weight[num_node * tarIm + nidx] = weight_c * exp(-sim_cost_curvature) + weight_s * exp(-sim_cost_stretch);
	}
}

__global__ void calEdgeImdiff(int srcIm, slEdge *d_edge, int* d_NodeImg, int* d_NodeVer, float2* d_VerImgTex, float* d_edgeImg_diff, uchar* d_image, int num_edge, int nImage, int nNode, int nVert, int w, int h, float weight) {

	int e_samnum = 9;
	
	int eidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (eidx >= num_edge) return;
	slEdge eCurrent = d_edge[eidx];
	if (d_NodeImg[nNode* srcIm+ eCurrent.f1] < 0) return;
	int img_size = h * w;
	for (int tarIm = 0; tarIm < nImage; tarIm++) {
		if (d_NodeImg[nNode * tarIm + eCurrent.f2] < 0) continue;
		float2 f1v1_tex = d_VerImgTex[srcIm * nVert + d_NodeVer[d_edge[eidx].f1 * 3 + d_edge[eidx].v_v1]];
		float2 f1v2_tex = d_VerImgTex[srcIm * nVert + d_NodeVer[d_edge[eidx].f1 * 3 + ((d_edge[eidx].v_v1 + 1) % 3)]];
		float2 f2v1_tex = d_VerImgTex[tarIm * nVert + d_NodeVer[d_edge[eidx].f2 * 3 + d_edge[eidx].v_v2]];
		float2 f2v2_tex = d_VerImgTex[tarIm * nVert + d_NodeVer[d_edge[eidx].f2 * 3 + ((d_edge[eidx].v_v2 + 1) % 3)]];

		float tmp_cost = 0.0;

		float line_diff = 0.0;
		float pre_color1 = 0.0;
		float pre_color2 = 0.0;
		for (int s = 0; s < e_samnum; s++) {
			float ratio = s / (float)(e_samnum - 1);
			/*if (e_samnum == 1)
				ratio = 0.5;
			else
				ratio = s / (e_samnum-1);*/
			int2 smp_tex1 = make_int2(f1v1_tex.x * ratio + f1v2_tex.x * (1.0 - ratio), f1v1_tex.y * ratio + f1v2_tex.y * (1.0 - ratio));
			if (d_edge[eidx].backward)
				ratio = 1.0 - ratio;
			int2 smp_tex2 = make_int2(f2v1_tex.x * ratio + f2v2_tex.x * (1.0 - ratio), f2v1_tex.y * ratio + f2v2_tex.y * (1.0 - ratio));
			tmp_cost += abs((float)d_image[srcIm * img_size + smp_tex1.y * w + smp_tex1.x] - (float)d_image[tarIm * img_size + smp_tex2.y * w + smp_tex2.x]);

			if (s > 0) {
				line_diff += abs(((float)d_image[srcIm * img_size + smp_tex1.y * w + smp_tex1.x] / 255.0 - pre_color1) - ((float)d_image[tarIm * img_size + smp_tex2.y * w + smp_tex2.x] / 255.0 - pre_color2)) / (e_samnum - 1);
			}
			pre_color1 = (float)d_image[srcIm * img_size + smp_tex1.y * w + smp_tex1.x] / 255.0;
			pre_color2 = (float)d_image[tarIm * img_size + smp_tex2.y * w + smp_tex2.x] / 255.0;
			//e.smooth_table_spatial[tmp_count] += abs((float)images[fi1 * size + smp_tex1.y * w + smp_tex1.x] - (float)images[fi2 * size + smp_tex2.y * w + smp_tex2.x]);
		}
		d_edgeImg_diff[num_edge * tarIm + eidx] = weight *(0.5 * (line_diff) +0.5* (tmp_cost / (255.0 * e_samnum)));
		//d_edgeImg_diff[num_edge * tarIm + eidx] = 0;
		//d_edgeImg_diff[num_edge * tarIm + eidx] = (line_diff);
		//printf("%f !!!!!!!\n", d_edgeImg_diff[num_edge * tarIm + eidx]);
	}
}

__global__ void calFaceImdiff(int srcIm, int* d_NodeImg, int* d_NodeVer, float2* d_VerImgTex, float* d_nodeImg_diff, uchar* d_image, int num_node, int nImage, int nNode, int nVert, int w, int h, float weight) {
	int f_samnum = 15; // must be < SAMNUM
	float weight_table[3][SAMNUM] = { 0.5, 0.25, 0.25, 0.8, 0.1, 0.1, 0.5, 0.0, 0.5, 0.2, 0.6, 0.2, 0.1, 0.5, 0.4,
										  0.25, 0.5, 0.25, 0.1, 0.8, 0.1, 0.5, 0.5, 0.0, 0.2, 0.2, 0.6, 0.5, 0.4, 0.1,
										  0.25, 0.25, 0.5, 0.1, 0.1, 0.8, 0.0, 0.5, 0.5, 0.6, 0.2, 0.2, 0.4, 0.1, 0.5 };
	
	int nidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (nidx >= num_node) return;
	if (d_NodeImg[nNode * srcIm + nidx] < 0) return;
	int img_size = h * w;
	for (int tarIm = 0; tarIm < nImage; tarIm++) {
		if (d_NodeImg[nNode * tarIm + nidx] < 0) continue;

		float2 f1v0_tex = d_VerImgTex[srcIm * nVert + d_NodeVer[nidx * 3 + 0]];
		float2 f1v1_tex = d_VerImgTex[srcIm * nVert + d_NodeVer[nidx * 3 + 1]];
		float2 f1v2_tex = d_VerImgTex[srcIm * nVert + d_NodeVer[nidx * 3 + 2]];
		float2 f2v0_tex = d_VerImgTex[tarIm * nVert + d_NodeVer[nidx * 3 + 0]];
		float2 f2v1_tex = d_VerImgTex[tarIm * nVert + d_NodeVer[nidx * 3 + 1]];
		float2 f2v2_tex = d_VerImgTex[tarIm * nVert + d_NodeVer[nidx * 3 + 2]];


		float tmp_cost = 0.0;
		for (int s = 0; s < f_samnum; s++) {
			int2 smp_tex1 = make_int2(f1v0_tex.x * weight_table[0][s] + f1v1_tex.x * weight_table[1][s] + f1v2_tex.x * weight_table[2][s],
				f1v0_tex.y * weight_table[0][s] + f1v1_tex.y * weight_table[1][s] + f1v2_tex.y * weight_table[2][s]);
			int2 smp_tex2 = make_int2(f2v0_tex.x * weight_table[0][s] + f2v1_tex.x * weight_table[1][s] + f2v2_tex.x * weight_table[2][s],
				f2v0_tex.y * weight_table[0][s] + f2v1_tex.y * weight_table[1][s] + f2v2_tex.y * weight_table[2][s]);
			tmp_cost += abs((float)d_image[srcIm * img_size + smp_tex1.y * w + smp_tex1.x] - (float)d_image[tarIm * img_size + smp_tex2.y * w + smp_tex2.x]);
		}
		d_nodeImg_diff[num_node * tarIm + nidx] = weight * ((tmp_cost / (255.0 * f_samnum)));
	}
}

__global__ void calFaceImdiff_all(int srcIm, int* d_TriImg, int* d_TriVer, float2* d_VerImgTex, float* d_triImg_diff, uchar* d_image, int nTriangle, int nImage, int nVert, int w, int h) {
	int f_samnum = SAMNUM; // must be < SAMNUM
	float weight_table[3][SAMNUM] = { 0.5, 0.25, 0.25, 0.8, 0.1, 0.1, 0.5, 0.0, 0.5, 0.2, 0.6, 0.2, 0.1, 0.5, 0.4,
										  0.25, 0.5, 0.25, 0.1, 0.8, 0.1, 0.5, 0.5, 0.0, 0.2, 0.2, 0.6, 0.5, 0.4, 0.1,
										  0.25, 0.25, 0.5, 0.1, 0.1, 0.8, 0.0, 0.5, 0.5, 0.6, 0.2, 0.2, 0.4, 0.1, 0.5 };

	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (tidx >= nTriangle) return;
	if (d_TriImg[nTriangle * srcIm + tidx] < 0) return;
	int img_size = h * w;
	for (int tarIm = 0; tarIm < nImage; tarIm++) {
		if (d_TriImg[nTriangle * tarIm + tidx] < 0) continue;

		float2 f1v0_tex = d_VerImgTex[srcIm * nVert + d_TriVer[tidx * 3 + 0]];
		float2 f1v1_tex = d_VerImgTex[srcIm * nVert + d_TriVer[tidx * 3 + 1]];
		float2 f1v2_tex = d_VerImgTex[srcIm * nVert + d_TriVer[tidx * 3 + 2]];
		float2 f2v0_tex = d_VerImgTex[tarIm * nVert + d_TriVer[tidx * 3 + 0]];
		float2 f2v1_tex = d_VerImgTex[tarIm * nVert + d_TriVer[tidx * 3 + 1]];
		float2 f2v2_tex = d_VerImgTex[tarIm * nVert + d_TriVer[tidx * 3 + 2]];


		float tmp_cost = 0.0;
		for (int s = 0; s < f_samnum; s++) {
			int2 smp_tex1 = make_int2(f1v0_tex.x * weight_table[0][s] + f1v1_tex.x * weight_table[1][s] + f1v2_tex.x * weight_table[2][s],
				f1v0_tex.y * weight_table[0][s] + f1v1_tex.y * weight_table[1][s] + f1v2_tex.y * weight_table[2][s]);
			int2 smp_tex2 = make_int2(f2v0_tex.x * weight_table[0][s] + f2v1_tex.x * weight_table[1][s] + f2v2_tex.x * weight_table[2][s],
				f2v0_tex.y * weight_table[0][s] + f2v1_tex.y * weight_table[1][s] + f2v2_tex.y * weight_table[2][s]);
			tmp_cost += abs((float)d_image[srcIm * img_size + smp_tex1.y * w + smp_tex1.x] - (float)d_image[tarIm * img_size + smp_tex2.y * w + smp_tex2.x]);
		}
		d_triImg_diff[nTriangle * tarIm + tidx] = (tmp_cost / (255.0 * f_samnum));
	}
}

__global__ void calWideImdiff(int* d_valid2all, int* d_edge_face_size, int* d_edge_face_offset, int* d_edge_face_arr, float* d_face_diff_img, float* d_edgeImg_diff, int num_edge, int num_node, int nImage, int nTriangle, float weight) {
	int eidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (eidx >= num_edge) return;

	int face_size = d_edge_face_size[eidx];
	int offset = d_edge_face_offset[eidx];

	for (int tarIm = 0; tarIm < nImage; tarIm++) {
		int valid_count = 0;
		for (int fi = 0; fi < face_size; fi++) {
			int tmp_node = d_edge_face_arr[offset + fi];
			if (d_face_diff_img[nTriangle * tarIm + tmp_node] < 0)
				continue;
			if (d_edgeImg_diff[num_edge * tarIm + eidx] < 0)
				d_edgeImg_diff[num_edge * tarIm + eidx] = 0;
			d_edgeImg_diff[num_edge * tarIm + eidx] += d_face_diff_img[nTriangle * tarIm + tmp_node];
			valid_count++;
		}
		if (valid_count > 0) {
			d_edgeImg_diff[num_edge * tarIm + eidx] /= valid_count;
			d_edgeImg_diff[num_edge * tarIm + eidx] = exp(d_edgeImg_diff[num_edge * tarIm + eidx]);
			d_edgeImg_diff[num_edge * tarIm + eidx] *= weight;
		}
	}
}

__global__ void calWindowImdiff(int srcIm, int* d_valid2all, float* d_face_diff_img, float* d_nodeImg_diff, int num_node, int n_half_window, int nImage, int nTriangle, float weight) {
	int nidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (nidx >= num_node) return;
	
	for (int tarIm = 0; tarIm < nImage; tarIm++) {
		int valid_count = 0;
		for (int offset = 0; offset < 2 * n_half_window; offset++) {
			if (tarIm - n_half_window + offset < 0 || tarIm - n_half_window + offset > nImage - 1)
				continue;
			if (d_face_diff_img[offset * nImage * nTriangle + nTriangle * (tarIm - n_half_window + offset) + d_valid2all[nidx]] < 0)
				continue;
			if (d_nodeImg_diff[num_node * tarIm + nidx] < 0)
				d_nodeImg_diff[num_node * tarIm + nidx] = 0;
			else {
				d_nodeImg_diff[num_node * tarIm + nidx] += d_face_diff_img[offset * nImage * nTriangle + nTriangle * (tarIm - n_half_window + offset) + d_valid2all[nidx]];
			}
			valid_count++;
		}
		d_nodeImg_diff[num_node * tarIm + nidx] /= valid_count;
		d_nodeImg_diff[num_node * tarIm + nidx] = exp(d_nodeImg_diff[num_node * tarIm + nidx]);
		d_nodeImg_diff[num_node * tarIm + nidx] *= weight;
	}
}

void calNodeweight(vector<int>& valid2all, vector<hostTriangle>& hTs, vector<set<uint>>& n_hops_tri_union, vector<set<uint>>& n_hops_ver_union, vector<vector<int>>& candidateLabels_ori, vector<vector<float3>>& stretchvec, vector<vector<float>>& curvaturevec, int nImage, float weight_c, float weight_s, float weight_q, vector<vector<vector<float>>>& result) {
	int num_nodes = valid2all.size();
	int nTriangle = hTs.size();
	int nVert = curvaturevec[0].size();

	result.resize(num_nodes);
	
	vector<int> node_face_size;
	vector<int> node_face_offset;
	vector<int> node_face_arr;

	vector<int> node_vert_size;
	vector<int> node_vert_offset;
	vector<int> node_vert_arr;

	node_face_size.resize(num_nodes);
	node_face_offset.resize(num_nodes);

	node_vert_size.resize(num_nodes);
	node_vert_offset.resize(num_nodes);
	int face_offset_now = 0;
	int vert_offset_now = 0;

	vector<int> h_NodeImg(num_nodes * nImage, -1);
	for (int ti = 0; ti < num_nodes; ti++) {
		for (auto im : hTs[valid2all[ti]]._Img) {
			h_NodeImg[valid2all.size() * im + ti] = 1;
		}
	}

	for (int ni = 0; ni < num_nodes; ni++) {
		int fi_size = candidateLabels_ori[valid2all[ni]].size();
		result[ni].resize(nImage);
		for (int i = 0; i < nImage; i++) {
			result[ni][i].resize(fi_size, 100);
		}
		node_face_offset[ni] = node_face_arr.size();
		node_vert_offset[ni] = node_vert_arr.size();
		node_face_size[ni] = n_hops_tri_union[valid2all[ni]].size();
		node_vert_size[ni] = n_hops_ver_union[valid2all[ni]].size();
		for (auto f : n_hops_tri_union[valid2all[ni]]) {
			node_face_arr.push_back(f);
		}
		for (auto v : n_hops_ver_union[valid2all[ni]]) {
			node_vert_arr.push_back(v);
		}
	}
	int* d_NodeImg;
	checkCudaErrors(cudaMalloc((void**)&d_NodeImg, sizeof(int)* num_nodes * nImage));
	checkCudaErrors(cudaMemcpy(d_NodeImg, h_NodeImg.data(), sizeof(int)* num_nodes * nImage, cudaMemcpyHostToDevice));

	int* d_node_face_size;
	checkCudaErrors(cudaMalloc((void**)&d_node_face_size, sizeof(int) * num_nodes));
	checkCudaErrors(cudaMemcpy(d_node_face_size, node_face_size.data(), sizeof(int) * num_nodes, cudaMemcpyHostToDevice));

	int* d_node_face_offset;
	checkCudaErrors(cudaMalloc((void**)&d_node_face_offset, sizeof(int) * num_nodes));
	checkCudaErrors(cudaMemcpy(d_node_face_offset, node_face_offset.data(), sizeof(int) * num_nodes, cudaMemcpyHostToDevice));

	int* d_node_face_arr;
	checkCudaErrors(cudaMalloc((void**)&d_node_face_arr, sizeof(int) * node_face_arr.size()));
	checkCudaErrors(cudaMemcpy(d_node_face_arr, node_face_arr.data(), sizeof(int) * node_face_arr.size(), cudaMemcpyHostToDevice));
	
	int* d_node_vert_size;
	checkCudaErrors(cudaMalloc((void**)&d_node_vert_size, sizeof(int) * num_nodes));
	checkCudaErrors(cudaMemcpy(d_node_vert_size, node_vert_size.data(), sizeof(int) * num_nodes, cudaMemcpyHostToDevice));

	int* d_node_vert_offset;
	checkCudaErrors(cudaMalloc((void**)&d_node_vert_offset, sizeof(int) * num_nodes));
	checkCudaErrors(cudaMemcpy(d_node_vert_offset, node_vert_offset.data(), sizeof(int) * num_nodes, cudaMemcpyHostToDevice));

	int* d_node_vert_arr;
	checkCudaErrors(cudaMalloc((void**)&d_node_vert_arr, sizeof(int) * node_vert_arr.size()));
	checkCudaErrors(cudaMemcpy(d_node_vert_arr, node_vert_arr.data(), sizeof(int) * node_vert_arr.size(), cudaMemcpyHostToDevice));
	
	float3* d_stretchvec;
	float* d_curvaturevec;
	checkCudaErrors(cudaMalloc((void**)&d_stretchvec, sizeof(float3) * nImage * nTriangle));
	checkCudaErrors(cudaMalloc((void**)&d_curvaturevec, sizeof(float) * nImage * nVert));
	for (int i = 0; i < nImage; i++) {
		checkCudaErrors(cudaMemcpy(&d_stretchvec[i * nTriangle], stretchvec[i].data(), sizeof(float3) * nTriangle, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(&d_curvaturevec[i * nVert], curvaturevec[i].data(), sizeof(float) * nVert, cudaMemcpyHostToDevice));
	}


	for (int i = 0; i < nImage; i++) {
		float* d_nodeImg_weight;
		vector<float> h_nodeImg_weight(num_nodes * nImage, -1.0);

		checkCudaErrors(cudaMalloc((void**)&d_nodeImg_weight, sizeof(float) * num_nodes * nImage));
		checkCudaErrors(cudaMemcpy(d_nodeImg_weight, h_nodeImg_weight.data(), sizeof(float) * num_nodes * nImage, cudaMemcpyHostToDevice));

		dim3 gridSize((num_nodes + 256 - 1) / 256);
		dim3 blockSize(256);
		//printf("update_texture_coordinate\n");
		calNodeImweight << < gridSize, blockSize >> > (i, d_NodeImg, d_node_face_size, d_node_face_offset, d_node_face_arr, d_node_vert_size, d_node_vert_offset, d_node_vert_arr, d_stretchvec, d_curvaturevec, d_nodeImg_weight, weight_c, weight_s, nImage, nTriangle, nVert, num_nodes);
		//calFaceImdiff << < gridSize, blockSize >> > (i, d_NodeImg, d_NodeVer, d_VerImgTex, d_nodeImg_diff, d_image, num_nodes, nImage, valid2all.size(), hVs.size(), w, h);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaMemcpy(h_nodeImg_weight.data(), d_nodeImg_weight, sizeof(float) * num_nodes * nImage, cudaMemcpyDeviceToHost));

		for (int ni = 0; ni < num_nodes; ni++) {
			int ficount = 0;
			for (auto tarim : candidateLabels_ori[valid2all[ni]]) {
				if(h_nodeImg_weight[num_nodes * tarim + ni] >= 0)
					result[ni][i][ficount] = h_nodeImg_weight[num_nodes * tarim + ni] + weight_q * (-hTs[valid2all[ni]].getImgWeight(candidateLabels_ori[valid2all[ni]][ficount]));
					//result[ni][i][ficount] = h_nodeImg_weight[num_nodes * tarim + ni] + weight_q * exp(-hTs[valid2all[ni]].getImgWeight(candidateLabels_ori[valid2all[ni]][ficount]));
				ficount++;
			}
		}
		checkCudaErrors(cudaFree(d_nodeImg_weight));
		checkCudaErrors(cudaDeviceSynchronize());
	}

	checkCudaErrors(cudaFree(d_NodeImg));
	checkCudaErrors(cudaFree(d_node_face_size));
	checkCudaErrors(cudaFree(d_node_face_offset));
	checkCudaErrors(cudaFree(d_node_face_arr));
	checkCudaErrors(cudaFree(d_node_vert_size));
	checkCudaErrors(cudaFree(d_node_vert_offset));
	checkCudaErrors(cudaFree(d_node_vert_arr));
	checkCudaErrors(cudaFree(d_stretchvec));
	checkCudaErrors(cudaFree(d_curvaturevec));
}

void calNodeweight2(vector<int>& valid2all, vector<hostTriangle>& hTs, vector<set<uint>>& n_hops_tri_union, vector<set<uint>>& n_hops_ver_union, vector<vector<int>>& candidateLabels_ori, vector<vector<float3>>& stretchvec, vector<vector<float3>>& curvaturevec, int nImage, float weight_c, float weight_s, float weight_q, vector<vector<vector<float>>>& result) {
	int num_nodes = valid2all.size();
	int nTriangle = hTs.size();
	int nVert = curvaturevec[0].size();

	result.resize(num_nodes);

	vector<int> node_face_size;
	vector<int> node_face_offset;
	vector<int> node_face_arr;

	vector<int> node_vert_size;
	vector<int> node_vert_offset;
	vector<int> node_vert_arr;

	node_face_size.resize(num_nodes);
	node_face_offset.resize(num_nodes);

	node_vert_size.resize(num_nodes);
	node_vert_offset.resize(num_nodes);
	int face_offset_now = 0;
	int vert_offset_now = 0;

	vector<int> h_NodeImg(num_nodes * nImage, -1);
	for (int ti = 0; ti < num_nodes; ti++) {
		for (auto im : hTs[valid2all[ti]]._Img) {
			h_NodeImg[valid2all.size() * im + ti] = 1;
		}
	}

	for (int ni = 0; ni < num_nodes; ni++) {
		int fi_size = candidateLabels_ori[valid2all[ni]].size();
		result[ni].resize(nImage);
		for (int i = 0; i < nImage; i++) {
			result[ni][i].resize(fi_size, 100);
		}
		node_face_offset[ni] = node_face_arr.size();
		node_vert_offset[ni] = node_vert_arr.size();
		node_face_size[ni] = n_hops_tri_union[valid2all[ni]].size();
		node_vert_size[ni] = n_hops_ver_union[valid2all[ni]].size();
		for (auto f : n_hops_tri_union[valid2all[ni]]) {
			node_face_arr.push_back(f);
		}
		for (auto v : n_hops_ver_union[valid2all[ni]]) {
			node_vert_arr.push_back(v);
		}
	}
	int* d_NodeImg;
	checkCudaErrors(cudaMalloc((void**)&d_NodeImg, sizeof(int) * num_nodes * nImage));
	checkCudaErrors(cudaMemcpy(d_NodeImg, h_NodeImg.data(), sizeof(int) * num_nodes * nImage, cudaMemcpyHostToDevice));

	int* d_node_face_size;
	checkCudaErrors(cudaMalloc((void**)&d_node_face_size, sizeof(int) * num_nodes));
	checkCudaErrors(cudaMemcpy(d_node_face_size, node_face_size.data(), sizeof(int) * num_nodes, cudaMemcpyHostToDevice));

	int* d_node_face_offset;
	checkCudaErrors(cudaMalloc((void**)&d_node_face_offset, sizeof(int) * num_nodes));
	checkCudaErrors(cudaMemcpy(d_node_face_offset, node_face_offset.data(), sizeof(int) * num_nodes, cudaMemcpyHostToDevice));

	int* d_node_face_arr;
	checkCudaErrors(cudaMalloc((void**)&d_node_face_arr, sizeof(int) * node_face_arr.size()));
	checkCudaErrors(cudaMemcpy(d_node_face_arr, node_face_arr.data(), sizeof(int) * node_face_arr.size(), cudaMemcpyHostToDevice));

	int* d_node_vert_size;
	checkCudaErrors(cudaMalloc((void**)&d_node_vert_size, sizeof(int) * num_nodes));
	checkCudaErrors(cudaMemcpy(d_node_vert_size, node_vert_size.data(), sizeof(int) * num_nodes, cudaMemcpyHostToDevice));

	int* d_node_vert_offset;
	checkCudaErrors(cudaMalloc((void**)&d_node_vert_offset, sizeof(int) * num_nodes));
	checkCudaErrors(cudaMemcpy(d_node_vert_offset, node_vert_offset.data(), sizeof(int) * num_nodes, cudaMemcpyHostToDevice));

	int* d_node_vert_arr;
	checkCudaErrors(cudaMalloc((void**)&d_node_vert_arr, sizeof(int) * node_vert_arr.size()));
	checkCudaErrors(cudaMemcpy(d_node_vert_arr, node_vert_arr.data(), sizeof(int) * node_vert_arr.size(), cudaMemcpyHostToDevice));

	float3* d_stretchvec;
	float3* d_curvaturevec;
	checkCudaErrors(cudaMalloc((void**)&d_stretchvec, sizeof(float3) * nImage * nTriangle));
	checkCudaErrors(cudaMalloc((void**)&d_curvaturevec, sizeof(float3) * nImage * nVert));
	for (int i = 0; i < nImage; i++) {
		checkCudaErrors(cudaMemcpy(&d_stretchvec[i * nTriangle], stretchvec[i].data(), sizeof(float3) * nTriangle, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(&d_curvaturevec[i * nVert], curvaturevec[i].data(), sizeof(float3) * nVert, cudaMemcpyHostToDevice));
	}


	for (int i = 0; i < nImage; i++) {
		float* d_nodeImg_weight;
		vector<float> h_nodeImg_weight(num_nodes * nImage, -1.0);

		checkCudaErrors(cudaMalloc((void**)&d_nodeImg_weight, sizeof(float) * num_nodes * nImage));
		checkCudaErrors(cudaMemcpy(d_nodeImg_weight, h_nodeImg_weight.data(), sizeof(float) * num_nodes * nImage, cudaMemcpyHostToDevice));

		dim3 gridSize((num_nodes + 256 - 1) / 256);
		dim3 blockSize(256);
		//printf("update_texture_coordinate\n");
		calNodeImweight2 << < gridSize, blockSize >> > (i, d_NodeImg, d_node_face_size, d_node_face_offset, d_node_face_arr, d_node_vert_size, d_node_vert_offset, d_node_vert_arr, d_stretchvec, d_curvaturevec, d_nodeImg_weight, weight_c, weight_s, nImage, nTriangle, nVert, num_nodes);
		//calFaceImdiff << < gridSize, blockSize >> > (i, d_NodeImg, d_NodeVer, d_VerImgTex, d_nodeImg_diff, d_image, num_nodes, nImage, valid2all.size(), hVs.size(), w, h);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaMemcpy(h_nodeImg_weight.data(), d_nodeImg_weight, sizeof(float) * num_nodes * nImage, cudaMemcpyDeviceToHost));

		for (int ni = 0; ni < num_nodes; ni++) {
			int ficount = 0;
			for (auto tarim : candidateLabels_ori[valid2all[ni]]) {
				if (h_nodeImg_weight[num_nodes * tarim + ni] >= 0)
					result[ni][i][ficount] = h_nodeImg_weight[num_nodes * tarim + ni] + weight_q * exp(-hTs[valid2all[ni]].getImgWeight(candidateLabels_ori[valid2all[ni]][ficount]));
				ficount++;
			}
		}
		checkCudaErrors(cudaFree(d_nodeImg_weight));
		checkCudaErrors(cudaDeviceSynchronize());
	}

	checkCudaErrors(cudaFree(d_NodeImg));
	checkCudaErrors(cudaFree(d_node_face_size));
	checkCudaErrors(cudaFree(d_node_face_offset));
	checkCudaErrors(cudaFree(d_node_face_arr));
	checkCudaErrors(cudaFree(d_node_vert_size));
	checkCudaErrors(cudaFree(d_node_vert_offset));
	checkCudaErrors(cudaFree(d_node_vert_arr));
	checkCudaErrors(cudaFree(d_stretchvec));
	checkCudaErrors(cudaFree(d_curvaturevec));
}

void precalFacediff(vector<int>& valid2all, vector<hostVertex>& hVs, vector<hostTriangle>& hTs, vector<vector<int>>& candidateLabels_ori, vector<uchar>& images, int nImage, int w, int h, float weight, vector<vector<float>>& result) {
	int num_nodes = valid2all.size();
	int nTriangle = hTs.size();

	result.resize(nImage);
	for (int i = 0; i < nImage; i++) {
		result[i].resize(nImage * num_nodes);

	}

	vector<int> h_NodeImg(num_nodes * nImage, -1);
	vector<int> h_NodeVer(num_nodes * 3, -1);
	vector<float2> h_VerImgTex(hVs.size() * nImage, { -1.0 , -1.0 });
	for (int ti = 0; ti < num_nodes; ti++) {
		for (auto im : hTs[valid2all[ti]]._Img) {
			h_NodeImg[num_nodes * im + ti] = 1;
		}
		h_NodeVer[3 * ti + 0] = hTs[valid2all[ti]]._Vertices[0];
		h_NodeVer[3 * ti + 1] = hTs[valid2all[ti]]._Vertices[1];
		h_NodeVer[3 * ti + 2] = hTs[valid2all[ti]]._Vertices[2];
	}
	for (int vi = 0; vi < hVs.size(); vi++) {
		for (int i = 0; i < hVs[vi]._Img.size(); i++) {
			int im = hVs[vi]._Img[i];
			h_VerImgTex[hVs.size() * im + vi] = hVs[vi]._Img_Tex[i];
		}
	}

	int* d_NodeImg;
	checkCudaErrors(cudaMalloc((void**)&d_NodeImg, sizeof(int) * valid2all.size() * nImage));
	checkCudaErrors(cudaMemcpy(d_NodeImg, h_NodeImg.data(), sizeof(int) * valid2all.size() * nImage, cudaMemcpyHostToDevice));

	int* d_NodeVer;
	checkCudaErrors(cudaMalloc((void**)&d_NodeVer, sizeof(int) * valid2all.size() * 3));
	checkCudaErrors(cudaMemcpy(d_NodeVer, h_NodeVer.data(), sizeof(int) * valid2all.size() * 3, cudaMemcpyHostToDevice));

	float2* d_VerImgTex;
	checkCudaErrors(cudaMalloc((void**)&d_VerImgTex, sizeof(float2) * hVs.size() * nImage));
	checkCudaErrors(cudaMemcpy(d_VerImgTex, h_VerImgTex.data(), sizeof(float2) * hVs.size() * nImage, cudaMemcpyHostToDevice));

	uchar* d_image;
	checkCudaErrors(cudaMalloc((void**)&d_image, sizeof(uchar) * w * h * nImage));
	checkCudaErrors(cudaMemcpy(d_image, images.data(), sizeof(uchar) * w * h * nImage, cudaMemcpyHostToDevice));

	for (int i = 0; i < nImage; i++) {
		float* d_nodeImg_diff;
		vector<float> h_nodeImg_diff(num_nodes * nImage, -1.0);

		checkCudaErrors(cudaMalloc((void**)&d_nodeImg_diff, sizeof(float) * num_nodes * nImage));
		checkCudaErrors(cudaMemcpy(d_nodeImg_diff, h_nodeImg_diff.data(), sizeof(float) * num_nodes * nImage, cudaMemcpyHostToDevice));

		dim3 gridSize((num_nodes + 256 - 1) / 256);
		dim3 blockSize(256);
		//printf("update_texture_coordinate\n");
		calFaceImdiff << < gridSize, blockSize >> > (i, d_NodeImg, d_NodeVer, d_VerImgTex, d_nodeImg_diff, d_image, num_nodes, nImage, valid2all.size(), hVs.size(), w, h, weight);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaMemcpy(result[i].data(), d_nodeImg_diff, sizeof(float) * num_nodes * nImage, cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(d_nodeImg_diff));
		checkCudaErrors(cudaDeviceSynchronize());
	}
	checkCudaErrors(cudaFree(d_NodeImg));
	checkCudaErrors(cudaFree(d_NodeVer));
	checkCudaErrors(cudaFree(d_VerImgTex));
	checkCudaErrors(cudaFree(d_image));
}

void precalFacediff_all(vector<hostVertex>& hVs, vector<hostTriangle>& hTs, vector<vector<int>>& candidateLabels_ori, vector<uchar>& images, int nImage, int w, int h, vector<vector<float>>& result) {
	int nTriangle = hTs.size();

	result.resize(nImage);
	for (int i = 0; i < nImage; i++) {
		result[i].resize(nImage * nTriangle);
	}

	vector<int> h_NodeImg(nTriangle * nImage, -1);
	vector<int> h_NodeVer(nTriangle * 3, -1);
	vector<float2> h_VerImgTex(hVs.size() * nImage, { -1.0 , -1.0 });
	for (int ti = 0; ti < nTriangle; ti++) {
		for (auto im : hTs[ti]._Img) {
			h_NodeImg[nTriangle * im + ti] = 1;
		}
		h_NodeVer[3 * ti + 0] = hTs[ti]._Vertices[0];
		h_NodeVer[3 * ti + 1] = hTs[ti]._Vertices[1];
		h_NodeVer[3 * ti + 2] = hTs[ti]._Vertices[2];
	}
	for (int vi = 0; vi < hVs.size(); vi++) {
		for (int i = 0; i < hVs[vi]._Img.size(); i++) {
			int im = hVs[vi]._Img[i];
			h_VerImgTex[hVs.size() * im + vi] = hVs[vi]._Img_Tex[i];
		}
	}

	int* d_NodeImg;
	checkCudaErrors(cudaMalloc((void**)&d_NodeImg, sizeof(int) * nTriangle * nImage));
	checkCudaErrors(cudaMemcpy(d_NodeImg, h_NodeImg.data(), sizeof(int) * nTriangle * nImage, cudaMemcpyHostToDevice));

	int* d_NodeVer;
	checkCudaErrors(cudaMalloc((void**)&d_NodeVer, sizeof(int) * nTriangle * 3));
	checkCudaErrors(cudaMemcpy(d_NodeVer, h_NodeVer.data(), sizeof(int) * nTriangle * 3, cudaMemcpyHostToDevice));

	float2* d_VerImgTex;
	checkCudaErrors(cudaMalloc((void**)&d_VerImgTex, sizeof(float2) * hVs.size() * nImage));
	checkCudaErrors(cudaMemcpy(d_VerImgTex, h_VerImgTex.data(), sizeof(float2) * hVs.size() * nImage, cudaMemcpyHostToDevice));

	uchar* d_image;
	checkCudaErrors(cudaMalloc((void**)&d_image, sizeof(uchar) * w * h * nImage));
	checkCudaErrors(cudaMemcpy(d_image, images.data(), sizeof(uchar) * w * h * nImage, cudaMemcpyHostToDevice));

	for (int i = 0; i < nImage; i++) {
		float* d_triImg_diff;
		vector<float> h_triImg_diff(nTriangle * nImage, -1.0);

		checkCudaErrors(cudaMalloc((void**)&d_triImg_diff, sizeof(float) * nTriangle * nImage));
		checkCudaErrors(cudaMemcpy(d_triImg_diff, h_triImg_diff.data(), sizeof(float) * nTriangle * nImage, cudaMemcpyHostToDevice));

		dim3 gridSize((nTriangle + 256 - 1) / 256);
		dim3 blockSize(256);
		//printf("update_texture_coordinate\n");
		calFaceImdiff_all << < gridSize, blockSize >> > (i, d_NodeImg, d_NodeVer, d_VerImgTex, d_triImg_diff, d_image, nTriangle, nImage, hVs.size(), w, h);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaMemcpy(result[i].data(), d_triImg_diff, sizeof(float) * nTriangle * nImage, cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(d_triImg_diff));
		checkCudaErrors(cudaDeviceSynchronize());
	}
	checkCudaErrors(cudaFree(d_NodeImg));
	checkCudaErrors(cudaFree(d_NodeVer));
	checkCudaErrors(cudaFree(d_VerImgTex));
	checkCudaErrors(cudaFree(d_image));
}

void calWidediff(vector<slEdge>& edge, vector<int>& valid2all, vector<vector<int>>& candidateLabels_ori, vector<set<uint>> face_union, vector<vector<float>>& face_diff, int nImage, float weight, vector<vector<vector<float>>>& result) {
	int num_edges = edge.size();
	int num_nodes = valid2all.size();
	int nTriangle = candidateLabels_ori.size();

	vector<int> edge_face_size;
	vector<int> edge_face_offset;
	vector<int> edge_face_arr;

	result.resize(num_edges);
	edge_face_size.resize(num_edges);
	edge_face_offset.resize(num_edges);
	int offset_now = 0;

	for (int ei = 0; ei < num_edges; ei++) {
		int fi1_size = candidateLabels_ori[valid2all[edge[ei].f1]].size();
		int fi2_size = candidateLabels_ori[valid2all[edge[ei].f2]].size();
		result[ei].resize(fi1_size);
		for (int fi1 = 0; fi1 < fi1_size; fi1++) {
			result[ei][fi1].resize(fi2_size);
		}
		set<int> tmp_edge_face_arr;
		for (auto f : face_union[valid2all[edge[ei].f1]]) {
			tmp_edge_face_arr.insert(f);
		}
		for (auto f : face_union[valid2all[edge[ei].f2]]) {
			tmp_edge_face_arr.insert(f);
		}
		for (auto f : tmp_edge_face_arr) {
			edge_face_arr.push_back(f);
		}

		edge_face_size[ei] = tmp_edge_face_arr.size();
		edge_face_offset[ei] = offset_now;
		offset_now += tmp_edge_face_arr.size();
	}


	slEdge* d_edge;
	checkCudaErrors(cudaMalloc((void**)&d_edge, sizeof(slEdge) * num_edges));
	checkCudaErrors(cudaMemcpy(d_edge, edge.data(), sizeof(slEdge) * num_edges, cudaMemcpyHostToDevice));

	float* d_face_diff_img;
	checkCudaErrors(cudaMalloc((void**)&d_face_diff_img, sizeof(float) * nImage * nTriangle));

	int* d_edge_face_size;
	checkCudaErrors(cudaMalloc((void**)&d_edge_face_size, sizeof(int) * num_edges));
	checkCudaErrors(cudaMemcpy(d_edge_face_size, edge_face_size.data(), sizeof(int) * num_edges, cudaMemcpyHostToDevice));

	int* d_edge_face_offset;
	checkCudaErrors(cudaMalloc((void**)&d_edge_face_offset, sizeof(int) * num_edges));
	checkCudaErrors(cudaMemcpy(d_edge_face_offset, edge_face_offset.data(), sizeof(int) * num_edges, cudaMemcpyHostToDevice));

	int* d_edge_face_arr;
	checkCudaErrors(cudaMalloc((void**)&d_edge_face_arr, sizeof(int) * edge_face_arr.size()));
	checkCudaErrors(cudaMemcpy(d_edge_face_arr, edge_face_arr.data(), sizeof(int) * edge_face_arr.size(), cudaMemcpyHostToDevice));
	
	int* d_valid2all;
	checkCudaErrors(cudaMalloc((void**)&d_valid2all, sizeof(int) * num_nodes));
	checkCudaErrors(cudaMemcpy(d_valid2all, valid2all.data(), sizeof(int) * num_nodes, cudaMemcpyHostToDevice));

	
	for (int i = 0; i < nImage; i++) {
		checkCudaErrors(cudaMemcpy(d_face_diff_img, face_diff[i].data(), sizeof(float) *nImage * nTriangle, cudaMemcpyHostToDevice));

		float* d_edgeImg_diff;
		vector<float> h_edgeImg_diff(num_edges * nImage, -1.0);

		checkCudaErrors(cudaMalloc((void**)&d_edgeImg_diff, sizeof(float) * edge.size() * nImage));
		checkCudaErrors(cudaMemcpy(d_edgeImg_diff, h_edgeImg_diff.data(), sizeof(float) * edge.size() * nImage, cudaMemcpyHostToDevice));

		dim3 gridSize((num_edges + 256 - 1) / 256);
		dim3 blockSize(256);
		//printf("update_texture_coordinate\n");
		calWideImdiff << < gridSize, blockSize >> > (d_valid2all, d_edge_face_size, d_edge_face_offset, d_edge_face_arr, d_face_diff_img, d_edgeImg_diff, num_edges,num_nodes,nImage, nTriangle, weight);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaMemcpy(h_edgeImg_diff.data(), d_edgeImg_diff, sizeof(float) * edge.size() * nImage, cudaMemcpyDeviceToHost));
		for (int ei = 0; ei < num_edges; ei++) {
			int f1count = 0;
			for (auto srcim : candidateLabels_ori[valid2all[edge[ei].f1]]) {
				if (srcim != i) {
					f1count++;
					continue;
				}
				int f2count = 0;
				for (auto tarim : candidateLabels_ori[valid2all[edge[ei].f2]]) {
					if (h_edgeImg_diff[num_edges * tarim + ei] > 0) {
						result[ei][f1count][f2count] = h_edgeImg_diff[num_edges * tarim + ei];
						//result[ei][f1count][f2count] += min(2.0, (float)abs(tarim - srcim));
					}
					f2count++;
				}
				f1count++;
			}
		}

		checkCudaErrors(cudaFree(d_edgeImg_diff));
		checkCudaErrors(cudaDeviceSynchronize());

	}
	checkCudaErrors(cudaFree(d_edge));
	checkCudaErrors(cudaFree(d_face_diff_img));
	checkCudaErrors(cudaFree(d_edge_face_size));
	checkCudaErrors(cudaFree(d_edge_face_offset));
	checkCudaErrors(cudaFree(d_edge_face_arr));
}

void calWindowdiff(vector<int>& valid2all, vector<vector<int>>& candidateLabels_ori, vector<vector<float>>& face_diff, int nImage, int n_half_window, float weight, vector<vector<vector<float>>>& result) {
	int num_nodes = valid2all.size();
	int nTriangle = candidateLabels_ori.size();

	result.resize(num_nodes);
	for (int ni = 0; ni < num_nodes; ni++) {
		int fi1_size = candidateLabels_ori[valid2all[ni]].size();
		int fi2_size = candidateLabels_ori[valid2all[ni]].size();
		result[ni].resize(fi1_size);
		for (int fi1 = 0; fi1 < fi1_size; fi1++) {
			result[ni][fi1].resize(fi2_size);
		}
	}

	vector<int> h_NodeImg(num_nodes * nImage, -1);
	for (int ti = 0; ti < num_nodes; ti++) {
		for (auto im : candidateLabels_ori[valid2all[ti]]) {
			h_NodeImg[num_nodes * im + ti] = 1;
		}
	}

	int* d_NodeImg;
	checkCudaErrors(cudaMalloc((void**)&d_NodeImg, sizeof(int)* num_nodes * nImage));
	checkCudaErrors(cudaMemcpy(d_NodeImg, h_NodeImg.data(), sizeof(int)* num_nodes * nImage, cudaMemcpyHostToDevice));

	float* d_face_diff_img;
	checkCudaErrors(cudaMalloc((void**)&d_face_diff_img, sizeof(float) * nImage * nTriangle * 2 * n_half_window));

	int* d_valid2all;
	checkCudaErrors(cudaMalloc((void**)&d_valid2all, sizeof(int) * num_nodes));
	checkCudaErrors(cudaMemcpy(d_valid2all, valid2all.data(), sizeof(int) * num_nodes, cudaMemcpyHostToDevice));

	for (int i = n_half_window - 1; i < nImage - n_half_window; i++) {
		for(int offset = 0; offset < 2 * n_half_window; offset++)
			checkCudaErrors(cudaMemcpy(d_face_diff_img + offset * nImage * nTriangle, face_diff[i + offset - (n_half_window - 1)].data(), sizeof(float) * nImage * nTriangle, cudaMemcpyHostToDevice));

		float* d_nodeImg_diff;
		vector<float> h_nodeImg_diff(num_nodes * nImage, -1.0);

		checkCudaErrors(cudaMalloc((void**)&d_nodeImg_diff, sizeof(float) * num_nodes * nImage));
		checkCudaErrors(cudaMemcpy(d_nodeImg_diff, h_nodeImg_diff.data(), sizeof(float) * num_nodes * nImage, cudaMemcpyHostToDevice));

		dim3 gridSize((num_nodes + 256 - 1) / 256);
		dim3 blockSize(256);
		//printf("update_texture_coordinate\n");
		calWindowImdiff << < gridSize, blockSize >> > (i, d_valid2all, d_face_diff_img, d_nodeImg_diff, num_nodes, n_half_window, nImage, nTriangle, weight);
		//calFaceImdiff << < gridSize, blockSize >> > (i, d_NodeImg, d_NodeVer, d_VerImgTex, d_nodeImg_diff, d_image, num_nodes, nImage, valid2all.size(), hVs.size(), w, h);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaMemcpy(h_nodeImg_diff.data(), d_nodeImg_diff, sizeof(float) * num_nodes * nImage, cudaMemcpyDeviceToHost));

		for (int ni = 0; ni < num_nodes; ni++) {
			int f1count = 0;
			for (auto srcim : candidateLabels_ori[valid2all[ni]]) {
				if (srcim != i) {
					f1count++;
					continue;
				}
				int f2count = 0;
				for (auto tarim : candidateLabels_ori[valid2all[ni]]) {
					if (h_nodeImg_diff[num_nodes * tarim + ni] > 0) {
						result[ni][f1count][f2count] = h_nodeImg_diff[num_nodes * tarim + ni];
					}
					f2count++;
				}
				f1count++;
			}
		}
		checkCudaErrors(cudaFree(d_nodeImg_diff));
		checkCudaErrors(cudaDeviceSynchronize());
	}
	checkCudaErrors(cudaFree(d_face_diff_img));
	checkCudaErrors(cudaFree(d_NodeImg));
}

void calEdgediff(vector<slEdge>& edge, vector<int>& valid2all, vector<hostVertex>& hVs, vector<hostTriangle>& hTs, vector<vector<int>>& candidateLabels_ori, vector<uchar>& images, int nImage, int w, int h, float weight, vector<vector<vector<float>>>& result) {
	int num_edges = edge.size();
	result.resize(num_edges);
	for (int ei = 0; ei < num_edges; ei++) {
		int fi1_size = candidateLabels_ori[valid2all[edge[ei].f1]].size();
		int fi2_size = candidateLabels_ori[valid2all[edge[ei].f2]].size();
		result[ei].resize(fi1_size);
		for (int fi1 = 0; fi1 < fi1_size; fi1++) {
			result[ei][fi1].resize(fi2_size);
		}
	}	
	
	vector<int> h_NodeImg(valid2all.size() * nImage, -1);
	vector<int> h_NodeVer(valid2all.size() * 3, -1);
	vector<float2> h_VerImgTex(hVs.size() * nImage, {-1.0 , -1.0});
	for (int ti = 0; ti < valid2all.size(); ti++) {
		for (auto im : hTs[valid2all[ti]]._Img) {
			h_NodeImg[valid2all.size() * im + ti] = 1;
		}
		h_NodeVer[3 * ti + 0] = hTs[valid2all[ti]]._Vertices[0];
		h_NodeVer[3 * ti + 1] = hTs[valid2all[ti]]._Vertices[1];
		h_NodeVer[3 * ti + 2] = hTs[valid2all[ti]]._Vertices[2];
	}
	for (int vi = 0; vi < hVs.size(); vi++) {
		for (int i = 0; i < hVs[vi]._Img.size(); i++) {
			int im = hVs[vi]._Img[i];
			h_VerImgTex[hVs.size() * im + vi] = hVs[vi]._Img_Tex[i];
		}
	}
	
	int* d_NodeImg;
	checkCudaErrors(cudaMalloc((void**)&d_NodeImg, sizeof(int) * valid2all.size() * nImage));
	checkCudaErrors(cudaMemcpy(d_NodeImg, h_NodeImg.data(), sizeof(int) * valid2all.size() * nImage, cudaMemcpyHostToDevice));

	int* d_NodeVer;
	checkCudaErrors(cudaMalloc((void**)&d_NodeVer, sizeof(int) * valid2all.size() * 3));
	checkCudaErrors(cudaMemcpy(d_NodeVer, h_NodeVer.data(), sizeof(int) * valid2all.size() * 3, cudaMemcpyHostToDevice));

	float2* d_VerImgTex;
	checkCudaErrors(cudaMalloc((void**)&d_VerImgTex, sizeof(float2) * hVs.size() * nImage));
	checkCudaErrors(cudaMemcpy(d_VerImgTex, h_VerImgTex.data(), sizeof(float2) * hVs.size() * nImage, cudaMemcpyHostToDevice));
	
	slEdge* d_edge;
	checkCudaErrors(cudaMalloc((void**)&d_edge, sizeof(slEdge) * edge.size()));
	checkCudaErrors(cudaMemcpy(d_edge, edge.data(), sizeof(slEdge) * edge.size(), cudaMemcpyHostToDevice));

	uchar* d_image;
	checkCudaErrors(cudaMalloc((void**)&d_image, sizeof(uchar) * w * h * nImage));
	checkCudaErrors(cudaMemcpy(d_image, images.data(), sizeof(uchar) * w * h* nImage, cudaMemcpyHostToDevice));
	
	for (int i = 0; i < nImage; i++) {
		float* d_edgeImg_diff;
		vector<float> h_edgeImg_diff(num_edges * nImage, -1.0);

		checkCudaErrors(cudaMalloc((void**)&d_edgeImg_diff, sizeof(float) * edge.size() * nImage));
		checkCudaErrors(cudaMemcpy( d_edgeImg_diff, h_edgeImg_diff.data(), sizeof(float) * edge.size() * nImage, cudaMemcpyHostToDevice));

		dim3 gridSize((num_edges + 256 - 1) / 256);
		dim3 blockSize(256);
		//printf("update_texture_coordinate\n");
		calEdgeImdiff << < gridSize, blockSize >> > (i, d_edge, d_NodeImg, d_NodeVer, d_VerImgTex, d_edgeImg_diff, d_image, num_edges, nImage, valid2all.size(), hVs.size(), w, h, weight);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaMemcpy(h_edgeImg_diff.data(), d_edgeImg_diff, sizeof(float) * edge.size() * nImage, cudaMemcpyDeviceToHost));
		for (int ei = 0; ei < num_edges; ei++) {
			int f1count = 0;
			for (auto srcim : candidateLabels_ori[valid2all[edge[ei].f1]]) {
				if (srcim != i) {
					f1count++;
					continue;
				}
				int f2count = 0;
				for (auto tarim : candidateLabels_ori[valid2all[edge[ei].f2]]) {
					if (h_edgeImg_diff[num_edges * tarim + ei] >= 0) {
						result[ei][f1count][f2count] = h_edgeImg_diff[num_edges * tarim + ei];
						/*if (abs(tarim - srcim) > 2)
							result[ei][f1count][f2count] = h_edgeImg_diff[num_edges * tarim + ei];
						else
							result[ei][f1count][f2count] = 0;*/
						//여기여기여기
						//result[ei][f1count][f2count] += min(1.0, (float)abs(tarim - srcim));
					}
					f2count++;
				}
				f1count++;
			}
		}

		checkCudaErrors(cudaFree(d_edgeImg_diff));
		checkCudaErrors(cudaDeviceSynchronize());
	}
	checkCudaErrors(cudaFree(d_NodeImg));
	checkCudaErrors(cudaFree(d_NodeVer));
	checkCudaErrors(cudaFree(d_VerImgTex));
	checkCudaErrors(cudaFree(d_edge));
	checkCudaErrors(cudaFree(d_image));
}

void calFacediff(vector<int>& valid2all, vector<hostVertex>& hVs, vector<hostTriangle>& hTs, vector<vector<int>>& candidateLabels_ori, vector<uchar>& images, int nImage, int w, int h, float weight, vector<vector<vector<float>>>& result) {
	int num_nodes = valid2all.size();

	result.resize(num_nodes);
	for (int ni = 0; ni < num_nodes; ni++) {
		int fi1_size = candidateLabels_ori[valid2all[ni]].size();
		int fi2_size = candidateLabels_ori[valid2all[ni]].size();
		result[ni].resize(fi1_size);
		for (int fi1 = 0; fi1 < fi1_size; fi1++) {
			result[ni][fi1].resize(fi2_size);
		}
	}

	vector<int> h_NodeImg(num_nodes * nImage, -1);
	vector<int> h_NodeVer(num_nodes * 3, -1);
	vector<float2> h_VerImgTex(hVs.size() * nImage, { -1.0 , -1.0 });
	for (int ti = 0; ti < num_nodes; ti++) {
		for (auto im : hTs[valid2all[ti]]._Img) {
			h_NodeImg[num_nodes * im + ti] = 1;
		}
		h_NodeVer[3 * ti + 0] = hTs[valid2all[ti]]._Vertices[0];
		h_NodeVer[3 * ti + 1] = hTs[valid2all[ti]]._Vertices[1];
		h_NodeVer[3 * ti + 2] = hTs[valid2all[ti]]._Vertices[2];
	}
	for (int vi = 0; vi < hVs.size(); vi++) {
		for (int i = 0; i < hVs[vi]._Img.size(); i++) {
			int im = hVs[vi]._Img[i];
			h_VerImgTex[hVs.size() * im + vi] = hVs[vi]._Img_Tex[i];
		}
	}

	int* d_NodeImg;
	checkCudaErrors(cudaMalloc((void**)&d_NodeImg, sizeof(int) * valid2all.size() * nImage));
	checkCudaErrors(cudaMemcpy(d_NodeImg, h_NodeImg.data(), sizeof(int) * valid2all.size() * nImage, cudaMemcpyHostToDevice));

	int* d_NodeVer;
	checkCudaErrors(cudaMalloc((void**)&d_NodeVer, sizeof(int) * valid2all.size() * 3));
	checkCudaErrors(cudaMemcpy(d_NodeVer, h_NodeVer.data(), sizeof(int) * valid2all.size() * 3, cudaMemcpyHostToDevice));

	float2* d_VerImgTex;
	checkCudaErrors(cudaMalloc((void**)&d_VerImgTex, sizeof(float2) * hVs.size() * nImage));
	checkCudaErrors(cudaMemcpy(d_VerImgTex, h_VerImgTex.data(), sizeof(float2) * hVs.size() * nImage, cudaMemcpyHostToDevice));

	uchar* d_image;
	checkCudaErrors(cudaMalloc((void**)&d_image, sizeof(uchar) * w * h * nImage));
	checkCudaErrors(cudaMemcpy(d_image, images.data(), sizeof(uchar) * w * h * nImage, cudaMemcpyHostToDevice));

	for (int i = 0; i < nImage; i++) {
		float* d_nodeImg_diff;
		vector<float> h_nodeImg_diff(num_nodes * nImage, -1.0);

		checkCudaErrors(cudaMalloc((void**)&d_nodeImg_diff, sizeof(float) * num_nodes * nImage));
		checkCudaErrors(cudaMemcpy(d_nodeImg_diff, h_nodeImg_diff.data(), sizeof(float) * num_nodes * nImage, cudaMemcpyHostToDevice));

		dim3 gridSize((num_nodes + 256 - 1) / 256);
		dim3 blockSize(256);
		//printf("update_texture_coordinate\n");
		calFaceImdiff << < gridSize, blockSize >> > (i, d_NodeImg, d_NodeVer, d_VerImgTex, d_nodeImg_diff, d_image, num_nodes, nImage, valid2all.size(), hVs.size(), w, h, weight);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaMemcpy(h_nodeImg_diff.data(), d_nodeImg_diff, sizeof(float) * num_nodes * nImage, cudaMemcpyDeviceToHost));

		for (int ni = 0; ni < num_nodes; ni++) {
			int f1count = 0;
			for (auto srcim : candidateLabels_ori[valid2all[ni]]) {
				if (srcim != i) {
					f1count++;
					continue;
				}
				int f2count = 0;
				for (auto tarim : candidateLabels_ori[valid2all[ni]]) {
					if (h_nodeImg_diff[num_nodes * tarim + ni] > 0) {
						result[ni][f1count][f2count] = h_nodeImg_diff[num_nodes * tarim + ni];
					}
					f2count++;
				}
				f1count++;
			}
		}
		checkCudaErrors(cudaFree(d_nodeImg_diff));
		checkCudaErrors(cudaDeviceSynchronize());
	}
	checkCudaErrors(cudaFree(d_NodeImg));
	checkCudaErrors(cudaFree(d_NodeVer));
	checkCudaErrors(cudaFree(d_VerImgTex));
	checkCudaErrors(cudaFree(d_image));
}
