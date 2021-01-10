#include "Selector.h"
#include "curvature.h"
#include "mapmap\full.h"
#include "Calculator.cuh"
#include <igl/parula.h>

#define WIDE_TERM
//#define WIDE_SPATIAL
//#define FROM_FILE
//#define WIDE_TEMPORAL
//#define ONEHOT_INIT


using namespace NS_MAPMAP;

using cost_t = float;
constexpr uint_t simd_w = mapmap::sys_max_simd_width<cost_t>();
using unary_t = mapmap::UnaryTable<cost_t, simd_w>;
using pairwise_t = mapmap::PairwiseTable<cost_t, simd_w>;

bool ascent_pair(const pair<uint, float>& a, const pair<uint, float>& b) {
	return a.second < b.second;
}

size_t TexMap::Selector::getLabel(int targetIdx, int faceIdx) {
	if (all2valid[faceIdx] < 0)
		return -1;
	else
		return resultLabels[targetIdx][all2valid[faceIdx]];
}

void TexMap::Selector::Mapper4DandLabeling_new(Mapper4D* mapper4D) {
	mapper4D_ptr = mapper4D;

	float param_pair_spatial = 10.0;
	float param_pair_temporal = 2;

	float theta_n = 0.3;
	float theta_b = 20.0;
	float omega_g = 0.9;

	int label_tree_node = 500;
	int sel_label = 20;
	int nTransition = 1;

	int temporal_refine_num = 2;
	bool tempo_selective_w = true;
	bool spa_selective_w = true;

	vector<hostVertex> tmpVs;
	vector<hostTriangle> tmpTs;
	vector<vector<int>> candidateLabels_ori;
	vector<vector<int>> candidateLabels_all;
	vector<vector<int>> candidateLabels_temporal;
	vector<set<uint>> n_hops_tri_union_tmp;
	vector<set<uint>> n_hops_tri_union_tmp_pre;
	vector<set<uint>> n_hops_tri_union_bound;
	vector<set<uint>> n_hops_tri_union_colsim;
	vector<set<uint>> n_hops_tri_union;
	vector<set<uint>> one_hops_tri_union;

	vector<vector<float>> rendered_sim;
	vector<float> spa_weight;
	vector<vector<float>> tmp_weight;

	mapper4D_ptr->GetNumInfo(&nVertex, &nTriangle, &nImage);
	mapper4D_ptr->Get_VT_layer(tmpVs, tmpTs, LAYERNUM - 1);
	candidateLabels_ori.resize(nTriangle);
	candidateLabels_all.resize(nTriangle);
	candidateLabels_temporal.resize(nTriangle);

	n_hops_tri_union_tmp.resize(nTriangle);
	n_hops_tri_union_tmp_pre.resize(nTriangle);
	n_hops_tri_union_bound.resize(nTriangle);
	n_hops_tri_union_colsim.resize(nTriangle);
	n_hops_tri_union.resize(nTriangle);
	one_hops_tri_union.resize(nTriangle);

	rendered_sim.resize(nImage);
	spa_weight.resize(nTriangle);
	tmp_weight.resize(nImage);

	// Step function is applied in "*.csv" file already (theta: 0.95)
	string sim_table_path = data_root_path + "sim_table.csv";
	ifstream sim_table;
	sim_table.open(sim_table_path, ios::in);

	string sim_row;
	for (int i = 0; i < nImage; i++)
		rendered_sim[i].resize(nImage);
	int tmp_frame_idx = 0;
	while (getline(sim_table, sim_row) && sim_table.good()) {
		std::vector<std::string> row = csv_read_row(sim_row, ',');
		for (int ri = 0; ri < row.size(); ri++) {
			rendered_sim[tmp_frame_idx][ri] = stof(row[ri]);
		}
		tmp_frame_idx++;
	}

	string var_table_path = data_root_path + "var_table.csv";
	ifstream var_table;
	var_table.open(var_table_path, ios::in);

	string var_row;
	getline(var_table, var_row);
	std::vector<std::string> spa_row = csv_read_row(var_row, ',');
	for (int ri = 0; ri < spa_row.size(); ri++) {
		spa_weight[ri] = stof(spa_row[ri]);
	}
	for (int i = 0; i < nImage; i++) {
		tmp_weight[i].resize(nTriangle);
		getline(var_table, var_row);
		std::vector<std::string> tmp_row = csv_read_row(var_row, ',');
		for (int ri = 0; ri < tmp_row.size(); ri++) {
			tmp_weight[i][ri] = stof(tmp_row[ri]);
		}
	}

	std::cout << "set union...";
	StopWatchInterface* time_union = NULL;
	sdkCreateTimer(&time_union);
	sdkStartTimer(&time_union);
	//// refind valid label (away n hops from boundry)
	// initailize triangles union
	for (int i = 0; i < nTriangle; i++) {
		n_hops_tri_union_tmp[i].insert(i);
		n_hops_tri_union_tmp_pre[i].insert(i);
		n_hops_tri_union_bound[i].insert(i);
		n_hops_tri_union[i].insert(i);
		one_hops_tri_union[i].insert(i);
		n_hops_tri_union_colsim[i].insert(i);
	}
	int max_hop = max(max(BOUND_HOPS, GEOSIM_HOPS_STRETCH), COLSIM_HOPS);
	int hop = 0;
	while (hop < max_hop) {
		hop++;
		for (int i = 0; i < nTriangle; i++) {
			set<uint> o_ring_tri;
			for (auto fi : n_hops_tri_union_tmp_pre[i]) {
				map<uint, int> tmp_vertex;
				for (int fv_i = 0; fv_i < 3; fv_i++) {
					int fv_idx = tmpTs[fi]._Vertices[fv_i];
					tmp_vertex.insert(make_pair(fv_idx, fv_i));
					for (auto fvf_idx : tmpVs[fv_idx]._Triangles) {
						o_ring_tri.insert(fvf_idx);
					}
				}
			}
			n_hops_tri_union_tmp_pre[i].insert(o_ring_tri.begin(), o_ring_tri.end());
			for (auto t : n_hops_tri_union_tmp[i]) {
				n_hops_tri_union_tmp_pre[i].erase(t);
			}
			n_hops_tri_union_tmp[i].insert(o_ring_tri.begin(), o_ring_tri.end());
			if (hop == 1)
				one_hops_tri_union[i].insert(n_hops_tri_union_tmp[i].begin(), n_hops_tri_union_tmp[i].end());
			if (hop == BOUND_HOPS)
				n_hops_tri_union_bound[i].insert(n_hops_tri_union_tmp[i].begin(), n_hops_tri_union_tmp[i].end());
			if (hop == GEOSIM_HOPS_STRETCH)
				n_hops_tri_union[i].insert(n_hops_tri_union_tmp[i].begin(), n_hops_tri_union_tmp[i].end());
			if (hop == COLSIM_HOPS)
				n_hops_tri_union_colsim[i].insert(n_hops_tri_union_tmp[i].begin(), n_hops_tri_union_tmp[i].end());
		}
	}

	sdkStopTimer(&time_union);
	float union_time = sdkGetAverageTimerValue(&time_union) / 1000.0f;
	std::cout << "time : " << union_time << "s" << std::endl;


	// find candidate except around spatial boundary
	for (int i = 0; i < nTriangle; i++) {
		candidateLabels_ori[i].resize(tmpTs[i]._Img.size());
		candidateLabels_ori[i].assign(tmpTs[i]._Img.begin(), tmpTs[i]._Img.end());
	}
	// check valid face to construct graph node
	all2valid.resize(nTriangle, -1);
	for (int i = 0; i < nTriangle; i++) {
		if (candidateLabels_ori[i].size() > 0) {
			all2valid[i] = valid2all.size();
			valid2all.push_back(i);
		}
	}

	num_nodes = valid2all.size();
	// find spatial edge
	for (auto fi : valid2all) {
		set<uint> o_ring_tri;
		map<uint, int> tmp_vertex;
		for (int fv_i = 0; fv_i < 3; fv_i++) {
			int fv_idx = tmpTs[fi]._Vertices[fv_i];
			tmp_vertex.insert(make_pair(fv_idx, fv_i));
			for (auto fvf_idx : tmpVs[fv_idx]._Triangles) {
				o_ring_tri.insert(fvf_idx);
			}
		}
		for (auto ringf_idx : o_ring_tri) {
			//if (ringf_idx <= fi)
			if (ringf_idx <= fi || all2valid[ringf_idx] < 0)
				continue;
			int match_count = 0;
			int vxv[4];
			for (int fv_i = 0; fv_i < 3; fv_i++) {
				auto iter = tmp_vertex.find(tmpTs[ringf_idx]._Vertices[fv_i]);
				if (iter != tmp_vertex.end()) {
					vxv[match_count++] = iter->second;
					vxv[match_count++] = fv_i;
				}
			}
			// edge를 공유하는 삼각형
			if (match_count == 4) {
				slEdge e;
				//e.f1 = fi;
				//e.f2 = ringf_idx;
				e.f1 = all2valid[fi];
				e.f2 = all2valid[ringf_idx];
				bool back = false;
				if ((vxv[0] + 1) % 3 == vxv[2])
					e.v_v1 = vxv[0];
				else {
					e.v_v1 = vxv[2];
					back = !back;
				}
				if ((vxv[1] + 1) % 3 == vxv[3])
					e.v_v2 = vxv[1];
				else {
					e.v_v2 = vxv[3];
					back = !back;
				}
				e.backward = back;
				edge_vec.push_back(e);
			}
		}
	}

	num_edges = edge_vec.size();

	// find wide color patches


	std::cout << "Nodes: " << num_nodes << std::endl;
	// spatial edges
	std::cout << "Edges: " << num_edges << std::endl;

	vector<set<uint>> neighbor_nodes;
	neighbor_nodes.resize(num_nodes);

	vector<vector<int>> pivot_label;
	pivot_label.resize(nImage);
	for (int i = 0; i < nImage; i++) {
		pivot_label[i].resize(num_nodes);
	}

#ifdef FROM_FILE
	std::ifstream fin_s("./temp", std::ofstream::binary);
	for (int i = 0; i < nImage; i++) {
		fin_s.read((char*)pivot_label[i].data(), sizeof(int) * num_nodes);
	}
	fin_s.close();
#else

	// bring images
	int w = mapper4D_ptr->colorImages[0].cols;
	int h = mapper4D_ptr->colorImages[0].rows;
	int img_size = h * w;

	// gray image data
	vector<uchar> images;
	images.resize(img_size * nImage);
	for (int i = 0; i < nImage; i++) {
		cv::Mat tmpgray;
		cv::cvtColor(mapper4D_ptr->colorImages[i], tmpgray, cv::COLOR_BGR2GRAY);
		/*cv::imshow("111", tmpgray);
		cv::waitKey(0);*/
		memcpy(&images[img_size * i], tmpgray.data, sizeof(uchar) * img_size);
	}

	//// prepare error
	// unary
	std::cout << "set unary error Table...";
	StopWatchInterface* time_ue = NULL;
	sdkCreateTimer(&time_ue);
	sdkStartTimer(&time_ue);
	vector<vector<vector<float>>> node_img_weight;
	node_img_weight.resize(num_nodes);
	for (int ni = 0; ni < num_nodes; ni++) {
		int fi_size = candidateLabels_ori[valid2all[ni]].size();
		node_img_weight[ni].resize(nImage);
		for (int i = 0; i < nImage; i++) {
			node_img_weight[ni][i].resize(fi_size, 0.0);
		}
	}

	// SHOT
	for (int ni = 0; ni < num_nodes; ni++) {
		int l_size = candidateLabels_ori[valid2all[ni]].size();

		hostVertex* TV[3];
		TV[0] = &tmpVs[tmpTs[valid2all[ni]]._Vertices[0]];
		TV[1] = &tmpVs[tmpTs[valid2all[ni]]._Vertices[1]];
		TV[2] = &tmpVs[tmpTs[valid2all[ni]]._Vertices[2]];

		// rendered sim
		for (int l = 0; l < l_size; l++) {
			float blur_factor;
			int im_b = tmpTs[valid2all[ni]]._Img[l] - 1;
			int im_f = tmpTs[valid2all[ni]]._Img[l] + 1;
			int im_n = tmpTs[valid2all[ni]]._Img[l];
			if (tmpTs[valid2all[ni]].getImgIdx(im_b) < 0 || tmpTs[valid2all[ni]].getImgIdx(im_f) < 0)
				blur_factor = 1.0;
			else {
				float2 vary_coord_b = make_float2(0.0, 0.0);
				float2 vary_coord_f = make_float2(0.0, 0.0);
				float2 vary_coord_n = make_float2(0.0, 0.0);
				for (int fvi = 0; fvi < 3; fvi++) {
					vary_coord_b += TV[fvi]->_Img_Tex_ori[TV[fvi]->getImgIdx(im_b)];
					vary_coord_f += TV[fvi]->_Img_Tex_ori[TV[fvi]->getImgIdx(im_f)];
					vary_coord_n += TV[fvi]->_Img_Tex_ori[TV[fvi]->getImgIdx(im_n)];
				}
				vary_coord_b /= 3.0;
				vary_coord_f /= 3.0;
				vary_coord_n /= 3.0;

				blur_factor = (length(vary_coord_b - vary_coord_n) + length(vary_coord_f - vary_coord_n)) / 2.0 > theta_b ? 1.0 : 0.0;
			}
			//float quality_factor = 1.0 / clamp(tmpTs[valid2all[ni]]._Img_Weight[l], 0.1, 0.5);
			float quality_factor = tmpTs[valid2all[ni]]._Img_Weight[l] > theta_n ? 0.0 : 1.0;
			for (int i = 0; i < nImage; i++) {
				node_img_weight[ni][i][l] += rendered_sim[i][candidateLabels_ori[valid2all[ni]][l]] * omega_g + quality_factor + blur_factor;
			}
		}
	}

	// read local sim table
	std::cout << "read SHOT descriptors: " << std::endl << std::endl;
	std::ifstream local_sim_file;
	local_sim_file.open(data_root_path + "local_sim_table.dat", std::ofstream::binary);
	vector<float> local_sim_table;
	local_sim_table.resize(nImage);
	if (local_sim_file.is_open()) {
		for (int i = 0; i < nVertex; i++) {
			printProgBar(((float)i / (nVertex - 1.0)) * 100);
			for (int j = 0; j < nImage; j++) {
				local_sim_file.read((char*)local_sim_table.data(), sizeof(float) * nImage);
				for (auto vf : tmpVs[i]._Triangles) {
					if (all2valid[vf] < 0)
						continue;
					for (int l = 0; l < candidateLabels_ori[vf].size(); l++) {
						node_img_weight[all2valid[vf]][j][l] += (-local_sim_table[candidateLabels_ori[vf][l]]) * (1 - omega_g) / 3.0;
					}
				}
			}
		}
	}
	local_sim_file.close();
	std::cout << "... Done" << std::endl;

	sdkStopTimer(&time_ue);
	float ue_time = sdkGetAverageTimerValue(&time_ue) / 1000.0f;
	std::cout << "time : " << ue_time << "s" << std::endl;

	// spatial
	std::cout << "set spatial edge error Table...";
	StopWatchInterface* time_see = NULL;
	sdkCreateTimer(&time_see);
	sdkStartTimer(&time_see);
	vector<vector<vector<float>>> edge_img_diff;
	calEdgediff(edge_vec, valid2all, tmpVs, tmpTs, candidateLabels_ori, images, nImage, w, h, param_pair_spatial, edge_img_diff);
	// various weighting
	int e_count = 0;
	for (auto e : edge_vec) {
		for (auto e_f1 : edge_img_diff[e_count]) {
			for (auto e_f1_f2 : e_f1) {
				if (spa_selective_w)
					e_f1_f2 *= ((spa_weight[valid2all[e.f1]] + spa_weight[valid2all[e.f2]]) / 2.0);
			}
		}
		e_count++;
	}

	sdkStopTimer(&time_see);
	float see_time = sdkGetAverageTimerValue(&time_see) / 1000.0f;
	std::cout << "time : " << see_time << "s" << std::endl;
	// temporal
	std::cout << "set temporal edge error Table...";
	StopWatchInterface* time_tee = NULL;
	sdkCreateTimer(&time_tee);
	sdkStartTimer(&time_tee);
	vector<vector<vector<float>>> edge_tempo_diff;
	calFacediff(valid2all, tmpVs, tmpTs, candidateLabels_ori, images, nImage, w, h, param_pair_temporal, edge_tempo_diff);
	for (int ni = 0; ni < num_nodes; ni++) {
		int f1_idx = 0;
		for (auto& e_f1 : edge_tempo_diff[ni]) {
			int f2_idx = 0;
			for (auto& e_f1_f2 : e_f1) {
				//float tmp_www = fminf(tmp_weight[candidateLabels_now[n][l]][valid2all[n]], tmp_weight[pivot_label[window_start +  1][n]][valid2all[n]]);
				if (tempo_selective_w)
					e_f1_f2 *= fminf(tmp_weight[candidateLabels_ori[valid2all[ni]][f1_idx]][valid2all[ni]], tmp_weight[candidateLabels_ori[valid2all[ni]][f2_idx]][valid2all[ni]]);
				f2_idx++;
			}
			f1_idx++;
		}
	}

	sdkStopTimer(&time_tee);
	float tee_time = sdkGetAverageTimerValue(&time_tee) / 1000.0f;
	std::cout << "time : " << tee_time << "s" << std::endl;

	// set transitions (idx is same as candidate)
	std::cout << "set transition Table...";
	StopWatchInterface* time_trans = NULL;
	sdkCreateTimer(&time_trans);
	sdkStartTimer(&time_trans);
	vector<vector<vector<int>>> trans_Forward;
	vector<vector<vector<int>>> trans_Backward;
	trans_Forward.resize(num_nodes);
	trans_Backward.resize(num_nodes);
	for (int ni = 0; ni < num_nodes; ni++) {
		int l_size = candidateLabels_ori[valid2all[ni]].size();
		trans_Forward[ni].resize(l_size);
		trans_Backward[ni].resize(l_size);
		for (int l = 0; l < l_size; l++) {
			vector<pair<int, float>> tmp_F;
			vector<pair<int, float>> tmp_B;
			for (int tar_l = 0; tar_l < l_size; tar_l++) {
				tmp_F.push_back(make_pair(tar_l, edge_tempo_diff[ni][l][tar_l]));
				tmp_B.push_back(make_pair(tar_l, edge_tempo_diff[ni][tar_l][l]));
			}
			sort(tmp_F.begin(), tmp_F.end(), [](const pair<int, float>& l, const pair<int, float>& r) {return l.second < r.second; });
			sort(tmp_B.begin(), tmp_B.end(), [](const pair<int, float>& l, const pair<int, float>& r) {return l.second < r.second; });
			for (int li = 0; li < nTransition; li++) {
				if (l_size < li + 1)
					break;
				trans_Forward[ni][l].push_back(tmp_F[li].first);
				trans_Backward[ni][l].push_back(tmp_B[li].first);
			}
		}
	}
	sdkStopTimer(&time_trans);
	float tans_time = sdkGetAverageTimerValue(&time_trans) / 1000.0f;
	std::cout << "time : " << tans_time << "s" << std::endl;

	// spatially initializing
#if TRUE
	for (int window_start = 0; window_start < nImage; window_start++) {
		int hierarchy_level = 0;
		while (true) {
			std::cout << std::endl;
			std::cout << std::endl;
			std::cout << "Window position: " << window_start << std::endl;
			std::cout << "Hierarchy Level: " << hierarchy_level << std::endl;
			std::cout << "Sampling...";
			StopWatchInterface* time_sampling = NULL;
			sdkCreateTimer(&time_sampling);
			sdkStartTimer(&time_sampling);
			bool all_done = true;
			vector<vector<int>> candidateLabels_now;
			vector<vector<int>> Labelidx_now;
			candidateLabels_now.resize(num_nodes);
			Labelidx_now.resize(num_nodes);
			vector<pair<uint, float>> node_img_init;
			for (int ni = 0; ni < num_nodes; ni++) {
				node_img_init.clear();
				for (int l = 0; l < candidateLabels_ori[valid2all[ni]].size(); l++) {
					node_img_init.push_back(make_pair(l, node_img_weight[ni][window_start][l]));
				}
				sort(node_img_init.begin(), node_img_init.end(), ascent_pair);
				int l_count = 0;
				while (l_count < sel_label && l_count < candidateLabels_ori[valid2all[ni]].size()) {
					Labelidx_now[ni].push_back(node_img_init[l_count].first);
					candidateLabels_now[ni].push_back(candidateLabels_ori[valid2all[ni]][node_img_init[l_count].first]);
					l_count++;
				}
			}
			hierarchy_level++;
			sdkStopTimer(&time_sampling);
			float sampling_time = sdkGetAverageTimerValue(&time_sampling) / 1000.0f;
			std::cout << "time : " << sampling_time << "s" << std::endl;

			/* solver instance */
			mapMAP<cost_t, simd_w> mapmap;

			std::unique_ptr<Graph<cost_t>> graph;
			std::unique_ptr<LabelSet<cost_t, simd_w>> label_set;
			std::vector<std::unique_ptr<unary_t>> unaries;
			std::vector<std::unique_ptr<PairwiseTable<cost_t, simd_w>>> pairwise_spatial;

			//// Graph topology
			// arrangement sequence
			// e.g. #A [ #B ] -> [bbbbb bbbbb bbbbb ...]
			// node : #frame [ #valid face ]
			// edge : #frame [ #spatial edge ], #frame-1 [ #temporal edge ]
			////

			// construct graph
			graph = unique_ptr<Graph<cost_t>>(new Graph<cost_t>(num_nodes));

			// 에지 추가, weight = 1.0 initialize
			// spatial edge
			for (auto e : edge_vec)
			{
				graph->add_edge(e.f1, e.f2, 1.0);
			}
			// 확정
			graph->update_components();

			// 라벨셋 생성
			label_set = unique_ptr<LabelSet<cost_t, simd_w>>(new LabelSet<cost_t, simd_w>(num_nodes, false));
			for (uint n = 0; n < num_nodes; n++) {
				const uint n_labels = candidateLabels_now[n].size();
				std::vector<_iv_st<cost_t, simd_w>> lset(n_labels);
				//std::cout << n << " :";
				for (uint l = 0; l < n_labels; ++l) {
					lset[l] = candidateLabels_now[n][l];
					//std::cout << " " << candidateLabels_now[n][l];
				}
				label_set->set_label_set_for_node(n, lset);
				//std::cout << std::endl;
			}
			/* construct optimizer */
			mapmap.set_graph(graph.get());
			mapmap.set_label_set(label_set.get());

			// unary term
			unaries.reserve(num_nodes);
			for (uint n = 0; n < num_nodes; n++) {
				const uint n_labels = candidateLabels_now[n].size();
				unaries.emplace_back(std::unique_ptr<unary_t>(new unary_t(n, label_set.get())));
				vector<_s_t<cost_t, simd_w>> costs(n_labels);
				for (uint32_t l = 0; l < n_labels; ++l) {
					costs[l] = node_img_weight[n][window_start][Labelidx_now[n][l]];
					//std::cout << " " << costs[l];
				}
				//std::cout << std::endl;
				unaries.back()->set_costs(costs);
				mapmap.set_unary(n, unaries[n].get());
			}

			// pairwise term
			pairwise_spatial.resize(num_edges);

			// spatial edge
			std::cout << "set spatial pairwise Table...";
			StopWatchInterface* t = NULL;
			sdkCreateTimer(&t);
			sdkStartTimer(&t);
			int e_count = 0;
			for (auto e : edge_vec) {
				int tmp_count = 0;
				std::vector<_s_t<cost_t, simd_w>> costs(candidateLabels_now[e.f1].size() * candidateLabels_now[e.f2].size());

				//std::cout << e_count << " :";
				for (auto f1_idx : Labelidx_now[e.f1]) {
					for (auto f2_idx : Labelidx_now[e.f2]) {
						costs[tmp_count] = edge_img_diff[e_count][f1_idx][f2_idx];
						//std::cout << " " << edge_img_diff[e_count][f1_idx][f2_idx];
						//std::cout << " " << costs[tmp_count];
						tmp_count++;
					}
					//std::cout << std::endl;
				}
				//std::cout << std::endl;
				//std::cout << "====================================";
				//std::cout << std::endl;
				pairwise_spatial[e_count] = std::unique_ptr<PairwiseTable<cost_t, simd_w>>(new PairwiseTable<cost_t, simd_w>(e.f1, e.f2, label_set.get(), costs));
				e_count++;
			}
			sdkStopTimer(&t);
			float layer_time = sdkGetAverageTimerValue(&t) / 1000.0f;
			std::cout << "time : " << layer_time << "s" << std::endl;

			for (luint_t e_id = 0; e_id < num_edges; e_id++) {
				mapmap.set_pairwise(e_id, pairwise_spatial[e_id].get());
			}

			/* termination criterion and control flow */

			std::unique_ptr<TerminationCriterion<cost_t, simd_w>> terminate;
			// mapMap 0.0001, let there be color 0.01
			terminate = std::unique_ptr<TerminationCriterion<cost_t, simd_w>>(
				new StopWhenReturnsDiminish<cost_t, simd_w>(5, 0.01));

			mapMAP_control ctr;
			/* create (optional) control flow settings */
			/*ctr.use_multilevel = true;
			ctr.use_spanning_tree = true;
			ctr.use_acyclic = true;
			ctr.spanning_tree_multilevel_after_n_iterations = 5;
			ctr.force_acyclic = true;
			ctr.min_acyclic_iterations = 5;
			ctr.relax_acyclic_maximal = true;
			ctr.tree_algorithm = OPTIMISTIC_TREE_SAMPLER;*/

			ctr.use_multilevel = true;
			ctr.use_spanning_tree = true;
			ctr.use_acyclic = false;
			ctr.spanning_tree_multilevel_after_n_iterations = 5;
			ctr.force_acyclic = false;
			ctr.min_acyclic_iterations = 5;
			ctr.relax_acyclic_maximal = false;
			ctr.tree_algorithm = OPTIMISTIC_TREE_SAMPLER;

			/* set to true and select a seed for (serial) deterministic sampling */
			ctr.sample_deterministic = true;
			ctr.initial_seed = 548923563;

			/*auto display = [](const mapmap::luint_t time_ms,
				const mapmap::_iv_st<cost_t, simd_w> objective) {
					std::cout << "\t\t" << time_ms / 1000.0 << "\t" << objective << std::endl;
			};
			mapmap.set_logging_callback(display);*/
			mapmap.set_termination_criterion(terminate.get());


			std::cout << "Finished loading dataset." << std::endl;

			/* use standard multilevel and termination criterion and start */
			std::vector<_iv_st<cost_t, simd_w>> solution;

			/* catch errors thrown during optimization */
			try
			{
				mapmap.optimize(solution, ctr);
			}
			catch (std::runtime_error& e)
			{
				std::cout << UNIX_COLOR_RED
					<< "Caught an exception: "
					<< UNIX_COLOR_WHITE
					<< e.what()
					<< UNIX_COLOR_RED
					<< ", exiting..."
					<< UNIX_COLOR_RESET
					<< std::endl;
			}
			catch (std::domain_error& e)
			{
				std::cout << UNIX_COLOR_RED
					<< "Caught an exception: "
					<< UNIX_COLOR_WHITE
					<< e.what()
					<< UNIX_COLOR_RED
					<< ", exiting..."
					<< UNIX_COLOR_RESET
					<< std::endl;
			}
			for (uint n = 0; n < num_nodes; ++n)
			{
				pivot_label[window_start][n] = label_set->label_from_offset(n, solution[n]);
			}
			/* Extract lables from solution (vector of label indices) */
			if (all_done)
				break;
		}
	}
#endif

	// 요기요
	// temporally refining
	for (int k = 0; k < temporal_refine_num; k++) {
		for (int window_start = k % 2; window_start < nImage; window_start += 2) {
			while (true) {
				std::cout << std::endl;
				std::cout << std::endl;
				std::cout << "Window position: " << window_start << std::endl;
				std::cout << "Sampling...";
				StopWatchInterface* time_sampling = NULL;
				sdkCreateTimer(&time_sampling);
				sdkStartTimer(&time_sampling);
				bool all_done = true;
				vector<vector<int>> candidateLabels_now;
				vector<vector<int>> Labelidx_now;
				candidateLabels_now.resize(num_nodes);
				Labelidx_now.resize(num_nodes);
				/*for (int ni = 0; ni < num_nodes; ni++) {
					map<int, int> tmp_labels_indices;
					tmp_labels_indices.insert(pair<int, int>(pivot_label[window_start][ni], distance(candidateLabels_ori[valid2all[ni]].begin(), std::find(candidateLabels_ori[valid2all[ni]].begin(), candidateLabels_ori[valid2all[ni]].end(), pivot_label[window_start][ni]))));
					if (window_start > 0) {
						int l = distance(candidateLabels_ori[valid2all[ni]].begin(), std::find(candidateLabels_ori[valid2all[ni]].begin(), candidateLabels_ori[valid2all[ni]].end(), pivot_label[window_start - 1][ni]));
						for (int t = 0; t < nTransition; t++) {
							if (trans_Forward[ni][l].size() < t + 1)
								break;
							tmp_labels_indices.insert(pair<int, int>(candidateLabels_ori[valid2all[ni]][trans_Forward[ni][l][t]], trans_Forward[ni][l][t]));
						}
					}
					if (window_start < nImage - 1) {
						int l = distance(candidateLabels_ori[valid2all[ni]].begin(), std::find(candidateLabels_ori[valid2all[ni]].begin(), candidateLabels_ori[valid2all[ni]].end(), pivot_label[window_start + 1][ni]));
						for (int t = 0; t < nTransition; t++) {
							if (trans_Backward[ni][l].size() < t + 1)
								break;
							tmp_labels_indices.insert(pair<int, int>(candidateLabels_ori[valid2all[ni]][trans_Backward[ni][l][t]], trans_Backward[ni][l][t]));
						}
					}
					for (auto labidx : tmp_labels_indices) {
						candidateLabels_now[ni].push_back(labidx.first);
						Labelidx_now[ni].push_back(labidx.second);
					}
				}*/

				vector<pair<uint, float>> node_img_init;
				for (int ni = 0; ni < num_nodes; ni++) {
					node_img_init.clear();
					for (int l = 0; l < candidateLabels_ori[valid2all[ni]].size(); l++) {
						node_img_init.push_back(make_pair(l, node_img_weight[ni][window_start][l]));
					}
					sort(node_img_init.begin(), node_img_init.end(), ascent_pair);
					int l_count = 0;
					while (l_count < sel_label && l_count < candidateLabels_ori[valid2all[ni]].size()) {
						Labelidx_now[ni].push_back(node_img_init[l_count].first);
						candidateLabels_now[ni].push_back(candidateLabels_ori[valid2all[ni]][node_img_init[l_count].first]);
						l_count++;
					}
					if (window_start < nImage - 1 && window_start > 0) {
						int l = distance(candidateLabels_ori[valid2all[ni]].begin(), std::find(candidateLabels_ori[valid2all[ni]].begin(), candidateLabels_ori[valid2all[ni]].end(), pivot_label[window_start + 1][ni]));
						int k = distance(candidateLabels_now[ni].begin(), std::find(candidateLabels_now[ni].begin(), candidateLabels_now[ni].end(), pivot_label[window_start + 1][ni]));
						if (l < candidateLabels_ori[valid2all[ni]].size() && k == candidateLabels_now[ni].size()) {
							Labelidx_now[ni].push_back(l);
							candidateLabels_now[ni].push_back(candidateLabels_ori[valid2all[ni]][l]);
						}
					}
				}

				vector<int> tmp_candidateLabels_now;
				vector<int> tmp_Labelidx_now;
				if (0) {
					for (int ni = 0; ni < num_nodes; ni++) {
						tmp_candidateLabels_now.clear();
						tmp_Labelidx_now.clear();
						for (int i = 0; i < candidateLabels_now[ni].size(); i++) {
							if (candidateLabels_now[ni][i] % 2 == 1) {
								tmp_candidateLabels_now.push_back(candidateLabels_now[ni][i]);
								tmp_Labelidx_now.push_back(Labelidx_now[ni][i]);
							}
						}
						if (tmp_candidateLabels_now.empty()) {
							tmp_candidateLabels_now.push_back(candidateLabels_now[ni][0]);
							tmp_Labelidx_now.push_back(Labelidx_now[ni][0]);
						}
						candidateLabels_now[ni].clear();
						Labelidx_now[ni].clear();
						candidateLabels_now[ni].assign(tmp_candidateLabels_now.begin(), tmp_candidateLabels_now.end());
						Labelidx_now[ni].assign(tmp_Labelidx_now.begin(), tmp_Labelidx_now.end());
					}
				}

				sdkStopTimer(&time_sampling);
				float sampling_time = sdkGetAverageTimerValue(&time_sampling) / 1000.0f;
				std::cout << "time : " << sampling_time << "s" << std::endl;

				/* solver instance */
				mapMAP<cost_t, simd_w> mapmap;

				std::unique_ptr<Graph<cost_t>> graph;
				std::unique_ptr<LabelSet<cost_t, simd_w>> label_set;
				std::vector<std::unique_ptr<unary_t>> unaries;
				std::vector<std::unique_ptr<PairwiseTable<cost_t, simd_w>>> pairwise_spatial;

				//// Graph topology
				// arrangement sequence
				// e.g. #A [ #B ] -> [bbbbb bbbbb bbbbb ...]
				// node : #frame [ #valid face ]
				// edge : #frame [ #spatial edge ], #frame-1 [ #temporal edge ]
				////

				// construct graph
				graph = unique_ptr<Graph<cost_t>>(new Graph<cost_t>(num_nodes));

				// 에지 추가, weight = 1.0 initialize
				// spatial edge
				for (auto e : edge_vec)
				{
					graph->add_edge(e.f1, e.f2, 1.0);
				}
				// 확정
				graph->update_components();

				// 라벨셋 생성
				label_set = unique_ptr<LabelSet<cost_t, simd_w>>(new LabelSet<cost_t, simd_w>(num_nodes, false));
				for (uint n = 0; n < num_nodes; n++) {
					const uint n_labels = candidateLabels_now[n].size();
					std::vector<_iv_st<cost_t, simd_w>> lset(n_labels);
					for (uint l = 0; l < n_labels; ++l)
						lset[l] = candidateLabels_now[n][l];
					label_set->set_label_set_for_node(n, lset);
				}
				/* construct optimizer */
				mapmap.set_graph(graph.get());
				mapmap.set_label_set(label_set.get());

				// unary term
				unaries.reserve(num_nodes);
				for (uint n = 0; n < num_nodes; n++) {
					const uint n_labels = candidateLabels_now[n].size();
					unaries.emplace_back(std::unique_ptr<unary_t>(new unary_t(n, label_set.get())));
					vector<_s_t<cost_t, simd_w>> costs(n_labels);
					for (uint32_t l = 0; l < n_labels; ++l) {
						/*
						//costs[l] = min(1.0, abs(candidateLabels[valid2all[n]][l] - i)) + 1.0 - tmpTs[valid2all[n]].getImgWeight(candidateLabels[valid2all[n]][l]);
						float sim_cost_curvature = 0.0;
						float sim_cost_stretch = 0.0;
						for (auto fi : n_hops_tri_union[valid2all[n]]) {
							sim_cost_stretch += dot(stretchvec[window_start][fi], stretchvec[l][fi]) / (length(stretchvec[window_start][fi]) * length(stretchvec[l][fi]));
							//sim_cost_stretch -= length(stretchvec[i + window_start][fi] - stretchvec[l][fi]);
						}
						for (auto vi : n_hops_ver_union[valid2all[n]]) {
							sim_cost_curvature += dot(curvaturevec[window_start][vi], curvaturevec[l][vi]) / (length(curvaturevec[window_start][vi]) * length(curvaturevec[l][vi]));
							//sim_cost_curvature -= length(curvaturevec[i + window_start][vi] - curvaturevec[l][vi]);
						}
						sim_cost_stretch /= n_hops_tri_union[valid2all[n]].size();
						sim_cost_curvature /= n_hops_ver_union[valid2all[n]].size();
						costs[l] = param_uni_sim_curvature * exp(-sim_cost_curvature) +
							param_uni_sim_stretch * exp(-sim_cost_stretch) +
							param_uni_quality * exp(-tmpTs[valid2all[n]].getImgWeight(candidateLabels_now[n][l]));
							*/
						costs[l] = node_img_weight[n][window_start][Labelidx_now[n][l]];

						if (window_start > 0) {
							//float tmp_www = fminf(tmp_weight[candidateLabels_now[n][l]][valid2all[n]], tmp_weight[pivot_label[window_start - 1][n]][valid2all[n]]);
							int f1_idx = std::distance(candidateLabels_ori[valid2all[n]].begin(), std::find(candidateLabels_ori[valid2all[n]].begin(), candidateLabels_ori[valid2all[n]].end(), pivot_label[window_start - 1][n]));
							int f2_idx = std::distance(candidateLabels_ori[valid2all[n]].begin(), std::find(candidateLabels_ori[valid2all[n]].begin(), candidateLabels_ori[valid2all[n]].end(), candidateLabels_now[n][l]));
							costs[l] += edge_tempo_diff[n][f1_idx][f2_idx];
							//costs[l] += param_pair_temporal * edge_tempo_diff[n][f1_idx][f2_idx];
						}
						if (window_start < nImage - 1) {
							//float tmp_www = fminf(tmp_weight[candidateLabels_now[n][l]][valid2all[n]], tmp_weight[pivot_label[window_start +  1][n]][valid2all[n]]);
							int f1_idx = std::distance(candidateLabels_ori[valid2all[n]].begin(), std::find(candidateLabels_ori[valid2all[n]].begin(), candidateLabels_ori[valid2all[n]].end(), candidateLabels_now[n][l]));
							int f2_idx = std::distance(candidateLabels_ori[valid2all[n]].begin(), std::find(candidateLabels_ori[valid2all[n]].begin(), candidateLabels_ori[valid2all[n]].end(), pivot_label[window_start + 1][n]));
							//costs[l] += 0.1 * edge_tempo_diff[n][f1_idx][f2_idx];
							costs[l] += edge_tempo_diff[n][f1_idx][f2_idx];
							//costs[l] += param_pair_temporal * edge_tempo_diff[n][f1_idx][f2_idx];
						}
					}
					unaries.back()->set_costs(costs);
					mapmap.set_unary(n, unaries[n].get());
				}

				// pairwise term
				pairwise_spatial.resize(num_edges);

				// spatial edge
				std::cout << "set spatial pairwise Table...";
				StopWatchInterface* t = NULL;
				sdkCreateTimer(&t);
				sdkStartTimer(&t);
				int e_count = 0;
				for (auto e : edge_vec) {
					int tmp_count = 0;
					std::vector<_s_t<cost_t, simd_w>> costs(candidateLabels_now[e.f1].size() * candidateLabels_now[e.f2].size());

					for (auto f1_idx : Labelidx_now[e.f1]) {
						for (auto f2_idx : Labelidx_now[e.f2]) {
							costs[tmp_count] = edge_img_diff[e_count][f1_idx][f2_idx];
							tmp_count++;
						}
					}
					pairwise_spatial[e_count] = std::unique_ptr<PairwiseTable<cost_t, simd_w>>(new PairwiseTable<cost_t, simd_w>(e.f1, e.f2, label_set.get(), costs));
					e_count++;
				}
				sdkStopTimer(&t);
				float layer_time = sdkGetAverageTimerValue(&t) / 1000.0f;
				std::cout << "time : " << layer_time << "s" << std::endl;

				for (luint_t e_id = 0; e_id < num_edges; e_id++) {
					mapmap.set_pairwise(e_id, pairwise_spatial[e_id].get());
				}

				/* termination criterion and control flow */

				std::unique_ptr<TerminationCriterion<cost_t, simd_w>> terminate;
				// mapMap 0.0001, let there be color 0.01
				terminate = std::unique_ptr<TerminationCriterion<cost_t, simd_w>>(
					new StopWhenReturnsDiminish<cost_t, simd_w>(5, 0.01));

				mapMAP_control ctr;
				/* create (optional) control flow settings */
				ctr.use_multilevel = true;
				ctr.use_spanning_tree = true;
				ctr.use_acyclic = false;
				ctr.spanning_tree_multilevel_after_n_iterations = 5;
				ctr.force_acyclic = false;
				ctr.min_acyclic_iterations = 5;
				ctr.relax_acyclic_maximal = false;
				//ctr.tree_algorithm = LOCK_FREE_TREE_SAMPLER;
				ctr.tree_algorithm = OPTIMISTIC_TREE_SAMPLER;

				/* set to true and select a seed for (serial) deterministic sampling */
				ctr.sample_deterministic = true;
				ctr.initial_seed = 548923723;
				mapmap.set_termination_criterion(terminate.get());

				std::cout << "Finished loading dataset." << std::endl;

				/* use standard multilevel and termination criterion and start */
				std::vector<_iv_st<cost_t, simd_w>> solution;

				/* catch errors thrown during optimization */
				try
				{
					mapmap.optimize(solution, ctr);
				}
				catch (std::runtime_error& e)
				{
					std::cout << UNIX_COLOR_RED
						<< "Caught an exception: "
						<< UNIX_COLOR_WHITE
						<< e.what()
						<< UNIX_COLOR_RED
						<< ", exiting..."
						<< UNIX_COLOR_RESET
						<< std::endl;
				}
				catch (std::domain_error& e)
				{
					std::cout << UNIX_COLOR_RED
						<< "Caught an exception: "
						<< UNIX_COLOR_WHITE
						<< e.what()
						<< UNIX_COLOR_RED
						<< ", exiting..."
						<< UNIX_COLOR_RESET
						<< std::endl;
				}
				for (uint n = 0; n < num_nodes; ++n)
				{
					pivot_label[window_start][n] = label_set->label_from_offset(n, solution[n]);
				}
				/* Extract lables from solution (vector of label indices) */
				if (all_done)
					break;
			}
		}
	}


#endif // FROM_FILE

	// save result
	resultLabels.resize(nImage);
	for (int i = 0; i < nImage; i++) {
		resultLabels[i].resize(num_nodes);
		for (uint n = 0; n < num_nodes; ++n)
		{
			resultLabels[i][n] = pivot_label[i][n];
		}
	}

	boundLabels.resize(nImage);
	for (int t = 0; t < nImage; t++) {
		boundLabels[t].resize(nTriangle);
		// n_hops_tri_union
		for (int fi = 0; fi < nTriangle; fi++) {
			if (all2valid[fi] < 0)
				continue;
			set<int> tmp_bound_labels;
			int now_label = resultLabels[t][all2valid[fi]];
			bool is_bound = false;
			for (auto ffi : n_hops_tri_union_bound[fi]) {
				if (all2valid[ffi] < 0)
					continue;
				if (now_label != resultLabels[t][all2valid[ffi]]) {
					tmp_bound_labels.insert(now_label);
					tmp_bound_labels.insert(resultLabels[t][all2valid[ffi]]);
				}
			}
			boundLabels[t][fi].clear();
			boundLabels[t][fi].assign(tmp_bound_labels.begin(), tmp_bound_labels.end());
		}
	}

}
