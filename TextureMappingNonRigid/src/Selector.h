#pragma once

#ifndef SELECTOR
#define SELECTOR

#define NOMINMAX
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

#if !defined(__CUDACC__)
#include<Eigen\Eigen>
#endif

#include "any.hpp"
#include "TriangleAndVertex.h"
#include "tbb/task_scheduler_init.h"

#include "Mapper4D.h"

// 에지 정보 내에 각 face에서 어떤 vertex가 쓰이는지 알아야 한다. (pairwise error)
// f1(idx), f2, v'v1, v'v2
// v'v는 [01]->0, [12]->1, [20]->2 ([v'v, (v'v+1)%3] 으로 복구, backword 서로 역순일 때 ex: [01][10])
struct slEdge {
	int f1;
	int f2;
	int v_v1;
	int v_v2;
	bool backward;
	//vector<float> smooth_table_spatial;
};
/*data structure for graph*/
// version 1 - key frame, potts model

struct n_ary_Tree {
	vector<int> label;
	vector<shared_ptr<n_ary_Tree>> child;
};


namespace TexMap {

	class Selector {
	public:
		// edge_vec : vec[node1, node2]
		// data_term : vec[node->[label, cost]]

		void init(vector<hostVertex>& hVs, vector<hostTriangle>& hTs);
		//void setTable(vector<hostVertex>& hVs, vector<hostTriangle>& hTs, uchar* images, int w, int h);
		void Mapper4DandLabeling(Mapper4D* mapper4D);
		void Mapper4DandLabeling_ml(Mapper4D* mapper4D);
		void Mapper4DandLabeling_sep(Mapper4D* mapper4D);
		void Mapper4DandLabeling_new(Mapper4D* mapper4D);
		void Mapper4DandLabeling_initsel(Mapper4D* mapper4D);
		void Mapper4DandLabeling_initsel_qual(Mapper4D* mapper4D);
		void Mapper4DandLabeling_base_global_sim(Mapper4D* mapper4D);
		void Mapper4DandLabeling_sep_idxspa(Mapper4D* mapper4D);
		void run(vector<hostVertex>& hVs, vector<hostTriangle>& hTs, uchar* images, int w, int h);
		size_t getLabel(int targetIdx, int faceIdx);

		Mapper4D* mapper4D_ptr;

	private:
		int num_threads = 8;
		int e_samnum = 2; // must be > 1
		int f_samnum = 9; // must be < SAMNUM
		float weight_table[3][SAMNUM] = { 0.5, 0.25, 0.25, 0.8, 0.1, 0.1, 0.5, 0.0, 0.5, 0.2, 0.6, 0.2, 0.1, 0.5, 0.4,
											  0.25, 0.5, 0.25, 0.1, 0.8, 0.1, 0.5, 0.5, 0.0, 0.2, 0.2, 0.6, 0.5, 0.4, 0.1,
											  0.25, 0.25, 0.5, 0.1, 0.1, 0.8, 0.0, 0.5, 0.5, 0.6, 0.2, 0.2, 0.4, 0.1, 0.5 };
		bool reset = true;

		int nVertex;
		int nTriangle;
		int nImage;

		vector<int> valid2all;
		vector<int> all2valid;
		vector<vector<int>> resultLabels; // nImage * nTriangles -> single label
		vector<vector<vector<int>>> boundLabels; // nTriangles -> multi labels

		size_t num_nodes;
		size_t num_edges;
		//size_t num_labels;

		vector<slEdge> edge_vec;
		//vector<vector<pair<size_t, cost_t>>> data_term;

		vector<size_t> node_label;
	};
}



#endif