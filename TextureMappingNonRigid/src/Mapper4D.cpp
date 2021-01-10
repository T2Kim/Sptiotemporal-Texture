#include "Mapper4D.h"
#include "Simplifier.h"

using namespace TexMap;

static Simplifier* simpleifier_ptr = NULL;

Mapper4D::Mapper4D(string templateMeshPath, string NR_MeshPathAndPrefix, string streamPath, int startIdx, int endIdx)
{
	m_templateMeshPath = templateMeshPath;
	m_NR_MeshPathAndPrefix = NR_MeshPathAndPrefix;
	m_streamPath = streamPath;

	template_mesh.request_vertex_texcoords2D();
	template_mesh.request_face_normals();
	template_mesh.request_vertex_normals();
	opt += OpenMesh::IO::Options::VertexTexCoord;
	if (!OpenMesh::IO::read_mesh(template_mesh, templateMeshPath, opt))
	{
		std::cerr << "read error\n";
		exit(1);
	}
	template_mesh.update_normals();

	_Vertices.resize(template_mesh.n_vertices());
	_Triangles.resize(template_mesh.n_faces());

	numVer = template_mesh.n_vertices();
	numTri = template_mesh.n_faces();

	_CO_IR = cv::Mat::eye(4, 4, CV_32F);
	for (int i = 0; i < 12; i++) {
		_CO_IR.at<float>((int)(i / 4), (int)(i % 4)) = D_C_EXT[i];
	}
	_IR_CO = _CO_IR.inv();
	Calpos(streamPath, NR_MeshPathAndPrefix, startIdx, endIdx);
}

void Mapper4D::ConstructVertree() {

	//// Generate Progressive mesh
	if (!simpleifier_ptr)
		delete simpleifier_ptr;
	simpleifier_ptr = new Simplifier(layer_mesh_vec);
	simpleifier_ptr->simplify_layer(LAYERNUM);

	propAccum.resize(imgNum);
	propValid.resize(imgNum);
	float2 dudvInit;
	dudvInit.x = 0.0;
	dudvInit.y = 0.0;
	for (int i = 0; i < imgNum; i++) {
		propAccum[i].resize(numVer, dudvInit);
		propValid[i].resize(numVer, false);
	}
	layer_Vertices.resize(LAYERNUM);
	layer_Triangles.resize(LAYERNUM);
	layer_numVer.resize(LAYERNUM);
	layer_numTri.resize(LAYERNUM);

	vector<vector<set<uint>>> ref_imgs_set_pre_ver;
	vector<vector<set<uint>>> ref_imgs_set_pre_tri;
	vector<vector<map<uint, float2>>> ref_imgs_set_ver;
	vector<vector<map<uint, float>>> ref_imgs_set_tri_ori;
	vector<vector<map<uint, float>>> ref_imgs_set_tri;

	ref_imgs_set_pre_ver.resize(LAYERNUM - 1);
	ref_imgs_set_pre_tri.resize(LAYERNUM - 1);
	ref_imgs_set_ver.resize(LAYERNUM);
	ref_imgs_set_tri_ori.resize(LAYERNUM);
	ref_imgs_set_tri.resize(LAYERNUM);

	// fine to coarse
	// l -> LAYERNUM (fine) ~ 0 (coarse)
	for (int l = LAYERNUM - 1; l > -1; l--) {
		unsigned int tmp_sim_numVer = simpleifier_ptr->extVer[LAYERNUM - l - 1].size();
		unsigned int tmp_sim_numTri = simpleifier_ptr->extTri[LAYERNUM - l - 1].size();
		if (l < LAYERNUM - 1) {
			ref_imgs_set_pre_ver[l].resize(tmp_sim_numVer);
			ref_imgs_set_pre_tri[l].resize(tmp_sim_numTri);
		}

		ref_imgs_set_ver[l].resize(tmp_sim_numVer);
		ref_imgs_set_tri_ori[l].resize(tmp_sim_numTri);
		ref_imgs_set_tri[l].resize(tmp_sim_numTri);
	}

	//reserve
	for (int l = LAYERNUM - 1; l > -1; l--) {
		unsigned int tmp_sim_numVer = simpleifier_ptr->extVer[LAYERNUM - l - 1].size();
		unsigned int tmp_sim_numTri = simpleifier_ptr->extTri[LAYERNUM - l - 1].size();
		lVertex* tmp_layer_Vertices = simpleifier_ptr->extVer[LAYERNUM - l - 1].data();
		lTriangle* tmp_layer_Triangles = simpleifier_ptr->extTri[LAYERNUM - l - 1].data();

		layer_Vertices[l].resize(tmp_sim_numVer);
		layer_Triangles[l].resize(tmp_sim_numTri);
		layer_numVer[l] = tmp_sim_numVer;
		layer_numTri[l] = tmp_sim_numTri;


		for (int i = 0; i < tmp_sim_numVer; i++) {
			layer_Vertices[l][i]._Triangles.assign(tmp_layer_Vertices[i]._Triangles.begin(), tmp_layer_Vertices[i]._Triangles.end());
		}
		for (int i = 0; i < tmp_sim_numTri; i++) {
			layer_Triangles[l][i]._Vertices[0] = tmp_layer_Triangles[i]._Vertices[0];
			layer_Triangles[l][i]._Vertices[1] = tmp_layer_Triangles[i]._Vertices[1];
			layer_Triangles[l][i]._Vertices[2] = tmp_layer_Triangles[i]._Vertices[2];
		}

		vector<vector<float3>> positions_vec;
		vector<vector<float3>> normals_vec;
		vector<vector<float2>> pixel_vec;
		vector<vector<float>> areas_vec;
		vector<vector<float3>> tri_positions_vec;
		vector<vector<float>> tri_trans_vec_image;
		positions_vec.resize(imgNum);
		pixel_vec.resize(imgNum);
		normals_vec.resize(imgNum);
		areas_vec.resize(imgNum);
		tri_positions_vec.resize(imgNum);
		tri_trans_vec_image.resize(imgNum);


		for (int i = 0; i < imgNum; i++) {
			positions_vec[i].resize(tmp_sim_numVer);
			pixel_vec[i].resize(tmp_sim_numVer);
			normals_vec[i].resize(tmp_sim_numTri);
			areas_vec[i].resize(tmp_sim_numTri);
			tri_positions_vec[i].resize(tmp_sim_numTri);
			tri_trans_vec_image[i].resize(tmp_sim_numTri);
			cv::Mat _P_inv_IR = (_Pose.inv());
			cv::Mat _P_inv_CO = (_CO_IR * _P_inv_IR);

			for (int vi = 0; vi < tmp_sim_numVer; vi++) {
				positions_vec[i][vi] = tmp_layer_Vertices[vi]._Pos[i];

				//////////////////pixel cal
				float3 _Point_CO;
				float3 _Point_IR;
				float2 _Pixel_CO;
				float2 _Pixel_IR;

				Transform(positions_vec[i][vi], _P_inv_CO, _Point_CO);
				PointToPixel_CO(_Point_CO, _Pixel_CO);
				pixel_vec[i][vi] = _Pixel_CO;
			}
			for (int fi = 0; fi < tmp_sim_numTri; fi++) {
				int fvi0 = simpleifier_ptr->extTri[LAYERNUM - l - 1][fi]._Vertices[0];
				int fvi1 = simpleifier_ptr->extTri[LAYERNUM - l - 1][fi]._Vertices[1];
				int fvi2 = simpleifier_ptr->extTri[LAYERNUM - l - 1][fi]._Vertices[2];

				float3 _Point_CO0;
				float3 _Point_CO1;
				float3 _Point_CO2;

				Transform(positions_vec[i][fvi0], _P_inv_CO, _Point_CO0);
				Transform(positions_vec[i][fvi1], _P_inv_CO, _Point_CO1);
				Transform(positions_vec[i][fvi2], _P_inv_CO, _Point_CO2);

				float3 fvv1 = _Point_CO1 - _Point_CO0;
				float3 fvv2 = _Point_CO2 - _Point_CO0;

				float3 f_normal = cross(fvv1, fvv2);
				f_normal /= length(f_normal);

				normals_vec[i][fi] = f_normal;
				//normals_vec[i][fi] = simpleifier_ptr->extTri[LAYERNUM - l - 1][fi]._Normal[i];
			}
			for (int fi = 0; fi < tmp_sim_numTri; fi++) {
				int fvi0 = simpleifier_ptr->extTri[LAYERNUM - l - 1][fi]._Vertices[0];
				int fvi1 = simpleifier_ptr->extTri[LAYERNUM - l - 1][fi]._Vertices[1];
				int fvi2 = simpleifier_ptr->extTri[LAYERNUM - l - 1][fi]._Vertices[2];

				tri_positions_vec[i][fi] = (tmp_layer_Vertices[fvi0]._Pos[i] + tmp_layer_Vertices[fvi1]._Pos[i] + tmp_layer_Vertices[fvi2]._Pos[i]) / 3.0;
				if (i > 0) {
					float2 _Pixel_pre;
					float2 _Pixel_now;
					PointToPixel_CO(tri_positions_vec[i - 1][fi], _Pixel_pre);
					PointToPixel_CO(tri_positions_vec[i][fi], _Pixel_now);
					tri_trans_vec_image[i][fi] = Distance(_Pixel_pre, _Pixel_now);
				}
				else {
					tri_trans_vec_image[i][fi] = 0;
				}

				Vec3f v0; // = tmp_mesh_vec[i].point(fv_it++);
				Vec3f v1; // = tmp_mesh_vec[i].point(fv_it++);
				Vec3f v2; // = tmp_mesh_vec[i].point(fv_it);
				v0 = Vec3f(pixel_vec[i][fvi0].x, pixel_vec[i][fvi0].y, 1.0);
				v1 = Vec3f(pixel_vec[i][fvi1].x, pixel_vec[i][fvi1].y, 1.0);
				v2 = Vec3f(pixel_vec[i][fvi2].x, pixel_vec[i][fvi2].y, 1.0);

				Vec3d e1 = OpenMesh::vector_cast<Vec3d, Vec3f>(v1 - v0);
				Vec3d e2 = OpenMesh::vector_cast<Vec3d, Vec3f>(v2 - v0);

				Vec3d fN = OpenMesh::cross(e1, e2);
				double area = fN.norm() / 2.0;

				areas_vec[i][fi] = area;
			}
		}
		for (int vi = 0; vi < tmp_sim_numVer; vi++) {
			std::vector<mDummyImage> candidateImages;
			candidateImages.clear();
			for (int i = 0; i < imgNum; i++)
			{
				float3 _Point_CO;
				float3 _Point_IR;
				float2 _Pixel_CO;
				float2 _Pixel_IR;
				cv::Mat _P_inv_IR = (_Pose.inv());
				cv::Mat _P_inv_CO = (_CO_IR * _P_inv_IR);

				Transform(positions_vec[i][vi], _P_inv_IR, _Point_IR);
				Transform(positions_vec[i][vi], _P_inv_CO, _Point_CO);
				PointToPixel_CO(_Point_CO, _Pixel_CO);
				PointToPixel_IR(_Point_IR, _Pixel_IR);


				if (_Pixel_IR.x < 0 || _Pixel_IR.x >= IRIMX || _Pixel_IR.y < 0 || _Pixel_IR.y >= IRIMY)
					continue;
				if ((_Pixel_IR.x) < 20.0f || (_Pixel_IR.x) > IRIMX - 20.0 || (_Pixel_IR.y) < 20.0f || (_Pixel_IR.y) > IRIMY - 20.0f ||
					(_Pixel_CO.x) < 20.0f || (_Pixel_CO.x) > COIMX - 20.0 || (_Pixel_CO.y) < 20.0f || (_Pixel_CO.y) > COIMY - 20.0f)
				{

				}
				else
				{

					float _Depth = (float)depthImages[i].at<unsigned short>((int)_Pixel_IR.y, (int)_Pixel_IR.x) / 1000.0f;

					if (abs(_Depth - _Point_IR.z) < DTEST) {
						mDummyImage entry;
						entry._Img = i;
						entry._Img_Tex.x = _Pixel_CO.x;
						entry._Img_Tex.y = _Pixel_CO.y;
						entry._Weight = 1.0f;
						//float weight_max = 0.0;
						//bool is_valid = true;
						float weight = 0.0;
						float3 avgnormal = make_float3(0.0);
						for (auto vf_idx : layer_Vertices[l][vi]._Triangles) {
							avgnormal += normals_vec[i][vf_idx];
							//float _Dot = _Pose.at<float>(0, 2) * normals_vec[i][vf_idx].x + _Pose.at<float>(1, 2) * normals_vec[i][vf_idx].y + _Pose.at<float>(2, 2) * normals_vec[i][vf_idx].z;
							//weight_max = std::max(normal_direction * _Dot, weight_max);
						}
						avgnormal /= layer_Vertices[l][vi]._Triangles.size();
						float _Dot = _Pose.at<float>(0, 2) * avgnormal.x + _Pose.at<float>(1, 2) * avgnormal.y + _Pose.at<float>(2, 2) * avgnormal.z;
						if (normal_direction * _Dot > 0) {
							entry._Weight *= normal_direction * _Dot;
							candidateImages.push_back(entry);
						}

						//entry._Weight *= weight_max;
						//if (is_valid)
							//candidateImages.push_back(entry);
					}
				}
			}
			std::sort(candidateImages.begin(), candidateImages.end());
			for (auto& e : candidateImages) {
				ref_imgs_set_ver[l][vi].insert(make_pair(e._Img, e._Img_Tex));
			}

			if (l < LAYERNUM - 1) {
				for (auto vh : tmp_layer_Vertices[vi].verHistory) {
					int vs = simpleifier_ptr->idx_table_Ver[LAYERNUM - l - 2][vh];
					if (vs < 0)
						continue;
					for (auto v_im : layer_Vertices[l + 1][vs]._Img) {
						ref_imgs_set_pre_ver[l][vi].insert(v_im);
					}
				}
			}
		}

		//tri's visble img initialize
		for (int fi = 0; fi < tmp_sim_numTri; fi++) {
			set<uint> tmp_v_imgs[3];
			for (int fv_i = 0; fv_i < 3; fv_i++) {
				for (auto im_idx : ref_imgs_set_ver[l][layer_Triangles[l][fi]._Vertices[fv_i]]) {
					tmp_v_imgs[fv_i].insert(im_idx.first);
				}
			}
			set<uint> inter_result, inter_result01;

			for (auto im_idx : tmp_v_imgs[0]) {
				if (tmp_v_imgs[1].find(im_idx) != tmp_v_imgs[1].end())
					inter_result01.insert(im_idx);
			}
			for (auto im_idx : inter_result01) {
				if (tmp_v_imgs[2].find(im_idx) != tmp_v_imgs[2].end())
					inter_result.insert(im_idx);
			}

			map<int, float> tmp_ref_imgs;
			vector<pair<int, float>> tmp_ref_imgs_vec;

			for (auto im_idx : inter_result) {
				int fvi0 = simpleifier_ptr->extTri[LAYERNUM - l - 1][fi]._Vertices[0];
				int fvi1 = simpleifier_ptr->extTri[LAYERNUM - l - 1][fi]._Vertices[1];
				int fvi2 = simpleifier_ptr->extTri[LAYERNUM - l - 1][fi]._Vertices[2];
				float3 ray = tri_positions_vec[im_idx][fi] / length(tri_positions_vec[im_idx][fi]);

				float _Dot = _Pose.at<float>(0, 2) * normals_vec[im_idx][fi].x + _Pose.at<float>(1, 2) * normals_vec[im_idx][fi].y + _Pose.at<float>(2, 2) * normals_vec[im_idx][fi].z;
				//float _Dot = ray.x * normals_vec[im_idx][fi].x + ray.y * normals_vec[im_idx][fi].y + ray.z * normals_vec[im_idx][fi].z;
				if (_Dot * normal_direction > 0.0) {
					tmp_ref_imgs.insert(make_pair(im_idx, normal_direction * _Dot));
				}
				if (_Dot * normal_direction > 0.0) {
					/*if (fi > 0) {
						int fvi0_pre = simpleifier_ptr->extTri[LAYERNUM - l - 1][fi - 1]._Vertices[0];
						float3 v_pre = simpleifier_ptr->extVer[LAYERNUM - l - 1][fvi0_pre]._Pos[im_idx];
						float3 v_now = simpleifier_ptr->extVer[LAYERNUM - l - 1][fvi0]._Pos[im_idx];
						float2 p_pre, p_now;
						PointToPixel_CO(v_pre, p_pre);
						PointToPixel_CO(v_now, p_now);
						if (length(p_now - p_pre) > 10)
							_Dot /= 0.1 * (length(p_now - p_pre));
					}*/
					//float weight = areas_vec[im_idx][fi] / max(1.0, 0.05 * tri_trans_vec_image[im_idx][fi]);
					//여기여기여기
					/*if (im_idx % 3 == 0)
						continue;*/
					float weight = areas_vec[im_idx][fi];
					//_Dot /= max(1.0, 0.05 * tri_trans_vec_image[im_idx][fi]);

					ref_imgs_set_tri[l][fi].insert(make_pair(im_idx, normal_direction * _Dot));
					ref_imgs_set_tri_ori[l][fi].insert(make_pair(im_idx, normal_direction * _Dot));
					/*ref_imgs_set_tri[l][fi].insert(make_pair(im_idx, weight));
					ref_imgs_set_tri_ori[l][fi].insert(make_pair(im_idx, weight));*/
					//pixel area
					/*ref_imgs_set_tri[l][fi].insert(make_pair(im_idx, areas_vec[im_idx][fi]));
					ref_imgs_set_tri_ori[l][fi].insert(make_pair(im_idx, areas_vec[im_idx][fi]));*/
				}				
			}

			// 여기여기
			/*for (auto im_idx_weight : tmp_ref_imgs) {
				tmp_ref_imgs_vec.push_back(make_pair(im_idx_weight.first, im_idx_weight.second));
			}
			sort(tmp_ref_imgs_vec.begin(), tmp_ref_imgs_vec.end(), [](const pair<int, float>& l, const pair<int, float>& r) {return l.second < r.second; });
			int sort_count = 0;
			for (auto sort_im_idx_weight : tmp_ref_imgs_vec) {
				if (sort_im_idx_weight.second > 0.3) {
					ref_imgs_set_tri[l][fi].insert(sort_im_idx_weight);
					ref_imgs_set_tri_ori[l][fi].insert(sort_im_idx_weight);
				}
				if (tmp_ref_imgs_vec.size() - sort_count <= 2 && ref_imgs_set_tri[l][fi].size() < 2) {
					ref_imgs_set_tri[l][fi].insert(sort_im_idx_weight);
					ref_imgs_set_tri_ori[l][fi].insert(sort_im_idx_weight);
				}
				sort_count++;
			}*/

			// for coherency (fine to coarse)
			// union of finer level result
			if (l < LAYERNUM - 1) {
				set<uint> tmp_pre_v_imgs[3];
				for (int fv_i = 0; fv_i < 3; fv_i++) {
					for (auto i_idx : ref_imgs_set_pre_ver[l][layer_Triangles[l][fi]._Vertices[fv_i]]) {
						tmp_pre_v_imgs[fv_i].insert(i_idx);
					}
				}
				set<uint> pre_inter_result, pre_inter_result01;

				for (auto im_idx : tmp_pre_v_imgs[0]) {
					if (tmp_pre_v_imgs[1].find(im_idx) != tmp_pre_v_imgs[1].end())
						pre_inter_result01.insert(im_idx);
				}
				for (auto im_idx : pre_inter_result01) {
					if (tmp_pre_v_imgs[2].find(im_idx) != tmp_pre_v_imgs[2].end())
						pre_inter_result.insert(im_idx);
				}
				for (auto im_idx : pre_inter_result) {
					ref_imgs_set_pre_tri[l][fi].insert(im_idx);
				}
			}
		}

		// line projection
		if (RECORD_UNIT) {
			for (int tttt = 0; tttt < imgNum; tttt++) {
				cv::Mat tmp;
				cv::cvtColor(colorImages[tttt], tmp, CV_BGRA2BGR);
				for (int i = 0; i < tmp_sim_numTri; i++) {
					for (auto tmp_t_img : ref_imgs_set_tri_ori[l][i]) {
						if (tmp_t_img.first == tttt) {
							float2 uv1;
							float2 uv2;
							float2 uv3;
							if (ref_imgs_set_ver[l][layer_Triangles[l][i]._Vertices[0]].find(tttt) == ref_imgs_set_ver[l][layer_Triangles[l][i]._Vertices[0]].end())
								break;
							if (ref_imgs_set_ver[l][layer_Triangles[l][i]._Vertices[1]].find(tttt) == ref_imgs_set_ver[l][layer_Triangles[l][i]._Vertices[1]].end())
								break;
							if (ref_imgs_set_ver[l][layer_Triangles[l][i]._Vertices[2]].find(tttt) == ref_imgs_set_ver[l][layer_Triangles[l][i]._Vertices[2]].end())
								break;

							uv1 = ref_imgs_set_ver[l][layer_Triangles[l][i]._Vertices[0]].find(tttt)->second;
							uv2 = ref_imgs_set_ver[l][layer_Triangles[l][i]._Vertices[1]].find(tttt)->second;
							uv3 = ref_imgs_set_ver[l][layer_Triangles[l][i]._Vertices[2]].find(tttt)->second;

							cv::line(tmp, cv::Point(uv1.x, uv1.y), cv::Point(uv2.x, uv2.y), cv::Scalar(255, 0, 0), 1);
							cv::line(tmp, cv::Point(uv2.x, uv2.y), cv::Point(uv3.x, uv3.y), cv::Scalar(255, 0, 0), 1);
							cv::line(tmp, cv::Point(uv3.x, uv3.y), cv::Point(uv1.x, uv1.y), cv::Scalar(255, 0, 0), 1);
						}
					}
				}
				cv::imwrite(std::string(data_root_path + "/unit_test/" + unit_test_path + "/projection/" + std::to_string(l)) + "/sampled_Frame_" + std::to_string(tttt) + std::string(".png"), tmp);
			}
		}

		//tri's 1 ring neighbor tri initialize
		/*vector<set<uint>> union_tri;
		union_tri.resize(tmp_sim_numTri);
		for (int fi = 0; fi < tmp_sim_numTri; fi++) {
			for (int fv_i = 0; fv_i < 3; fv_i++) {
				for (auto fvf_idx : layer_Vertices[l][layer_Triangles[l][fi]._Vertices[fv_i]]._Triangles)
					union_tri[fi].insert(fvf_idx);
			}
		}*/
		
		// assign values
		for (int vi = 0; vi < tmp_sim_numVer; vi++) {
			set<uint> one_ring_img;
			for (auto vf_idx : layer_Vertices[l][vi]._Triangles) {
				for (auto im_idx : ref_imgs_set_tri[l][vf_idx])
					one_ring_img.insert(im_idx.first);
			}
			for (auto im_idx : one_ring_img) {
				auto e = ref_imgs_set_ver[l][vi].find(im_idx);
				layer_Vertices[l][vi]._Img.push_back(e->first);
				layer_Vertices[l][vi]._Img_Tex.push_back(e->second);
				layer_Vertices[l][vi]._Img_Tex_ori.push_back(e->second);
			}
			layer_Vertices[l][vi]._Pos_time.resize(imgNum);
			layer_Vertices[l][vi]._Norm_time.resize(imgNum);
			for (int ai = 0; ai < imgNum; ai++) {
				layer_Vertices[l][vi]._Pos_time[ai] = positions_vec[ai][vi];
				float3 normnorm = { 0.0 };
				for (auto vf_idx : layer_Vertices[l][vi]._Triangles) {
					normnorm += normals_vec[ai][vf_idx];
				}
				layer_Vertices[l][vi]._Norm_time[ai] = normnorm / layer_Vertices[l][vi]._Triangles.size();
			}
		}
		for (int fi = 0; fi < tmp_sim_numTri; fi++) {
			for (auto im_idx : ref_imgs_set_tri[l][fi]) {
				layer_Triangles[l][fi]._Img.push_back(im_idx.first);
				layer_Triangles[l][fi]._Bound.push_back(false);
				layer_Triangles[l][fi]._Label.push_back(false);
				layer_Triangles[l][fi]._Img_Weight.push_back(im_idx.second);
			}
			layer_Triangles[l][fi]._Area.resize(imgNum);
			for (int ai = 0; ai < imgNum; ai++)
				layer_Triangles[l][fi]._Area[ai] = simpleifier_ptr->extTri[LAYERNUM - l - 1][fi]._Area[ai];
		}

		// weight normalization
		for (int fi = 0; fi < tmp_sim_numTri; fi++) {
			float wsum = 0;
			for (auto im_weight : layer_Triangles[l][fi]._Img_Weight) wsum += im_weight;
			for (int wi = 0; wi < layer_Triangles[l][fi]._Img_Weight.size(); wi++)
				layer_Triangles[l][fi]._Img_Weight[wi] /= wsum;
		}

		positions_vec.clear();
		normals_vec.clear();
		pixel_vec.clear();
		areas_vec.clear();
	}

	/*vector<int> cccount;
	cccount.resize(imgNum, 0);
	for (auto tri : layer_Triangles[LAYERNUM - 1]) {
		for (auto i : tri._Img) {
			cccount[i]++;
		}
	}
	cout << "Tri Count: " << layer_Triangles[LAYERNUM - 1].size() << endl;
	for (int i = 0; i < imgNum; i++) {
		cout << cccount[i] << endl;
	}*/

	cout << "Convert tree Done" << endl;
}

void Mapper4D::BackgroundSubtraction(bool use_keyframe) {
	bool ori_mode = m_use_keyframe;
	SetValidFrame(use_keyframe);
	if (use_keyframe) {
		for (int layer = 0; layer < LAYERNUM; layer++) {
			cout << "background subtraction: " << layer;
			int erase_count = 0;
			for (auto &hT : layer_Triangles_key[layer]) {
				hostTriangle tempTri;
				int vi[3];
				vi[0] = hT._Vertices[0];
				vi[1] = hT._Vertices[1];
				vi[2] = hT._Vertices[2];
				for (int i = 0; i < hT._Img.size();) {
					int im_idx = hT._Img[i];
					bool valid = true;
					for (int j = 0; j < 3; j++) {
						float2 p = layer_Vertices_key[layer][vi[j]]._Img_Tex[layer_Vertices_key[layer][vi[j]].getImgIdx(im_idx)];
						if (maskImages[validFrame_table[im_idx]].data[(uint)(p.y + 0.5) * COIMX + (uint)(p.x + 0.5)] < 1.0) {
							valid = false;
							break;
						}
					}
					/*if (!valid)
						hT._Img_Weight[i] *= 0.001;
					i++;*/
					if (!valid) {
						hT._Img.erase(hT._Img.begin() + i);
						hT._Img_Weight.erase(hT._Img_Weight.begin() + i);
						hT._Bound.erase(hT._Bound.begin() + i);
						hT._Label.erase(hT._Label.begin() + i);
						erase_count++;
					}
					else
						i++;
				}
				float n_weight_sum = 0.0;
				for (int i = 0; i < hT._Img.size(); i++) {
					n_weight_sum += hT._Img_Weight[i];
				}
				for (int i = 0; i < hT._Img.size(); i++) {
					hT._Img_Weight[i] /= n_weight_sum;
				}
			}
			cout << " ==> erase count: " << erase_count << endl;
			for (auto &hV : layer_Vertices_key[layer]) {
				set<uint> tri_im_set;
				for (auto const &ti : hV._Triangles) {
					for (auto im_idx_t : layer_Triangles_key[layer][ti]._Img) {
						tri_im_set.insert(im_idx_t);
					}
				}
				for (int i = 0; i < hV._Img.size();) {
					if (tri_im_set.find(hV._Img[i]) == tri_im_set.end()) {
						hV._Img.erase(hV._Img.begin() + i);
						hV._Img_Tex.erase(hV._Img_Tex.begin() + i);
						hV._Img_Tex_ori.erase(hV._Img_Tex_ori.begin() + i);
					}
					else
						i++;
				}
			}
		}
	}
	else {
		for (int layer = 0; layer < LAYERNUM; layer++) {
			cout << "background subtraction: " << layer;
			int erase_count = 0;
			for (auto &hT : layer_Triangles[layer]) {
				hostTriangle tempTri;
				int vi[3];
				vi[0] = hT._Vertices[0];
				vi[1] = hT._Vertices[1];
				vi[2] = hT._Vertices[2];
				for (int i = 0; i < hT._Img.size();) {
					int im_idx = hT._Img[i];
					bool valid = true;
					for (int j = 0; j < 3; j++) {
						float2 p = layer_Vertices[layer][vi[j]]._Img_Tex[layer_Vertices[layer][vi[j]].getImgIdx(im_idx)];
						if (maskImages[validFrame_table[im_idx]].data[(uint)(p.y + 0.5) * COIMX + (uint)(p.x + 0.5)] < 1.0) {
							valid = false;
							break;
						}
					}
					/*if (!valid)
						hT._Img_Weight[i] *= 0.001;
					i++;*/
					if (!valid) {
						hT._Img.erase(hT._Img.begin() + i);
						hT._Img_Weight.erase(hT._Img_Weight.begin() + i);
						hT._Bound.erase(hT._Bound.begin() + i);
						hT._Label.erase(hT._Label.begin() + i);
						erase_count++;
					}
					else
						i++;
				}
				float n_weight_sum = 0.0;
				for (int i = 0; i < hT._Img.size(); i++) {
					n_weight_sum += hT._Img_Weight[i];
				}
				if(n_weight_sum > 0)
					for (int i = 0; i < hT._Img.size(); i++) {
						hT._Img_Weight[i] /= n_weight_sum;
					}
			}
			cout << " ==> erase count: " << erase_count << endl;
			for (auto &hV : layer_Vertices[layer]) {
				set<uint> tri_im_set;
				for (auto const &ti : hV._Triangles) {
					for (auto im_idx_t : layer_Triangles[layer][ti]._Img) {
						tri_im_set.insert(im_idx_t);
					}
				}
				for (int i = 0; i < hV._Img.size();) {
					if (tri_im_set.find(hV._Img[i]) == tri_im_set.end()) {
						hV._Img.erase(hV._Img.begin() + i);
						hV._Img_Tex.erase(hV._Img_Tex.begin() + i);
						hV._Img_Tex_ori.erase(hV._Img_Tex_ori.begin() + i);
					}
					else
						i++;
				}
			}
		}
	}
	SetValidFrame(ori_mode);
}

void Mapper4D::SetPropVec(vector<vector<float2>> propVec, int layer) {

	vector<int> parentNode(simpleifier_ptr->extVer[LAYERNUM - 1 - layer].size() , -1);

	for (int i = 0; i < simpleifier_ptr->extVer[LAYERNUM - layer].size(); i++) {
		int v_Img_Num = layer_Vertices[layer - 1][i]._Img.size();
		for (auto vh : simpleifier_ptr->extVer[LAYERNUM - layer][i].verHistory) {
			for (int j = 0; j < v_Img_Num; j++) {
				int k = layer_Vertices[layer-1][i]._Img[j];
				propAccum[k][vh] += propVec[k][i];
				propValid[k][vh] = true;
			}

			//link parent node
			int tmp_idx = simpleifier_ptr->idx_table_Ver[LAYERNUM - 1 - layer][vh];
			if (tmp_idx >= 0)
				parentNode[tmp_idx] = i;
		}
	}
	for (int i = 0; i < numVer; i++) {
		int tmp_idx = simpleifier_ptr->idx_table_Ver[LAYERNUM - 1 - layer][i];
		if (tmp_idx < 0)
			continue;
		
		int v_Img_Num = layer_Vertices[layer][tmp_idx]._Img.size();

		set<unsigned int> ringv_idx;
		//(v1 average)->(v2 weighted sum) of parent 1 ring node
		for (int ti = 0; ti < layer_Vertices[layer-1][parentNode[tmp_idx]]._Triangles.size(); ti++) {
			unsigned int ringTri = layer_Vertices[layer-1][parentNode[tmp_idx]]._Triangles[ti];
			for (auto rv : layer_Triangles[layer-1][ringTri]._Vertices) {
				ringv_idx.insert(simpleifier_ptr->idx_table_Ver_inv[LAYERNUM - layer][rv]);
			}
		}
		
		for (int j = 0; j < v_Img_Num; j++) {
			int k = layer_Vertices[layer][tmp_idx]._Img[j];
			float2 meanProp;
			meanProp.x = 0; meanProp.y = 0;
			int n_move = 0;
			float weight_move = 0.0;
			for (auto rv : ringv_idx) {
				int parentLocalrv = simpleifier_ptr->idx_table_Ver[LAYERNUM - layer][rv];
				if (parentLocalrv < 0)
					continue;
				int parentImgIdx = layer_Vertices[layer - 1][parentLocalrv].getImgIdx(k);
				if (propValid[k][rv] && parentImgIdx >= 0) {
					float weight = 1.0 / exp(abs(Distance(layer_Vertices[layer][tmp_idx]._Img_Tex[j], layer_Vertices[layer - 1][parentLocalrv]._Img_Tex[parentImgIdx])));
					meanProp += propAccum[k][rv] * weight;
					weight_move += weight;
					n_move++;
				}
			}
			if (weight_move > 0 && n_move > 0)
				meanProp /= weight_move;
			layer_Vertices[layer][tmp_idx]._Img_Tex[j] += meanProp;
			layer_Vertices[layer][tmp_idx]._Img_Tex[j].x = clamp(layer_Vertices[layer][tmp_idx]._Img_Tex[j].x, (float)0, (float)COIMX - 1);
			layer_Vertices[layer][tmp_idx]._Img_Tex[j].y = clamp(layer_Vertices[layer][tmp_idx]._Img_Tex[j].y, (float)0, (float)COIMY - 1);
			/*layer_Vertices[layer][tmp_idx]._Img_Tex_ori[j] += meanProp;
			layer_Vertices[layer][tmp_idx]._Img_Tex_ori[j].x = clamp(layer_Vertices[layer][tmp_idx]._Img_Tex[j].x, (float)0, (float)COIMX - 1);
			layer_Vertices[layer][tmp_idx]._Img_Tex_ori[j].y = clamp(layer_Vertices[layer][tmp_idx]._Img_Tex[j].y, (float)0, (float)COIMY - 1);*/
		}
	}
}

void Mapper4D::SetPropVec_key(vector<vector<float2>> propVec, int layer) {
	// last update
	if (layer < 0) {
		for (int i = 0; i < simpleifier_ptr->extVer[0].size(); i++) {
			int v_Img_Num = layer_Vertices_key[LAYERNUM - 1][i]._Img.size();
			for (auto vh : simpleifier_ptr->extVer[0][i].verHistory) {
				for (int j = 0; j < v_Img_Num; j++) {
					int k = layer_Vertices_key[LAYERNUM - 1][i]._Img[j];
					propAccum[k][vh] += propVec[k][i];
					propValid[k][vh] = true;
				}
			}
		}
		return;
	}

	vector<int> parentNode(simpleifier_ptr->extVer[LAYERNUM - 1 - layer].size(), -1);

	for (int i = 0; i < simpleifier_ptr->extVer[LAYERNUM - layer].size(); i++) {
		int v_Img_Num = layer_Vertices_key[layer - 1][i]._Img.size();
		for (auto vh : simpleifier_ptr->extVer[LAYERNUM - layer][i].verHistory) {
			for (int j = 0; j < v_Img_Num; j++) {
				int k = layer_Vertices_key[layer - 1][i]._Img[j];
				propAccum[k][vh] += propVec[k][i];
				propValid[k][vh] = true;
			}

			//link parent node
			int tmp_idx = simpleifier_ptr->idx_table_Ver[LAYERNUM - 1 - layer][vh];
			if (tmp_idx >= 0)
				parentNode[tmp_idx] = i;
		}
	}
	for (int i = 0; i < numVer; i++) {
		int tmp_idx = simpleifier_ptr->idx_table_Ver[LAYERNUM - 1 - layer][i];
		if (tmp_idx < 0)
			continue;

		int v_Img_Num = layer_Vertices_key[layer][tmp_idx]._Img.size();

		set<unsigned int> ringv_idx;
		//(v1 average)->(v2 weighted sum) of parent 1 ring node
		for (int ti = 0; ti < layer_Vertices_key[layer - 1][parentNode[tmp_idx]]._Triangles.size(); ti++) {
			unsigned int ringTri = layer_Vertices_key[layer - 1][parentNode[tmp_idx]]._Triangles[ti];
			for (auto rv : layer_Triangles_key[layer - 1][ringTri]._Vertices) {
				ringv_idx.insert(simpleifier_ptr->idx_table_Ver_inv[LAYERNUM - layer][rv]);
			}
		}

		for (int j = 0; j < v_Img_Num; j++) {
			int k = layer_Vertices_key[layer][tmp_idx]._Img[j];
			float2 meanProp;
			meanProp.x = 0; meanProp.y = 0;
			int n_move = 0;
			float weight_move = 0.0;
			for (auto rv : ringv_idx) {
				int parentLocalrv = simpleifier_ptr->idx_table_Ver[LAYERNUM - layer][rv];
				if (parentLocalrv < 0)
					continue;
				int parentImgIdx = layer_Vertices_key[layer - 1][parentLocalrv].getImgIdx(k);
				if (propValid[k][rv] && parentImgIdx >= 0) {
					float weight = 1.0 / exp(abs(Distance(layer_Vertices_key[layer][tmp_idx]._Img_Tex_ori[j], layer_Vertices_key[layer - 1][parentLocalrv]._Img_Tex_ori[parentImgIdx])));
					meanProp += propAccum[k][rv] * weight;
					weight_move += weight;
					n_move++;
				}
			}
			if (weight_move > 0 && n_move > 0)
				meanProp /= weight_move;
			layer_Vertices_key[layer][tmp_idx]._Img_Tex[j] += meanProp;
			layer_Vertices_key[layer][tmp_idx]._Img_Tex[j].x = clamp(layer_Vertices_key[layer][tmp_idx]._Img_Tex[j].x, (float)0, (float)COIMX - 1);
			layer_Vertices_key[layer][tmp_idx]._Img_Tex[j].y = clamp(layer_Vertices_key[layer][tmp_idx]._Img_Tex[j].y, (float)0, (float)COIMY - 1);
			/*layer_Vertices_key[layer][tmp_idx]._Img_Tex_ori[j] += meanProp;
			layer_Vertices_key[layer][tmp_idx]._Img_Tex_ori[j].x = clamp(layer_Vertices_key[layer][tmp_idx]._Img_Tex[j].x, (float)0, (float)COIMX - 1);
			layer_Vertices_key[layer][tmp_idx]._Img_Tex_ori[j].y = clamp(layer_Vertices_key[layer][tmp_idx]._Img_Tex[j].y, (float)0, (float)COIMY - 1);*/
		}
	}
}

void Mapper4D::BackPropVec_key() {
	ResetTex_key();

	for (int layer = 0; layer < LAYERNUM; layer++) {
		for (int i = 0; i < numVer; i++) {
			int tmp_idx = simpleifier_ptr->idx_table_Ver[LAYERNUM - 1 - layer][i];
			if (tmp_idx < 0)
				continue;

			set<unsigned int> ringv_idx;
			int v_Img_Num = layer_Vertices_key[layer][tmp_idx]._Img.size();
			for (int j = 0; j < v_Img_Num; j++) {
				float2 meanProp = make_float2(0);
				int n_move = 0;
				float weight_move = 0.0;
				for (auto vh : simpleifier_ptr->extVer[LAYERNUM - 1 - layer][tmp_idx].verHistory) {
					int k = layer_Vertices_key[layer][tmp_idx]._Img[j];

					int parentImgIdx = layer_Vertices_key[LAYERNUM - 1][vh].getImgIdx(k);

					if (propValid[k][vh] && parentImgIdx >= 0) {
						float weight = 1.0 / exp(abs(Distance(layer_Vertices_key[layer][tmp_idx]._Img_Tex_ori[j], layer_Vertices_key[LAYERNUM - 1][vh]._Img_Tex_ori[parentImgIdx])));
						meanProp += propAccum[k][vh] * weight;
						weight_move += weight;
						n_move++;
					}
				}
				if (weight_move > 0 && n_move > 0)
					meanProp /= weight_move;
				layer_Vertices_key[layer][tmp_idx]._Img_Tex[j] += meanProp;
				layer_Vertices_key[layer][tmp_idx]._Img_Tex[j].x = clamp(layer_Vertices_key[layer][tmp_idx]._Img_Tex[j].x, (float)0, (float)COIMX - 1);
				layer_Vertices_key[layer][tmp_idx]._Img_Tex[j].y = clamp(layer_Vertices_key[layer][tmp_idx]._Img_Tex[j].y, (float)0, (float)COIMY - 1);
			}
		}
	}

	propAccum.clear();
	propValid.clear();
	propAccum.resize(imgNum);
	propValid.resize(imgNum);
	float2 dudvInit;
	dudvInit.x = 0.0;
	dudvInit.y = 0.0;
	for (int i = 0; i < imgNum; i++) {
		propAccum[i].resize(numVer, dudvInit);
		propValid[i].resize(numVer, false);
	}
}

void Mapper4D::SetValidFrame(bool use_keyframe) {
	m_use_keyframe = use_keyframe;
	if (use_keyframe) {
		validFrame_table.clear();
		validFrame_table.assign(keyFrame_idx.begin(), keyFrame_idx.end());
	}
	else {
		validFrame_table.clear();
		for (int i = 0; i < imgNum; i++) {
			validFrame_table.push_back(i);
		}
	}

}

void Mapper4D::ResetTex() {
	for (int vi = 0; vi < layer_numVer[LAYERNUM - 1]; vi++) {
		layer_Vertices[LAYERNUM - 1][vi]._Img_Tex.clear();
		layer_Vertices[LAYERNUM - 1][vi]._Img_Tex.assign(layer_Vertices[LAYERNUM - 1][vi]._Img_Tex_ori.begin(), layer_Vertices[LAYERNUM - 1][vi]._Img_Tex_ori.end());
	}
}

void Mapper4D::ResetTex_key() {
	for (int vi = 0; vi < layer_numVer[LAYERNUM - 1]; vi++) {
		layer_Vertices_key[LAYERNUM - 1][vi]._Img_Tex.clear();
		layer_Vertices_key[LAYERNUM - 1][vi]._Img_Tex.assign(layer_Vertices_key[LAYERNUM - 1][vi]._Img_Tex_ori.begin(), layer_Vertices_key[LAYERNUM - 1][vi]._Img_Tex_ori.end());
	}
}

void Mapper4D::ResetBound() {
	for (int layer = 0; layer < LAYERNUM; layer++) {
		for (int fi = 0; fi < layer_numTri[layer]; fi++) {
			for (int i = 0; i < layer_Triangles[layer][fi]._Img.size(); i++) {
				layer_Triangles[layer][fi]._Bound[i] = false;
				layer_Triangles[layer][fi]._Label[i] = false;
			}
		}
	}

}

void Mapper4D::DrawDisplacement() {

	for (int tttt = 0; tttt < imgNum; tttt++) {
		cv::Mat tmp;
		cv::cvtColor(colorImages[tttt], tmp, CV_BGRA2BGR);
		for (int vi = 0; vi < numVer; vi++) {
			for (int im_idx = 0; im_idx < layer_Vertices[LAYERNUM - 1][vi]._Img.size(); im_idx++) {
				if (layer_Vertices[LAYERNUM - 1][vi]._Img[im_idx] == tttt) {
					float2 disp = layer_Vertices[LAYERNUM - 1][vi]._Img_Tex[im_idx] - layer_Vertices[LAYERNUM - 1][vi]._Img_Tex_ori[im_idx];
					disp /= length(disp);					
					cv::Point arrow_base[2];
					arrow_base[0].x = layer_Vertices[LAYERNUM - 1][vi]._Img_Tex_ori[im_idx].x + -disp.y;
					arrow_base[0].y = layer_Vertices[LAYERNUM - 1][vi]._Img_Tex_ori[im_idx].y + disp.x;
					arrow_base[1].x = layer_Vertices[LAYERNUM - 1][vi]._Img_Tex_ori[im_idx].x + disp.y;
					arrow_base[1].y = layer_Vertices[LAYERNUM - 1][vi]._Img_Tex_ori[im_idx].y + -disp.x;
					cv::Point ori_p = cv::Point(layer_Vertices[LAYERNUM - 1][vi]._Img_Tex_ori[im_idx].x, layer_Vertices[LAYERNUM - 1][vi]._Img_Tex_ori[im_idx].y);
					cv::Point opt_p = cv::Point(layer_Vertices[LAYERNUM - 1][vi]._Img_Tex[im_idx].x, layer_Vertices[LAYERNUM - 1][vi]._Img_Tex[im_idx].y);
					/*vector<vector<cv::Point>> contours;
					contours.resize(1);
					contours[0].push_back(opt_p);
					contours[0].push_back(arrow_base[0]);
					contours[0].push_back(arrow_base[1]);
					cv::drawContours(tmp, contours, 0, cv::Scalar(0, 0, 255), -1);
					cv::circle(tmp, ori_p, 2, cv::Scalar(255, 0, 0), 1);*/

					cv::line(tmp, opt_p, ori_p, cv::Scalar(255, 0, 0), 1);
					cv::circle(tmp, ori_p, 1, cv::Scalar(0, 0, 255), -1);
				}
			}
		}
		cv::imwrite(data_root_path + "/unit_test/" + unit_test_path + "/projection/disp_Frame_" + std::to_string(tttt) + std::string(".png"), tmp);
	}	
}

void Mapper4D::RecomputeWeight() {
	for (int ti = 0; ti < numTri; ti++) {
		for (int tri_im_idx = 0; tri_im_idx < layer_Triangles[LAYERNUM - 1][ti]._Img.size(); tri_im_idx++) {
			layer_Triangles[LAYERNUM - 1][ti]._Img_Weight[tri_im_idx] = 0;
		}
	}
	for (int i = 0; i < imgNum; i++) {
		cv::Mat tmpImage;
		cv::Mat grayImage;
		cv::cvtColor(colorImages[i], tmpImage, CV_BGR2GRAY);
		cv::GaussianBlur(tmpImage, tmpImage, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
		tmpImage.convertTo(grayImage, CV_32FC1);
		grayImage /= 255.0;
		cv::Mat grad_u, grad_v;
		cv::Sobel(grayImage, grad_u, CV_32FC1, 0, 1);
		cv::Sobel(grayImage, grad_v, CV_32FC1, 1, 0);
		/*grad_u /= 4.0;
		grad_v /= 4.0;*/
		cv::cuda::GpuMat d_grad_u(grad_u);
		cv::cuda::GpuMat d_grad_v(grad_v);
		cv::cuda::GpuMat d_grad;
		cv::cuda::pow(d_grad_u, 2, d_grad_u);
		cv::cuda::pow(d_grad_v, 2, d_grad_v);
		cv::cuda::add(d_grad_u, d_grad_v, d_grad);
		cv::cuda::sqrt(d_grad, d_grad);
		cv::Mat grad(d_grad);

		/*cv::imshow("111", grad);
		cv::waitKey(0);*/
		cv::Mat1i mask(grayImage.rows, grayImage.cols, -1);

		float* grad_data = (float*)grad.data;
		for (int ti = 0; ti < numTri; ti++) {
			if (layer_Triangles[LAYERNUM - 1][ti].getImgIdx(i) < 0)
				continue;
			vector<vector<cv::Point>> contours;
			contours.resize(1);
			cv::Point uv[3];
			cv::Point uv_v[2];
			for (int tvi = 0; tvi < 3; tvi++) {
				int v_im_idx = layer_Vertices[LAYERNUM - 1][layer_Triangles[LAYERNUM - 1][ti]._Vertices[tvi]].getImgIdx(i);
				uv[tvi] = cv::Point(layer_Vertices[LAYERNUM - 1][layer_Triangles[LAYERNUM - 1][ti]._Vertices[tvi]]._Img_Tex[v_im_idx].x, layer_Vertices[LAYERNUM - 1][layer_Triangles[LAYERNUM - 1][ti]._Vertices[tvi]]._Img_Tex[v_im_idx].y);
				contours[0].push_back(uv[tvi]);
			}
			uv_v[0] = uv[1] - uv[0];
			uv_v[1] = uv[2] - uv[0];
			double area = abs(uv_v[0].cross(uv_v[1]));
			if (area < 9.0) {
				cv::Point uv_center;
				uv_center = (uv[2] + uv[1] + uv[0]) / 3.0;
				int tri_im_idx = layer_Triangles[LAYERNUM - 1][ti].getImgIdx(i);
				float grad_p = grad_data[(int)uv_center.y * grayImage.cols + (int)uv_center.x];
				layer_Triangles[LAYERNUM - 1][ti]._Img_Weight[tri_im_idx] += grad_p * grad_p * area;
			}
			else
				cv::drawContours(mask, contours, 0, ti, -1);
		}
		/*cv::Mat1f maskf;
		mask.convertTo(maskf, CV_32FC1);
		maskf += 1.0;
		maskf /= numTri;
		cv::imshow("111", maskf);
		cv::waitKey(0);*/
		int* mask_data = (int*)mask.data;
		for (int hi = 0; hi < grayImage.rows; hi++) {
			for (int wi = 0; wi < grayImage.cols; wi++) {
				int ti = mask.data[hi * grayImage.cols + wi];
				if (ti < 0)
					continue;
				int tri_im_idx = layer_Triangles[LAYERNUM - 1][ti].getImgIdx(i);
				float grad_p = grad_data[hi * grayImage.cols + wi];
				layer_Triangles[LAYERNUM - 1][ti]._Img_Weight[tri_im_idx] += grad_p * grad_p;
			}
		}
	}

	// background weight * 0.0000001
	/*for (auto& hT : layer_Triangles[LAYERNUM - 1]) {
		hostTriangle tempTri;
		int vi[3];
		vi[0] = hT._Vertices[0];
		vi[1] = hT._Vertices[1];
		vi[2] = hT._Vertices[2];
		for (int i = 0; i < hT._Img.size(); i++) {
			int im_idx = hT._Img[i];
			bool valid = true;
			for (int j = 0; j < 3; j++) {
				float2 p = layer_Vertices[LAYERNUM - 1][vi[j]]._Img_Tex[layer_Vertices[LAYERNUM - 1][vi[j]].getImgIdx(im_idx)];
				if (maskImages[validFrame_table[im_idx]].data[(uint)(p.y + 0.5) * COIMX + (uint)(p.x + 0.5)] < 255.0) {
					valid = false;
					break;
				}
			}
			if (!valid) {
				hT._Img_Weight[i] *= 0.0000001;
			}
		}
	}*/

	// weight normalization
	for (int ti = 0; ti < numTri; ti++) {
		float wsum = 0;
		float wmin = 1000000.0;
		//cout << "+++++++++++++++++" << ti << "+++++++++++++++++" << endl;
		for (auto im_weight : layer_Triangles[LAYERNUM - 1][ti]._Img_Weight) {
			if (im_weight != 0 && wmin > im_weight)
				wmin = im_weight;
			wsum += im_weight;
		}
		for (int wi = 0; wi < layer_Triangles[LAYERNUM - 1][ti]._Img_Weight.size(); wi++) {
			if (layer_Triangles[LAYERNUM - 1][ti]._Img_Weight[wi] == 0) {
				layer_Triangles[LAYERNUM - 1][ti]._Img_Weight[wi] = wmin;
				wsum += wmin;
			}
		}
		//cout << "+++++++++++++++++" << wsum << "+++++++++++++++++" << endl;
		for (int wi = 0; wi < layer_Triangles[LAYERNUM - 1][ti]._Img_Weight.size(); wi++) {
			//cout << layer_Triangles[LAYERNUM - 1][ti]._Img_Weight[wi] << endl;
			layer_Triangles[LAYERNUM - 1][ti]._Img_Weight[wi] /= wsum;
		}
	}
}

void Mapper4D::RecomputeWeight_normal() {
	cv::Mat _P_inv_IR = (_Pose.inv());
	cv::Mat _P_inv_CO = (_CO_IR * _P_inv_IR);
	for (int fi = 0; fi < numTri; fi++) {

		int fvi0 = layer_Triangles[LAYERNUM - 1][fi]._Vertices[0];
		int fvi1 = layer_Triangles[LAYERNUM - 1][fi]._Vertices[1];
		int fvi2 = layer_Triangles[LAYERNUM - 1][fi]._Vertices[2];
		for (int k = 0; k < layer_Triangles[LAYERNUM - 1][fi]._Img.size(); k++) {
			int im = layer_Triangles[LAYERNUM - 1][fi]._Img[k];
			float3 _Point_D0 = layer_Vertices[LAYERNUM - 1][fvi0]._Pos_time[im];
			float3 _Point_D1 = layer_Vertices[LAYERNUM - 1][fvi1]._Pos_time[im];
			float3 _Point_D2 = layer_Vertices[LAYERNUM - 1][fvi2]._Pos_time[im];
			float3 _Point_CO0;
			float3 _Point_CO1;
			float3 _Point_CO2;

			Transform(_Point_D0, _P_inv_CO, _Point_CO0);
			Transform(_Point_D1, _P_inv_CO, _Point_CO1);
			Transform(_Point_D2, _P_inv_CO, _Point_CO2);

			float3 fvv1 = _Point_CO1 - _Point_CO0;
			float3 fvv2 = _Point_CO2 - _Point_CO0;

			float3 f_normal = cross(fvv1, fvv2);
			f_normal /= length(f_normal);

			float _Dot = _Pose.at<float>(0, 2) * f_normal.x + _Pose.at<float>(1, 2) * f_normal.y + _Pose.at<float>(2, 2) * f_normal.z;
			//float _Dot = ray.x * normals_vec[im_idx][fi].x + ray.y * normals_vec[im_idx][fi].y + ray.z * normals_vec[im_idx][fi].z;
			float weight = _Dot * normal_direction;

			layer_Triangles[LAYERNUM - 1][fi]._Img_Weight[k] = weight;
		}
	}
}

void lerp(vector<float2>& disp_key, vector<float2>& disp_all, vector<int>& valid_key, int start_idx, int end_idx) {
	int start_now = valid_key[start_idx] ? start_idx : -1;
	int end_now = valid_key[start_idx] ? start_idx + 1 : start_idx;
	while (end_now <= end_idx && end_now >= 0) {
		while (end_now <= end_idx) {
			if (valid_key[end_now])
				break;
			end_now++;
		}
		if (end_now > end_idx)
			end_now = -2;

		if (start_now < 0 && end_now < 0) {
			// do nothing
		}
		else if (start_now < 0) {
			for (int t = start_idx; t <= end_now; t++)
				disp_all[t] = disp_key[end_now];
		}
		else if (end_now < 0) {
			for (int t = start_now; t <= end_idx; t++)
				disp_all[t] = disp_key[start_now];
		}
		else {
			float len = end_now - start_now;
			for (int t = start_now; t <= end_now; t++) {
				float ratio = (end_now - t) / len;
				disp_all[t] = disp_key[start_now] * ratio + disp_key[end_now] * (1 - ratio);
			}

		}
		start_now = end_now;
		end_now++;
	}
}

void Mapper4D::InterpKeytoAll() {
	vector<vector<float2>> disp_key_vol;
	vector<vector<float2>> disp_all_vol;
	vector<vector<int>> valid_key_vol;
	vector<vector<int>> valid_all_vol;

	disp_key_vol.resize(numVer);
	disp_all_vol.resize(numVer);
	valid_key_vol.resize(numVer);
	valid_all_vol.resize(numVer);

	for (int vi = 0; vi < numVer; vi++) {
		// download
		disp_key_vol[vi].resize(imgNum, make_float2(0));
		disp_all_vol[vi].resize(imgNum, make_float2(0));
		valid_key_vol[vi].resize(imgNum, 0);
		valid_all_vol[vi].resize(imgNum, 0);
		for (int i = 0; i < layer_Vertices_key[LAYERNUM - 1][vi]._Img.size(); i++) {
			disp_key_vol[vi][keyFrame_idx[layer_Vertices_key[LAYERNUM - 1][vi]._Img[i]]] = layer_Vertices_key[LAYERNUM - 1][vi]._Img_Tex[i] - layer_Vertices_key[LAYERNUM - 1][vi]._Img_Tex_ori[i];
			valid_key_vol[vi][keyFrame_idx[layer_Vertices_key[LAYERNUM - 1][vi]._Img[i]]] = 1;
		}
		for (auto im : layer_Vertices[LAYERNUM - 1][vi]._Img) {
			valid_all_vol[vi][im] = 1;
		}
		
		// interpolation part
		bool in_chunk = false;
		int start_idx = 0;
		for (int t = 0; t < imgNum; t++) {
			// enter to the chunk
			if (valid_all_vol[vi][t] && !in_chunk) {
				start_idx = t;
				in_chunk = true;
			}
			// escape from chunk, conduct interpolation
			else if (!valid_all_vol[vi][t] && in_chunk) {
				lerp(disp_key_vol[vi], disp_all_vol[vi], valid_key_vol[vi], start_idx, t - 1);
				in_chunk = false;
			}
		}
		if (in_chunk)
			lerp(disp_key_vol[vi], disp_all_vol[vi], valid_key_vol[vi], start_idx, imgNum - 1);

		// upload
		for (int i = 0; i < layer_Vertices[LAYERNUM - 1][vi]._Img.size(); i++) {
			float2 n_tex = disp_all_vol[vi][layer_Vertices[LAYERNUM - 1][vi]._Img[i]] + layer_Vertices[LAYERNUM - 1][vi]._Img_Tex_ori[i];
			layer_Vertices[LAYERNUM - 1][vi]._Img_Tex[i] = make_float2(clamp(n_tex.x, 0.0, COIMX - 1.0), clamp(n_tex.y, 0.0, COIMY - 1.0));
		}
	}
}

void Mapper4D::SetBoundary_fine(vector<vector<vector<int>>> boundLabels, vector<vector<int>> resultLabels, vector<int> all2valid) {
	Bound_vec.resize(imgNum);
	Label_vec.resize(imgNum);
	for (int t = 0; t < imgNum; t++) {
		ResetBound();
		for (int fi = 0; fi < layer_numTri[LAYERNUM - 1]; fi++) {
			for (int i = 0; i < layer_Triangles[LAYERNUM - 1][fi]._Img.size(); i++) {
				if (find(boundLabels[t][fi].begin(), boundLabels[t][fi].end(), layer_Triangles[LAYERNUM - 1][fi]._Img[i]) != boundLabels[t][fi].end())
					layer_Triangles[LAYERNUM - 1][fi]._Bound[i] = true;
				if (all2valid[fi] >= 0)
					if (layer_Triangles[LAYERNUM - 1][fi]._Img[i] == resultLabels[t][all2valid[fi]])
						layer_Triangles[LAYERNUM - 1][fi]._Label[i] = true;
				/*Bound_vec[t].push_back(layer_Triangles[LAYERNUM - 1][fi]._Bound[i]);
				Label_vec[t].push_back(layer_Triangles[LAYERNUM - 1][fi]._Label[i]);*/
			}
			int b_size = Bound_vec[t].size();
			int l_size = Label_vec[t].size();
			int i_size = layer_Triangles[LAYERNUM - 1][fi]._Img.size();
			Bound_vec[t].resize(b_size + i_size);
			Label_vec[t].resize(l_size + i_size);
			
			std::copy(layer_Triangles[LAYERNUM - 1][fi]._Bound.begin(), layer_Triangles[LAYERNUM - 1][fi]._Bound.end(), Bound_vec[t].begin() + b_size);
			std::copy(layer_Triangles[LAYERNUM - 1][fi]._Label.begin(), layer_Triangles[LAYERNUM - 1][fi]._Label.end(), Label_vec[t].begin() + l_size);
		}
	}
}

void Mapper4D::Calpos(string streamPath, string NR_MeshPathAndPrefix, int startIdx, int endIdx) {
	char frameNameBuffer[512] = { 0 };

	std::ifstream img_dummy_file(intermediate_data_path + "stream.vat", std::ios::in | std::ios::binary);
	if (img_dummy_file.is_open() && FROM_FILE) {
		std::cout << "read file: " << intermediate_data_path + "stream.vat" << std::endl;

		int t_DepthImageWidth;
		int t_DepthImageHeight;
		int t_ColorImageWidth;
		int t_ColorImageHeight;
		int t_nImage = endIdx - startIdx;

		img_dummy_file.read((char*)&t_DepthImageWidth, sizeof(int));
		img_dummy_file.read((char*)&t_DepthImageHeight, sizeof(int));
		img_dummy_file.read((char*)&t_ColorImageWidth, sizeof(int));
		img_dummy_file.read((char*)&t_ColorImageHeight, sizeof(int));

		colorImages.resize(t_nImage);
		depthImages.resize(t_nImage);

		for (int i = 0; i < t_nImage; i++) {
			colorImages[i].create(t_ColorImageHeight, t_ColorImageWidth, CV_8UC3);
			depthImages[i].create(t_DepthImageHeight, t_DepthImageWidth, CV_16UC1);
			img_dummy_file.read((char*)colorImages[i].data, sizeof(uchar) * 3 * t_ColorImageWidth * t_ColorImageHeight);
			img_dummy_file.read((char*)depthImages[i].data, sizeof(ushort) * t_DepthImageWidth * t_DepthImageHeight);
		}
		img_dummy_file.close();
	}
	else {
		std::cout << "images read" << std::endl;
		std::ofstream img_dummy_file_w;
		img_dummy_file_w.open(intermediate_data_path + "stream.vat", std::ofstream::binary);
		for (int frameIdx = startIdx; frameIdx < endIdx; ++frameIdx)
		{
			printProgBar(((float)frameIdx / (endIdx - 1.0)) * 100);
			sprintf(frameNameBuffer, "/depth/Frame_%06d.png", frameIdx);
			//sprintf(frameNameBuffer, "/depth_erosion/Frame_%06d.png", frameIdx);
			//sprintf(frameNameBuffer, "/depth/Frame_%06d.png", frameIdx);
			//sprintf(frameNameBuffer, "/renderedDepth/Frame_%06d.png", frameIdx);
			//sprintf(frameNameBuffer, "/depth/Frame_%06d.png", frameIdx);
			cv::Mat depthImage = cv::imread(streamPath + frameNameBuffer, CV_LOAD_IMAGE_ANYDEPTH);
			depthImages.push_back(depthImage);
			sprintf(frameNameBuffer, "/color/Frame_%06d.png", frameIdx);
			//sprintf(frameNameBuffer, "/filteredColor/Frame_%06d.png", frameIdx);
			//sprintf(frameNameBuffer, "/color/Frame_%03d.png", frameIdx);
			cv::Mat colorImage = cv::imread(streamPath + frameNameBuffer);
			if (colorImage.channels() == 4)
				cv::cvtColor(colorImage, colorImage, CV_BGRA2BGR);
			colorImages.push_back(colorImage);
			sprintf(frameNameBuffer, "/mask/Frame_%06d.png", frameIdx);
			cv::Mat maskImage = cv::imread(streamPath + frameNameBuffer);
			cv::cvtColor(maskImage, maskImage, CV_BGR2GRAY);
			maskImages.push_back(maskImage);
		}
		if (img_dummy_file_w.is_open()) {
			int t_DepthImageWidth = depthImages[0].cols;
			int t_DepthImageHeight = depthImages[0].rows;
			int t_ColorImageWidth = colorImages[0].cols;
			int t_ColorImageHeight = colorImages[0].rows;
			int t_nImage = endIdx - startIdx;

			img_dummy_file_w.write((const char*)&t_DepthImageWidth, sizeof(int));
			img_dummy_file_w.write((const char*)&t_DepthImageHeight, sizeof(int));
			img_dummy_file_w.write((const char*)&t_ColorImageWidth, sizeof(int));
			img_dummy_file_w.write((const char*)&t_ColorImageHeight, sizeof(int));
			for (int i = 0; i < t_nImage; i++)
			{
				img_dummy_file_w.write((const char*)colorImages[i].data, sizeof(uchar) * 3 * t_ColorImageWidth * t_ColorImageHeight);
				img_dummy_file_w.write((const char*)depthImages[i].data, sizeof(ushort) * t_DepthImageWidth * t_DepthImageHeight);
			}
			img_dummy_file_w.close();
		}
		std::cout << "Done..." << std::endl;
	}
		
	imgNum = endIdx - startIdx;

	for (MyMesh::VertexIter v_it = template_mesh.vertices_begin(); v_it != template_mesh.vertices_end(); ++v_it) {
		_Vertices[v_it->idx()]._Pos.x = template_mesh.point(*v_it)[0];
		_Vertices[v_it->idx()]._Pos.y = template_mesh.point(*v_it)[1];
		_Vertices[v_it->idx()]._Pos.z = template_mesh.point(*v_it)[2];

		for (MyMesh::VertexFaceIter vf_it = template_mesh.vf_iter(*v_it); vf_it.is_valid(); ++vf_it) {
			_Vertices[v_it->idx()]._Triangles.push_back(vf_it->idx());
		}
	}
	for (MyMesh::FaceIter f_it = template_mesh.faces_begin(); f_it != template_mesh.faces_end(); ++f_it) {
		int vi = 0;
		for (MyMesh::FaceVertexIter fv_it = template_mesh.fv_iter(*f_it); fv_it.is_valid(); ++fv_it) {
			if (vi > 2) {
				printf("too many vertex");
				continue;
			}
			_Triangles[f_it->idx()]._Vertices[vi] = fv_it->idx();
			vi++;
		}

		_Triangles[f_it->idx()]._Normal.x = template_mesh.normal(*f_it)[0];
		_Triangles[f_it->idx()]._Normal.y = template_mesh.normal(*f_it)[1];
		_Triangles[f_it->idx()]._Normal.z = template_mesh.normal(*f_it)[2];

	}

	opt -= OpenMesh::IO::Options::VertexTexCoord;
	vector<vector<float3>> positions_vec;
	vector<vector<float3>> normals_vec;
	vector<vector<float2>> pixel_vec;
	vector<vector<float>> areas_vec;
	layer_mesh_vec.resize(imgNum);
	positions_vec.resize(imgNum);
	pixel_vec.resize(imgNum);
	normals_vec.resize(imgNum);
	areas_vec.resize(imgNum);

	std::cout << "mesh read" << std::endl;
	for (int i = 0; i < imgNum; i++) {
		printProgBar(((float)i / (imgNum - 1.0)) * 100);
		layer_mesh_vec[i].request_face_normals();
		layer_mesh_vec[i].request_vertex_normals();

		string filename;
		filename = NR_MeshPathAndPrefix + "_" + zeroPadding(to_string(i), 3) + "." + mesh_extension;
		if (!OpenMesh::IO::read_mesh(layer_mesh_vec[i], filename, opt))
		{
			std::cerr << "read error\n";
			exit(1);
		}
		layer_mesh_vec[i].update_normals();
		positions_vec[i].resize(numVer);
		pixel_vec[i].resize(numVer);
		normals_vec[i].resize(numTri);
		areas_vec[i].resize(numTri);

		cv::Mat _P_inv_IR = (_Pose.inv());
		cv::Mat _P_inv_CO = (_CO_IR * _P_inv_IR);

		for (MyMesh::VertexIter v_it = layer_mesh_vec[i].vertices_begin(); v_it != layer_mesh_vec[i].vertices_end(); ++v_it) {
			positions_vec[i][v_it->idx()].x = layer_mesh_vec[i].point(*v_it)[0];
			positions_vec[i][v_it->idx()].y = layer_mesh_vec[i].point(*v_it)[1];
			positions_vec[i][v_it->idx()].z = layer_mesh_vec[i].point(*v_it)[2];

			//////////////////pixel cal
			float3 _Point_CO;
			float3 _Point_IR;
			float2 _Pixel_CO;
			float2 _Pixel_IR;

			Transform(positions_vec[i][v_it->idx()], _P_inv_CO, _Point_CO);
			PointToPixel_CO(_Point_CO, _Pixel_CO);

			pixel_vec[i][v_it->idx()] = _Pixel_CO;
		}

		for (MyMesh::FaceIter f_it = layer_mesh_vec[i].faces_begin(); f_it != layer_mesh_vec[i].faces_end(); ++f_it) {
			normals_vec[i][f_it->idx()].x = layer_mesh_vec[i].normal(*f_it)[0];
			normals_vec[i][f_it->idx()].y = layer_mesh_vec[i].normal(*f_it)[1];
			normals_vec[i][f_it->idx()].z = layer_mesh_vec[i].normal(*f_it)[2];
		}
		for (MyMesh::FaceIter f_it = layer_mesh_vec[i].faces_begin(); f_it != layer_mesh_vec[i].faces_end(); ++f_it) {
			MyMesh::FaceVertexIter fv_it = template_mesh.fv_iter(*f_it);

			Vec3f v0;
			Vec3f v1;
			Vec3f v2;
			v0 = Vec3f(pixel_vec[i][fv_it->idx()].x, pixel_vec[i][fv_it->idx()].y, 1.0);
			fv_it++;
			v1 = Vec3f(pixel_vec[i][fv_it->idx()].x, pixel_vec[i][fv_it->idx()].y, 1.0);
			fv_it++;
			v2 = Vec3f(pixel_vec[i][fv_it->idx()].x, pixel_vec[i][fv_it->idx()].y, 1.0);

			Vec3d e1 = OpenMesh::vector_cast<Vec3d, Vec3f>(v1 - v0);
			Vec3d e2 = OpenMesh::vector_cast<Vec3d, Vec3f>(v2 - v0);

			Vec3d fN = OpenMesh::cross(e1, e2);
			double area = fN.norm() / 2.0;

			areas_vec[i][f_it->idx()] = area;
		}
	}
	

	vector<size_t> accumVisTri(numTri, 0);
	vector<map<size_t, float>> visTri(imgNum);

	//////////////// visibility check
	std::cout << "\nvisibility check" << std::endl;
	for (int k = 0; k < imgNum; k++)
	{
		printProgBar(((float)k / (imgNum - 1.0)) * 100);
		cv::Mat _P_inv_IR = (_Pose.inv());
		cv::Mat _P_inv_CO = (_CO_IR * _P_inv_IR);
		for (MyMesh::FaceIter f_it = template_mesh.faces_begin(); f_it != template_mesh.faces_end(); ++f_it) {
			bool _valid = true;
			float _weight = 0;
			for (MyMesh::FaceVertexIter fv_it = template_mesh.fv_iter(*f_it); fv_it.is_valid(); ++fv_it) {
				float3 _Point_CO;
				float3 _Point_IR;
				float2 _Pixel_CO;
				float2 _Pixel_IR;
				Transform(positions_vec[k][fv_it->idx()], _P_inv_IR, _Point_IR);
				Transform(positions_vec[k][fv_it->idx()], _P_inv_CO, _Point_CO);
				PointToPixel_CO(_Point_CO, _Pixel_CO);
				PointToPixel_IR(_Point_IR, _Pixel_IR);
				if ((_Pixel_IR.x) < 20.0f || (_Pixel_IR.x) > IRIMX - 40 || (_Pixel_IR.y) < 20.0f || (_Pixel_IR.y) > IRIMY - 20.0f)
				{
					_valid = false;
					break;
				}
				else
				{
					float _Depth = (float)depthImages[k].at<unsigned short>((int)_Pixel_IR.y, (int)_Pixel_IR.x) / 1000.0f;
					if (abs(_Depth - _Point_IR.z) < DTEST) {
						float tmp_Weight = 1.0;
						float weight_max = 0.0;
						float3 avgnormal = make_float3(0.0);
						for (auto fvf_idx : _Vertices[fv_it->idx()]._Triangles) {
							avgnormal += normals_vec[k][fvf_idx];
							/*float _Dot = _Pose.at<float>(0, 2) * normals_vec[k][fvf_idx].x + _Pose.at<float>(1, 2) * normals_vec[k][fvf_idx].y + _Pose.at<float>(2, 2) * normals_vec[k][fvf_idx].z;
							if (normal_direction * _Dot < 0)
								_valid = false;*/
							//weight_max = std::max(normal_direction * _Dot, weight_max);
						}
						avgnormal /= _Vertices[fv_it->idx()]._Triangles.size();
						float _Dot = _Pose.at<float>(0, 2) * avgnormal.x + _Pose.at<float>(1, 2) *avgnormal.y + _Pose.at<float>(2, 2) * avgnormal.z;
						if (normal_direction * _Dot < 0.3)
							_valid = false;
						weight_max = normal_direction * _Dot;
						tmp_Weight *= weight_max;
						if (_weight < tmp_Weight)
							_weight = tmp_Weight;
					}
					else {
						_valid = false;
						break;
					}
				}
			}
			if (_valid) {
				accumVisTri[f_it->idx()]++;
				visTri[k].insert(make_pair(f_it->idx(), _weight));
			}
		}
	}
	std::cout << "Done..." << std::endl;

}

void Mapper4D::ExtractKeyframes() {
	std::cout << std::endl << "Start extracting keyframes" << std::endl;

	// uniform sampling
	keyFrame_idx.clear();
	for (int i = 0; i < imgNum; i++) {
		if(i % 4 == 0)
			keyFrame_idx.push_back(i);
	}

	std::cout << keyFrame_idx.size() << " frames extracted in spatial sampling" << std::endl;


	layer_Vertices_key.resize(LAYERNUM);
	layer_Triangles_key.resize(LAYERNUM);
	for (int l = 0; l < LAYERNUM; l++) {
		layer_Vertices_key[l].resize(layer_Vertices[l].size());
		for (int vi = 0; vi < layer_Vertices[l].size(); vi++) {
			layer_Vertices_key[l][vi]._Pos = layer_Vertices[l][vi]._Pos;
			layer_Vertices_key[l][vi]._Col = layer_Vertices[l][vi]._Col;
			layer_Vertices_key[l][vi]._Triangles.assign(layer_Vertices[l][vi]._Triangles.begin(), layer_Vertices[l][vi]._Triangles.end());
			for (auto img_idx : keyFrame_idx) {
				layer_Vertices_key[l][vi]._Pos_time.push_back(layer_Vertices[l][vi]._Pos_time[img_idx]);
				layer_Vertices_key[l][vi]._Norm_time.push_back(layer_Vertices[l][vi]._Norm_time[img_idx]);
			}

			for (int i = 0; i < layer_Vertices[l][vi]._Img.size(); i++) {
				auto iter = find(keyFrame_idx.begin(), keyFrame_idx.end(), layer_Vertices[l][vi]._Img[i]);
				if (iter == keyFrame_idx.end())
					continue;
				layer_Vertices_key[l][vi]._Img_Tex.push_back(layer_Vertices[l][vi]._Img_Tex[i]);
				layer_Vertices_key[l][vi]._Img_Tex_ori.push_back(layer_Vertices[l][vi]._Img_Tex_ori[i]);
				layer_Vertices_key[l][vi]._Img.push_back(distance(keyFrame_idx.begin(), iter));
			}
		}
		layer_Triangles_key[l].resize(layer_Triangles[l].size());
		for (int ti = 0; ti < layer_Triangles[l].size(); ti++) {
			layer_Triangles_key[l][ti]._Vertices[0] = layer_Triangles[l][ti]._Vertices[0];
			layer_Triangles_key[l][ti]._Vertices[1] = layer_Triangles[l][ti]._Vertices[1];
			layer_Triangles_key[l][ti]._Vertices[2] = layer_Triangles[l][ti]._Vertices[2];

			layer_Triangles_key[l][ti]._Tex_BIGPIC[0] = layer_Triangles[l][ti]._Tex_BIGPIC[0];
			layer_Triangles_key[l][ti]._Tex_BIGPIC[1] = layer_Triangles[l][ti]._Tex_BIGPIC[1];
			layer_Triangles_key[l][ti]._Tex_BIGPIC[2] = layer_Triangles[l][ti]._Tex_BIGPIC[2];

			layer_Triangles_key[l][ti]._Normal = layer_Triangles[l][ti]._Normal;
			for (auto img_idx : keyFrame_idx) {
				layer_Triangles_key[l][ti]._Area.push_back(layer_Triangles[l][ti]._Area[img_idx]);
			}

			for (int i = 0; i < layer_Triangles[l][ti]._Img.size(); i++) {
				auto iter = find(keyFrame_idx.begin(), keyFrame_idx.end(), layer_Triangles[l][ti]._Img[i]);
				if (iter == keyFrame_idx.end())
					continue;
				layer_Triangles_key[l][ti]._Bound.push_back(true);
				layer_Triangles_key[l][ti]._Label.push_back(true);
				layer_Triangles_key[l][ti]._Img_Weight.push_back(layer_Triangles[l][ti]._Img_Weight[i]);
				layer_Triangles_key[l][ti]._Img.push_back(distance(keyFrame_idx.begin(), iter));
			}
			//renormalize
			float wsum = 0;
			for (auto im_weight : layer_Triangles_key[l][ti]._Img_Weight) wsum += im_weight;
			for (int wi = 0; wi < layer_Triangles_key[l][ti]._Img_Weight.size(); wi++)
				layer_Triangles_key[l][ti]._Img_Weight[wi] /= wsum;
		}
	}
}

void Mapper4D::Get_VT_layer(vector<hostVertex> &hVs, vector<hostTriangle> &hTs, int layer) {
	hVs.clear();
	hTs.clear();
	/*hVs.resize(layer_Vertices[layer].size());
	hTs.resize(layer_Triangles[layer].size());*/
	hVs.assign(layer_Vertices[layer].begin(), layer_Vertices[layer].end());
	hTs.assign(layer_Triangles[layer].begin(), layer_Triangles[layer].end());
}

void Mapper4D::Get_VT_layer_key(vector<hostVertex>& hVs, vector<hostTriangle>& hTs, int layer) {
	hVs.clear();
	hTs.clear();
	/*hVs.resize(layer_Vertices[layer].size());
	hTs.resize(layer_Triangles[layer].size());*/
	hVs.assign(layer_Vertices_key[layer].begin(), layer_Vertices_key[layer].end());
	hTs.assign(layer_Triangles_key[layer].begin(), layer_Triangles_key[layer].end());
}

void Mapper4D::Get_V_layer(vector<hostVertex>& hVs, int layer) {
	hVs.clear();
	hVs.assign(layer_Vertices[layer].begin(), layer_Vertices[layer].end());
}

void Mapper4D::Push_VT_layer(vector<hostVertex>& hVs, vector<hostTriangle>& hTs, int layer) {
	layer_Vertices[layer].clear();
	layer_Triangles[layer].clear();
	layer_Vertices[layer].assign(hVs.begin(), hVs.end());
	layer_Triangles[layer].assign(hTs.begin(), hTs.end());
}

void Mapper4D::ForceSetnow2ori() {
	for (int i = 0; i < numVer; i++) {
		int v_Img_Num = layer_Vertices[LAYERNUM - 1][i]._Img.size();
		for (int j = 0; j < v_Img_Num; j++) {
			layer_Vertices[LAYERNUM - 1][i]._Img_Tex_ori[j] = layer_Vertices[LAYERNUM - 1][i]._Img_Tex[j];
		}
	}
}

void Mapper4D::Push_VT_layer_key(vector<hostVertex>& hVs, vector<hostTriangle>& hTs, int layer) {
	layer_Vertices_key[layer].clear();
	layer_Triangles_key[layer].clear();
	layer_Vertices_key[layer].assign(hVs.begin(), hVs.end());
	layer_Triangles_key[layer].assign(hTs.begin(), hTs.end());
}

void Mapper4D::Push_V_layer(vector<hostVertex>& hVs, int layer) {
	layer_Vertices[layer].clear();
	layer_Vertices[layer].assign(hVs.begin(), hVs.end());
}

void Mapper4D::GetNumInfo(int *nV, int *nT, int *nI)
{
	*nV = numVer;
	*nT = numTri;
	*nI = imgNum;
}

void Mapper4D::GetNumInfo_layer(int *nV, int *nT, int layer)
{
	*nV = layer_numVer[layer];
	*nT = layer_numTri[layer];
}

bool Mapper4D::SaveResult(string FileName) {
	std::ofstream fout(FileName, std::ofstream::binary);

	if (fout.is_open()) {
		int t_ColorImageWidth = colorImages[0].cols;
		int t_ColorImageHeight = colorImages[0].rows;
		int t_nImage = imgNum;

		fout.write((const char*)&t_nImage, sizeof(int));
		fout.write((const char*)&t_ColorImageWidth, sizeof(int));
		fout.write((const char*)&t_ColorImageHeight, sizeof(int));
		for (int i = 0; i < t_nImage; i++)
			fout.write((const char*)colorImages[i].data, sizeof(uchar) * 3 * t_ColorImageWidth * t_ColorImageHeight);

		fout.write((char*)&numVer, sizeof(int));
		fout.write((char*)&numTri, sizeof(int));
		fout.write((char*)&imgNum, sizeof(int));
		for (auto v : layer_Vertices[LAYERNUM - 1])
			v.write(&fout);
		for (auto t : layer_Triangles[LAYERNUM - 1])
			t.write(&fout);
		fout.close();
		return true;
	}
	std::cout << "Can not open file: " + FileName << std::endl;
	return false;
}

bool Mapper4D::LoadResult(string FileName) {
	layer_Vertices.resize(LAYERNUM);
	layer_Triangles.resize(LAYERNUM);
	std::ifstream fin(FileName, std::ios::in | std::ofstream::binary);

	if (fin.is_open()) {
		std::cout << "read file: " << FileName << std::endl;

		int t_ColorImageWidth;
		int t_ColorImageHeight;
		int t_nImage;

		fin.read((char*)&t_nImage, sizeof(int));
		fin.read((char*)&t_ColorImageWidth, sizeof(int));
		fin.read((char*)&t_ColorImageHeight, sizeof(int));

		colorImages.resize(t_nImage);

		for (int i = 0; i < t_nImage; i++) {
			colorImages[i].create(t_ColorImageHeight, t_ColorImageWidth, CV_8UC3);
			fin.read((char*)colorImages[i].data, sizeof(uchar) * 3 * t_ColorImageWidth * t_ColorImageHeight);
		}

		fin.read((char*)&numVer, sizeof(int));
		fin.read((char*)&numTri, sizeof(int));
		fin.read((char*)&imgNum, sizeof(int));
		layer_Vertices[LAYERNUM - 1].resize(numVer);
		layer_Triangles[LAYERNUM - 1].resize(numTri);
		for (auto &v : layer_Vertices[LAYERNUM - 1])
			v.read(&fin);
		for (auto &t : layer_Triangles[LAYERNUM - 1])
			t.read(&fin);
		fin.close();


		/*for (auto t : layer_Triangles[LAYERNUM - 1]) {
			float wsum = 0;
			for (auto w : t._Img_Weight) {
				wsum += w;
			}
			std::cout << wsum << std::endl;
		}
		system("PAUSE");*/

		return true;
	}

std::cout << "Can not open file: " + FileName << std::endl;
return false;
}
