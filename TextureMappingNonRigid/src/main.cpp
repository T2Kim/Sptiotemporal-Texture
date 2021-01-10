
#define NOMINMAX
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_functions.h>
#include <helper_cuda.h> 
#include <helper_math.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <fstream>
#include "Mapper4D.h"
#include "Optimizer.cuh"
#include "shader_utils.h"
#include "Renderer.h"
#include "Simplifier.h"
#include "Selector.h"

#include "any.hpp"
#include <Windows.h>

static void make_unit_dir(string unitTestDir) {
	CreateDirectory(unitTestDir.c_str(), NULL);
	CreateDirectory((unitTestDir + "/decimated_mesh").c_str(), NULL);
	CreateDirectory((unitTestDir + "/geo_render").c_str(), NULL);
	CreateDirectory((unitTestDir + "/ml_color_render").c_str(), NULL);
	CreateDirectory((unitTestDir + "/naive_color_render").c_str(), NULL);
	CreateDirectory((unitTestDir + "/projection").c_str(), NULL);
	CreateDirectory((unitTestDir + "/sl_color_render").c_str(), NULL);
	CreateDirectory((unitTestDir + "/sub_atlas").c_str(), NULL);
	CreateDirectory((unitTestDir + "/sub_atlas/mask").c_str(), NULL);
	CreateDirectory((unitTestDir + "/sub_atlas/texel").c_str(), NULL);
	for (int i = 0; i < LAYERNUM; i++) {
		CreateDirectory((unitTestDir + "/decimated_mesh/" + to_string(i)).c_str(), NULL);
		CreateDirectory((unitTestDir + "/projection/" + to_string(i)).c_str(), NULL);
	}
}

int main(int argc, char** argv) {

	bool subtexture = false;
	bool genglobalatlas = false;

	if (argc < 2) {
		TexMap::config.Setconfig("./conf.json", "case_main_test");
	}
	else {
		TexMap::config.Setconfig(argv[1], argv[2]);
	}
	string streamPath = data_root_path + "/stream";
	string mesh4DPathPrefix = data_root_path + "/mesh/Frame";
	string atlasPath = data_root_path + "/atlas/" + atlas_path;
	string atlasvideoPath = atlasPath + "/video/";
	string atlasvideoPathSplit = atlasvideoPath  + "/split/";
	string unitpath = data_root_path + "/unit_test/" + unit_test_path;
	string texobjFile = data_root_path + tex_mesh_path;
	intermediate_data_path = data_root_path + "/_intermediate_data/";
	CreateDirectory((data_root_path + "/atlas").c_str(), NULL);
	CreateDirectory((data_root_path + "/capture").c_str(), NULL);
	CreateDirectory((data_root_path + "/unit_test").c_str(), NULL);
	CreateDirectory(atlasPath.c_str(), NULL);
	CreateDirectory(atlasvideoPath.c_str(), NULL);
	CreateDirectory(atlasvideoPathSplit.c_str(), NULL);
	CreateDirectory(intermediate_data_path.c_str(), NULL);
	for (int i = start_idx; i < end_idx; i++) {
		CreateDirectory((atlasvideoPathSplit + zeroPadding(i, 6)).c_str(), NULL);
	}
	make_unit_dir(unitpath);

	string v_shader_filename = "./shaders/" + shader_model + ".v.glsl";
	string f_shader_filename = "./shaders/" + shader_model + ".f.glsl";

	TexMap::Renderer* renderer;
	if (!is_viewer) {
		StopWatchInterface* whole_time = NULL;
		sdkCreateTimer(&whole_time);
		sdkStartTimer(&whole_time);
		string template_mesh_name = mesh4DPathPrefix + "_" + zeroPadding(to_string(0), 3) + "." + mesh_extension;
		TexMap::Mapper4D* mapper4D;
		TexMap::Selector* selector;
		TexMap::Optimizer* optimizer;


		std::ifstream optVT_file(intermediate_data_path + "opt_VT.dat", std::ios::in | std::ios::binary);
		if (optVT_file.good()) {
			optVT_file.close();
			mapper4D = new TexMap::Mapper4D();
			mapper4D->LoadResult(intermediate_data_path + "opt_VT.dat");
			optimizer = new TexMap::Optimizer();
			optimizer->LoadModel4D(mapper4D, false);

			renderer = new TexMap::Renderer(optimizer);
			if (renderer->gl_init(&argc, argv) > 0)
				return -1;
			renderer->init_resources_UVAtlas(texobjFile);
			return 0;
		}
		else {
			// generate global atlas
			mapper4D = new TexMap::Mapper4D(template_mesh_name, mesh4DPathPrefix, streamPath, start_idx, end_idx);
			mapper4D->ConstructVertree();
			mapper4D->SaveResult(intermediate_data_path + "naive_VT.dat");
			mapper4D->ExtractKeyframes();

			optimizer = new TexMap::Optimizer();

			optimizer->SetMode("multi");

			mapper4D->SetValidFrame(true);

			clock_t start = clock(); // 시간 측정 시작

			optimizer->Model4DLoadandMultiUpdate_key(mapper4D);

			mapper4D->InterpKeytoAll();

			delete optimizer;
			optimizer = new TexMap::Optimizer("multi");
			mapper4D->SetValidFrame(false);
			optimizer->Model4DLoadandMultiRefine_all(mapper4D);

			clock_t end = clock(); // 시간 측정 끝
			double result = (double)(end - start);
			printf("%f", result); //결과 출력

			// 여기여기 now ori is optimized coordinate not subtracted.
			mapper4D->ForceSetnow2ori();
			mapper4D->BackgroundSubtraction(false);
			optimizer->LoadModel4D(mapper4D, false);

			mapper4D->SaveResult(intermediate_data_path + "opt_VT.dat");

			renderer = new TexMap::Renderer(optimizer);
			if (renderer->gl_init(&argc, argv) > 0)
				return -1;
			renderer->init_resources_UVAtlas(texobjFile);
			return 0;
		}
		
		mapper4D->RecomputeWeight_normal();

		selector = new TexMap::Selector();
		optimizer->SetSelector(selector);
		selector->Mapper4DandLabeling_new(mapper4D);

		renderer = new TexMap::Renderer(optimizer);
		if (renderer->gl_init(&argc, argv) > 0)
			return -1;
		renderer->init_resources_UVAtlas_video(v_shader_filename.c_str(), f_shader_filename.c_str(), texobjFile, start_idx, end_idx);

	}
	else {
		renderer = new TexMap::Renderer();
		if (renderer->gl_init(&argc, argv) > 0)
			return -1;
		renderer->init_resources_video(mesh4DPathPrefix, texobjFile, atlasPath, v_shader_filename.c_str(), f_shader_filename.c_str(), start_idx, end_idx);
		renderer->mainloop();
		renderer->free_resources();
	}
	return 0;
}