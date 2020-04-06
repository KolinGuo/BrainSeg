// Copyright 2014-2015 Isis Innovation Limited and the authors of gSLICr

#pragma once
#include "gSLICr_seg_engine.h"
#include <iostream>

using namespace std;
using namespace gSLICr;
using namespace gSLICr::objects;
using namespace gSLICr::engines;


seg_engine::seg_engine(const objects::settings& in_settings)
{
	gSLICr_settings = in_settings;
}


seg_engine::~seg_engine()
{
	if (max_dist_color != NULL) delete max_dist_color;
	if (source_img != NULL) delete source_img;
	if (cvt_img != NULL) delete cvt_img;
	if (idx_img != NULL) delete idx_img;
	if (spixel_map != NULL) delete spixel_map;
}

void seg_engine::Perform_Segmentation(const unsigned char* in_arr)
{
	std::cout << "##### Begin segmentating #####" << std::endl;
	//source_img->SetFrom(in_img, ORUtils::MemoryBlock<unsigned char>::CPU_TO_CUDA);
	
	// Fill the input image
	unsigned char* source_img_ptr = source_img->GetData(MEMORYDEVICE_UNIFIED);
	
	for (int y = 0; y < source_img->noDims.y; y++)
		for (int x = 0; x < source_img->noDims.x; x++)
		{
			unsigned int idx = x + y * (unsigned int)source_img->noDims.x;
			source_img_ptr[idx] = in_arr[idx];
		}

	Cvt_Img_Space(source_img, cvt_img, gSLICr_settings.color_space);

	Init_Cluster_Centers();	
	Find_Center_Association();
	
	for (int i = 0; i < gSLICr_settings.no_iters; i++)
	{
		std::cout << "### Begining " << i+1 << " / " << gSLICr_settings.no_iters << " iteration ###" << std::endl;
		Update_Cluster_Center();
		Find_Center_Association();
		std::cout << "### Finished " << i+1 << " / " << gSLICr_settings.no_iters << " iteration ###" << std::endl;
	}

	if(gSLICr_settings.do_enforce_connectivity) Enforce_Connectivity();
	ORcudaSafeCall(cudaDeviceSynchronize());
}



