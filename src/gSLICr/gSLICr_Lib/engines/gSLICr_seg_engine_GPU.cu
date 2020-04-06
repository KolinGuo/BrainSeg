// Copyright 2014-2015 Isis Innovation Limited and the authors of gSLICr

#ifndef COMPILE_WITHOUT_CUDA
#include "../../ORUtils/CUDADefines.h"
#endif
#include "gSLICr_seg_engine_GPU.h"
#include "gSLICr_seg_engine_shared.h"
#include <iostream>

using namespace std;
using namespace gSLICr;
using namespace gSLICr::objects;
using namespace gSLICr::engines;

// ----------------------------------------------------
//
//	kernel function defines
//
// ----------------------------------------------------

__global__ void Cvt_Img_Space_device(const unsigned char* inimg, float* outimg, Vector2i img_size, COLOR_SPACE color_space);

__global__ void Enforce_Connectivity_device(const int* in_idx_img, int* out_idx_img, Vector2i img_size);

__global__ void Init_Cluster_Centers_device(const float* inimg, spixel_info* out_spixel, Vector2i map_size, Vector2i img_size, unsigned int spixel_size);

__global__ void Find_Center_Association_device(const float* inimg, const spixel_info* in_spixel_map, const float* in_max_dist_color, int* out_idx_img, Vector2i map_size, Vector2i img_size, unsigned int spixel_size, float weight, float max_xy_dist, float max_color_dist, bool slic_zero);

__global__ void Update_Cluster_Center_device(const float* inimg, const int* in_idx_img, spixel_info* accum_map, Vector2i map_size, Vector2i img_size, unsigned int spixel_size, int no_blocks_per_line);

__global__ void Finalize_Reduction_Result_device(const spixel_info* accum_map, spixel_info* spixel_list, Vector2i map_size, int no_blocks_per_spixel);

__global__ void Update_Color_Distance_device(const float* inimg, const spixel_info* in_spixel_map, const int* in_idx_img, float* out_max_dist_color, Vector2i map_size, Vector2i img_size, unsigned int spixel_size, float weight, float max_xy_dist, float max_color_dist);

__global__ void Draw_Segmentation_Result_device(const int* idx_img, unsigned char* sourceimg, Vector4u* outimg, Vector2i img_size);

// ----------------------------------------------------
//
//	host function implementations
//
// ----------------------------------------------------

seg_engine_GPU::seg_engine_GPU(const settings& in_settings) : seg_engine(in_settings)
{
	source_img = new UCharImage(in_settings.img_size);
	std::cout << "Created source_img" << std::endl;
	cvt_img = new FloatImage(in_settings.img_size);
	std::cout << "Created cvt_img" << std::endl;
	idx_img = new IntImage(in_settings.img_size);
	std::cout << "Created idx_img" << std::endl;
	tmp_idx_img = new IntImage(in_settings.img_size);
	std::cout << "Created tmp_idx_img" << std::endl;

	// Calculate super pixel size
	if (in_settings.seg_method == GIVEN_NUM)
	{
		double cluster_size = (double)((unsigned int)in_settings.img_size.x * (unsigned int)in_settings.img_size.y) / (double)in_settings.no_segs;
		spixel_size = (unsigned int)ceil(sqrt(cluster_size));
	}
	else
	{
		spixel_size = in_settings.spixel_size;
	}
	
	// Calculate super pixel number per col and row
	unsigned int spixel_per_col = (unsigned int)ceil(in_settings.img_size.x / spixel_size);
	unsigned int spixel_per_row = (unsigned int)ceil(in_settings.img_size.y / spixel_size);	
	std::cout << "spixel_size: " << spixel_size << std::endl;

	// Create super pixel map
	Vector2i map_size = Vector2i(spixel_per_col, spixel_per_row);
	spixel_map = new SpixelMap(map_size);

	// Create max_dist_color for each segment and initialize to 1.0
	slic_zero = in_settings.slic_zero;
	max_dist_color = new MaxDistColorMap(map_size);
	float* max_dist_color_ptr = max_dist_color->GetData(MEMORYDEVICE_UNIFIED);
	
	for (unsigned int i = 0; i < spixel_per_col*spixel_per_row; i++)
		max_dist_color_ptr[i] = 1.0;
	//max_dist_color->UpdateDeviceFromHost();

	float total_pixel_to_search = (float)(spixel_size * spixel_size * 9);
	no_grid_per_center = (int)ceil(total_pixel_to_search / (float)(BLOCK_DIM * BLOCK_DIM));

	map_size.x *= no_grid_per_center;
	accum_map = new ORUtils::Image<spixel_info>(map_size);

	// normalizing factors
	max_xy_dist = 1.0f / (1.4142f * spixel_size); // sqrt(2) * spixel_size
	switch (in_settings.color_space)
	{
	case RGB:
		max_color_dist = 5.0f / (1.7321f * 255);
		break;
	case XYZ:
		max_color_dist = 5.0f / 1.7321f; 
		break; 
	case CIELAB:
		max_color_dist = 15.0f / (1.7321f * 128);
		break;
	case GRAY:
		max_color_dist = 5.0f / 255;
		break;
	}

	max_color_dist *= max_color_dist;
	max_xy_dist *= max_xy_dist;
}

gSLICr::engines::seg_engine_GPU::~seg_engine_GPU()
{
	if (accum_map != NULL) delete accum_map;
	if (tmp_idx_img != NULL) delete tmp_idx_img;
}


void gSLICr::engines::seg_engine_GPU::Cvt_Img_Space(UCharImage* inimg, FloatImage* outimg, COLOR_SPACE color_space)
{
	unsigned char* inimg_ptr = inimg->GetData(MEMORYDEVICE_UNIFIED);
	float* outimg_ptr = outimg->GetData(MEMORYDEVICE_UNIFIED);
	Vector2i img_size = inimg->noDims;

	dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
	dim3 gridSize((int)ceil((float)img_size.x / (float)blockSize.x), (int)ceil((float)img_size.y / (float)blockSize.y));

	Cvt_Img_Space_device<<<gridSize, blockSize>>>(inimg_ptr, outimg_ptr, img_size, color_space);
	ORcudaSafeCall(cudaDeviceSynchronize());
}

void gSLICr::engines::seg_engine_GPU::Init_Cluster_Centers()
{
	spixel_info* spixel_list = spixel_map->GetData(MEMORYDEVICE_UNIFIED);
	float* img_ptr = cvt_img->GetData(MEMORYDEVICE_UNIFIED);
	
	Vector2i map_size = spixel_map->noDims;
	Vector2i img_size = cvt_img->noDims;

	dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
	dim3 gridSize((int)ceil((float)map_size.x / (float)blockSize.x), (int)ceil((float)map_size.y / (float)blockSize.y));

	Init_Cluster_Centers_device<<<gridSize, blockSize>>>(img_ptr, spixel_list, map_size, img_size, spixel_size);
	ORcudaSafeCall(cudaDeviceSynchronize());
}

void gSLICr::engines::seg_engine_GPU::Find_Center_Association()
{
	spixel_info* spixel_list = spixel_map->GetData(MEMORYDEVICE_UNIFIED);
	float* img_ptr = cvt_img->GetData(MEMORYDEVICE_UNIFIED);
	float* in_max_dist_color = max_dist_color->GetData(MEMORYDEVICE_UNIFIED);
	int* idx_ptr = idx_img->GetData(MEMORYDEVICE_UNIFIED);

	Vector2i map_size = spixel_map->noDims;
	Vector2i img_size = cvt_img->noDims;

	dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
	dim3 gridSize((int)ceil((float)img_size.x / (float)blockSize.x), (int)ceil((float)img_size.y / (float)blockSize.y));

	Find_Center_Association_device<<<gridSize, blockSize>>>(img_ptr, spixel_list, in_max_dist_color, idx_ptr, map_size, img_size, spixel_size, gSLICr_settings.coh_weight,max_xy_dist,max_color_dist,slic_zero);
	ORcudaSafeCall(cudaDeviceSynchronize());
}

void gSLICr::engines::seg_engine_GPU::Update_Cluster_Center()
{
	spixel_info* accum_map_ptr = accum_map->GetData(MEMORYDEVICE_UNIFIED);
	spixel_info* spixel_list_ptr = spixel_map->GetData(MEMORYDEVICE_UNIFIED);
	float* img_ptr = cvt_img->GetData(MEMORYDEVICE_UNIFIED);
	float* out_max_dist_color = max_dist_color->GetData(MEMORYDEVICE_UNIFIED);
	int* idx_ptr = idx_img->GetData(MEMORYDEVICE_UNIFIED);

	Vector2i map_size = spixel_map->noDims;
	Vector2i img_size = cvt_img->noDims;

	int no_blocks_per_line = spixel_size * 3 / BLOCK_DIM;

	dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
	dim3 gridSize(map_size.x, map_size.y, no_grid_per_center);

	Update_Cluster_Center_device<<<gridSize,blockSize>>>(img_ptr, idx_ptr, accum_map_ptr, map_size, img_size, spixel_size, no_blocks_per_line);
	ORcudaSafeCall(cudaDeviceSynchronize());

	dim3 gridSize2(map_size.x, map_size.y);

	Finalize_Reduction_Result_device<<<gridSize2,blockSize>>>(accum_map_ptr, spixel_list_ptr, map_size, no_grid_per_center);
	ORcudaSafeCall(cudaDeviceSynchronize());

	if (slic_zero)
	{
		dim3 gridSize3((int)ceil((float)img_size.x / (float)blockSize.x), (int)ceil((float)img_size.y / (float)blockSize.y));
		Update_Color_Distance_device<<<gridSize3, blockSize>>>(img_ptr, spixel_list_ptr, idx_ptr, out_max_dist_color, map_size, img_size, spixel_size, gSLICr_settings.coh_weight,max_xy_dist,max_color_dist);
		ORcudaSafeCall(cudaDeviceSynchronize());
	}
}

void gSLICr::engines::seg_engine_GPU::Enforce_Connectivity()
{
	int* idx_ptr = idx_img->GetData(MEMORYDEVICE_UNIFIED);
	int* tmp_idx_ptr = tmp_idx_img->GetData(MEMORYDEVICE_UNIFIED);
	Vector2i img_size = idx_img->noDims;

	dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
	dim3 gridSize((int)ceil((float)img_size.x / (float)blockSize.x), (int)ceil((float)img_size.y / (float)blockSize.y));

	Enforce_Connectivity_device<<<gridSize, blockSize>>>(idx_ptr, tmp_idx_ptr, img_size);
	ORcudaSafeCall(cudaDeviceSynchronize());
	Enforce_Connectivity_device<<<gridSize, blockSize>>>(tmp_idx_ptr, idx_ptr, img_size);
	ORcudaSafeCall(cudaDeviceSynchronize());
}

void gSLICr::engines::seg_engine_GPU::Draw_Segmentation_Result(UChar4Image* out_img)
{
	unsigned char* inimg_ptr = source_img->GetData(MEMORYDEVICE_UNIFIED);
	Vector4u* outimg_ptr = out_img->GetData(MEMORYDEVICE_UNIFIED);
	int* idx_img_ptr = idx_img->GetData(MEMORYDEVICE_UNIFIED);
	
	Vector2i img_size = idx_img->noDims;

	dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
	dim3 gridSize((int)ceil((float)img_size.x / (float)blockSize.x), (int)ceil((float)img_size.y / (float)blockSize.y));

	Draw_Segmentation_Result_device<<<gridSize,blockSize>>>(idx_img_ptr, inimg_ptr, outimg_ptr, img_size);
	//out_img->UpdateHostFromDevice();
	ORcudaSafeCall(cudaDeviceSynchronize());
}



// ----------------------------------------------------
//
//	device function implementations
//
// ----------------------------------------------------

__global__ void Cvt_Img_Space_device(const unsigned char* inimg, float* outimg, Vector2i img_size, COLOR_SPACE color_space)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x > img_size.x - 1 || y > img_size.y - 1) return;

	cvt_img_space_shared(inimg, outimg, img_size, x, y, color_space);

}

__global__ void Draw_Segmentation_Result_device(const int* idx_img, unsigned char* sourceimg, Vector4u* outimg, Vector2i img_size)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x == 0 || y == 0 || x > img_size.x - 2 || y > img_size.y - 2) return;

	draw_superpixel_boundry_shared(idx_img, sourceimg, outimg, img_size, x, y);
}

__global__ void Init_Cluster_Centers_device(const float* inimg, spixel_info* out_spixel, Vector2i map_size, Vector2i img_size, unsigned int spixel_size)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x > map_size.x - 1 || y > map_size.y - 1) return;

	init_cluster_centers_shared(inimg, out_spixel, map_size, img_size, spixel_size, x, y);
}

__global__ void Find_Center_Association_device(const float* inimg, const spixel_info* in_spixel_map, const float* in_max_dist_color, int* out_idx_img, Vector2i map_size, Vector2i img_size, unsigned int spixel_size, float weight, float max_xy_dist, float max_color_dist, bool slic_zero)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x > img_size.x - 1 || y > img_size.y - 1) return;

	find_center_association_shared(inimg, in_spixel_map, in_max_dist_color, out_idx_img, map_size, img_size, spixel_size, weight, x, y,max_xy_dist,max_color_dist,slic_zero);
}

__global__ void Update_Cluster_Center_device(const float* inimg, const int* in_idx_img, spixel_info* accum_map, Vector2i map_size, Vector2i img_size, unsigned int spixel_size, int no_blocks_per_line)
{
	int local_id = threadIdx.y * blockDim.x + threadIdx.x;

	__shared__ float color_shared[BLOCK_DIM*BLOCK_DIM];
	__shared__ Vector2f xy_shared[BLOCK_DIM*BLOCK_DIM];
	__shared__ int count_shared[BLOCK_DIM*BLOCK_DIM];
	__shared__ bool should_add; 

	color_shared[local_id] = 0.0;
	xy_shared[local_id] = Vector2f(0, 0);
	count_shared[local_id] = 0;
	should_add = false;
	__syncthreads();

	int no_blocks_per_spixel = gridDim.z;

	int spixel_id = blockIdx.y * map_size.x + blockIdx.x;

	// compute the relative position in the search window
	int block_x = blockIdx.z % no_blocks_per_line;
	int block_y = blockIdx.z / no_blocks_per_line;

	int x_offset = block_x * BLOCK_DIM + threadIdx.x;
	int y_offset = block_y * BLOCK_DIM + threadIdx.y;

	if (x_offset < spixel_size * 3 && y_offset < spixel_size * 3)
	{
		// compute the start of the search window
		int x_start = blockIdx.x * spixel_size - spixel_size;	
		int y_start = blockIdx.y * spixel_size - spixel_size;

		int x_img = x_start + x_offset;
		int y_img = y_start + y_offset;

		if (x_img >= 0 && x_img < img_size.x && y_img >= 0 && y_img < img_size.y)
		{
			unsigned int img_idx = y_img * (unsigned int)img_size.x + x_img;
			if (in_idx_img[img_idx] == spixel_id)
			{
				color_shared[local_id] = inimg[img_idx];
				xy_shared[local_id] = Vector2f(x_img, y_img);
				count_shared[local_id] = 1;
				should_add = true;
			}
		}
	}
	__syncthreads();

	if (should_add)
	{
		if (local_id < 128)
		{
			color_shared[local_id] += color_shared[local_id + 128];
			xy_shared[local_id] += xy_shared[local_id + 128];
			count_shared[local_id] += count_shared[local_id + 128];
		}
		__syncthreads();

		if (local_id < 64)
		{
			color_shared[local_id] += color_shared[local_id + 64];
			xy_shared[local_id] += xy_shared[local_id + 64];
			count_shared[local_id] += count_shared[local_id + 64];
		}
		__syncthreads();

		if (local_id < 32)
		{
			color_shared[local_id] += color_shared[local_id + 32];
			color_shared[local_id] += color_shared[local_id + 16];
			color_shared[local_id] += color_shared[local_id + 8];
			color_shared[local_id] += color_shared[local_id + 4];
			color_shared[local_id] += color_shared[local_id + 2];
			color_shared[local_id] += color_shared[local_id + 1];

			xy_shared[local_id] += xy_shared[local_id + 32];
			xy_shared[local_id] += xy_shared[local_id + 16];
			xy_shared[local_id] += xy_shared[local_id + 8];
			xy_shared[local_id] += xy_shared[local_id + 4];
			xy_shared[local_id] += xy_shared[local_id + 2];
			xy_shared[local_id] += xy_shared[local_id + 1];

			count_shared[local_id] += count_shared[local_id + 32];
			count_shared[local_id] += count_shared[local_id + 16];
			count_shared[local_id] += count_shared[local_id + 8];
			count_shared[local_id] += count_shared[local_id + 4];
			count_shared[local_id] += count_shared[local_id + 2];
			count_shared[local_id] += count_shared[local_id + 1];
		}
	}
	__syncthreads();

	if (local_id == 0)
	{
		unsigned int accum_map_idx = spixel_id * no_blocks_per_spixel + blockIdx.z;
		accum_map[accum_map_idx].center = xy_shared[0];
		accum_map[accum_map_idx].color_info = color_shared[0];
		accum_map[accum_map_idx].no_pixels = count_shared[0];
	}
}

__global__ void Update_Color_Distance_device(const float* inimg, const spixel_info* in_spixel_map, const int* in_idx_img, float* out_max_dist_color, Vector2i map_size, Vector2i img_size, unsigned int spixel_size, float weight, float max_xy_dist, float max_color_dist)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x > img_size.x - 1 || y > img_size.y - 1) return;

	//__syncthreads();
	update_color_distance_shared(inimg, in_spixel_map, in_idx_img, out_max_dist_color, map_size, img_size, spixel_size, weight, x, y,max_xy_dist,max_color_dist);
}

__global__ void Finalize_Reduction_Result_device(const spixel_info* accum_map, spixel_info* spixel_list, Vector2i map_size, int no_blocks_per_spixel)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x > map_size.x - 1 || y > map_size.y - 1) return;

	finalize_reduction_result_shared(accum_map, spixel_list, map_size, no_blocks_per_spixel, x, y);
}

__global__ void Enforce_Connectivity_device(const int* in_idx_img, int* out_idx_img, Vector2i img_size)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x > img_size.x - 1 || y > img_size.y - 1) return;

	supress_local_lable(in_idx_img, out_idx_img, img_size, x, y);
}

