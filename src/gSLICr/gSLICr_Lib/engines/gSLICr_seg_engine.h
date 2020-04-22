// Copyright 2014-2015 Isis Innovation Limited and the authors of gSLICr

#pragma once
#include "../gSLICr_defines.h"
#include "../objects/gSLICr_settings.h"
#include "../objects/gSLICr_spixel_info.h"

namespace gSLICr
{
	namespace engines
	{
		class seg_engine
		{
		protected:

			// normalizing distances
			float max_color_dist;
			float max_xy_dist;

			// For Slic_Zero, max_dist_color for each segment
			bool slic_zero;
			MaxDistColorMap *max_dist_color;

			// images
			UCharImage *source_img;
			//Float3Image *cvt_img;
			FloatImage *cvt_img;
			IntImage *idx_img;

			// superpixel map
			SpixelMap* spixel_map;
			unsigned int spixel_size;

			objects::settings gSLICr_settings;

			virtual void Cvt_Img_Space(UCharImage* inimg, FloatImage* outimg, COLOR_SPACE color_space) = 0;
			virtual void Init_Cluster_Centers() = 0;
			virtual void Find_Center_Association() = 0;
			virtual void Update_Cluster_Center() = 0;
			virtual void Enforce_Connectivity() = 0;

		public:

			seg_engine(const objects::settings& in_settings);
			virtual ~seg_engine();

			const IntImage* Get_Seg_Mask() const {
				//idx_img->UpdateHostFromDevice();
				return idx_img;
			};

			const UCharImage* Get_Orig_Img() const {
				//idx_img->UpdateHostFromDevice();
				return source_img;
			};

			//void Perform_Segmentation(UCharImage* in_img);
			void Perform_Segmentation(const unsigned char* in_arr);
			virtual void Draw_Segmentation_Result(UChar4Image* out_img){};
		};
	}
}

