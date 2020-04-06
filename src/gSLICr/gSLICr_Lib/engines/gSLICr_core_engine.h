// Copyright 2014-2015 Isis Innovation Limited and the authors of gSLICr

#pragma once
#include "gSLICr_seg_engine_GPU.h"


namespace gSLICr
{
	namespace engines
	{
		class core_engine
		{
		private:

			seg_engine* slic_seg_engine;

		public:

			core_engine(const objects::settings& in_settings);
			~core_engine();

			// Function to segment in_img
			//void Process_Frame(UCharImage* in_img);
			void Process_Frame(const unsigned char* in_arr);

			// Function to get the pointer to the segmented mask image
			const IntImage * Get_Seg_Res();
			const UCharImage * Get_Orig_Img();

			// Function to draw segmentation result on out_img
			void Draw_Segmentation_Result(UChar4Image* out_img);

			// Write the segmentation result to a PGM image
			void Write_Seg_Res_To_PGM(const char* fileName);
		};
	}
}

