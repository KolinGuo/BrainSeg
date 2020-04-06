// Copyright 2014-2015 Isis Innovation Limited and the authors of gSLICr

#pragma once
#include <iostream>
#include <typeinfo>
#include "engines/gSLICr_core_engine.h"

namespace gSLICr
{
  class Run_SLIC
  {
  protected:
    objects::settings in_settings;
    engines::core_engine* gSLICr_engine;

  public:
    Run_SLIC() {}
    void* get_settings() {
      return &in_settings;
    }

    void print_settings() {
      std::cout << std::endl << std::endl
        << "##################################################" << std::endl
        << "               Settings for gSLICr               " << std::endl
        << "    Image size: (h, w) = (y, x) = (" << in_settings.img_size.y << ", " << in_settings.img_size.x << ")" << std::endl
        << "    Number of superpixels: " << in_settings.no_segs << std::endl
        << "    Number of iterations: " << in_settings.no_iters << std::endl
        << "    Compactness weight: " << in_settings.coh_weight << std::endl
        << "    Enforce connectivity: " << ((1==in_settings.do_enforce_connectivity) ? "True" : "False") << std::endl
        << "    SLIC_zero: " << ((1==in_settings.slic_zero) ? "True" : "False") << std::endl
        << "    Converting colorspace: "; 
      switch (in_settings.color_space)
      {
        case RGB:
          std::cout << "RGB" << std::endl; break;
        case XYZ:
          std::cout << "XYZ" << std::endl; break;
        case CIELAB:
          std::cout << "CIELAB" << std::endl; break;
        case GRAY:
          std::cout << "Grayscale" << std::endl; break;
      }
      std::cout << "    Segmentation method: " << ((0==in_settings.seg_method) ? "GIVEN_NUM" : "GIVEN_SIZE") << std::endl
        << "##################################################" << std::endl << std::endl;
    }

    void run(unsigned char* in_arr) {
      std::cout << "##### Begin running #####" << std::endl;

      // Instantiate a core_engine
      gSLICr_engine = new engines::core_engine(in_settings);

      std::cout << "Successfully instantiated core engine" << std::endl;

      // Perform segmentation
      gSLICr_engine->Process_Frame(in_arr);

      std::cout << "Successfully performed segmentation" << std::endl;
    }

    const int* get_mask() {
      // Get segmentation mask
      const IntImage* seg_mask = gSLICr_engine->Get_Seg_Res();

      const int* data_ptr = seg_mask->GetData(MEMORYDEVICE_UNIFIED);

      return data_ptr;
    }

    const unsigned char* get_orig_img() {
      const UCharImage* orig_img = gSLICr_engine->Get_Orig_Img();
      const unsigned char* orig_img_ptr = orig_img->GetData(MEMORYDEVICE_UNIFIED);
      return orig_img_ptr;
    }

    ~Run_SLIC() {
      std::cout << "##### Begin deleting #####" << std::endl;
      if (gSLICr_engine != NULL) delete gSLICr_engine;
      std::cout << "Successfully deleted" << std::endl;
    }
  };
}
