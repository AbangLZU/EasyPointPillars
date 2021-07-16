/******************************************************************************
 * Copyright 2020 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

/*
 * Copyright 2018-2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @author Kosuke Murakami
 * @date 2019/02/26
 */

/**
* @author Yan haixu
* Contact: just github.com/hova88
* @date 2021/04/30
*/



// headers in STL
#include <stdio.h>

// headers in local files
#include "common.h"
#include "preprocess.h"

__global__ void make_pillar_histo_kernel(
    const float* dev_points, float* dev_pillar_point_feature_in_coors,
    int* pillar_count_histo, const int num_points,
    const int max_points_per_pillar, const int grid_x_size,
    const int grid_y_size, const int grid_z_size, const float min_x_range,
    const float min_y_range, const float min_z_range, const float pillar_x_size,
    const float pillar_y_size, const float pillar_z_size,
    const int num_point_feature) {
  int th_i = blockIdx.x * blockDim.x +  threadIdx.x ;
  if (th_i >= num_points) {
    return;
  }
  int x_coor = floor((dev_points[th_i * num_point_feature + 0] - min_x_range) / pillar_x_size);
  int y_coor = floor((dev_points[th_i * num_point_feature + 1] - min_y_range) / pillar_y_size);
  int z_coor = floor((dev_points[th_i * num_point_feature + 2] - min_z_range) / pillar_z_size);

  if (x_coor >= 0 && x_coor < grid_x_size && y_coor >= 0 &&
      y_coor < grid_y_size && z_coor >= 0 && z_coor < grid_z_size) {
    int count =
        atomicAdd(&pillar_count_histo[y_coor * grid_x_size + x_coor], 1);
    if (count < max_points_per_pillar) {
      int ind =
          y_coor * grid_x_size * max_points_per_pillar * num_point_feature +
          x_coor * max_points_per_pillar * num_point_feature +
          count * num_point_feature;
 
      for (int i = 0; i < num_point_feature; ++i) {
        dev_pillar_point_feature_in_coors[ind + i] =
            dev_points[th_i * num_point_feature + i];
      }
    }
  }
}

__global__ void make_pillar_index_kernel(
    int* dev_pillar_count_histo, int* dev_counter, int* dev_pillar_count,
    int* dev_x_coors, int* dev_y_coors, float* dev_num_points_per_pillar,
    int* dev_sparse_pillar_map, const int max_pillars,
    const int max_points_per_pillar, const int grid_x_size,
    const int num_inds_for_scan) {
  int x = blockIdx.x;
  int y = threadIdx.x;
  int num_points_at_this_pillar = dev_pillar_count_histo[y * grid_x_size + x];
  if (num_points_at_this_pillar == 0) {
    return;
  }

  int count = atomicAdd(dev_counter, 1);
  if (count < max_pillars) {
    atomicAdd(dev_pillar_count, 1);
    if (num_points_at_this_pillar >= max_points_per_pillar) {
      dev_num_points_per_pillar[count] = max_points_per_pillar;
    } else {
      dev_num_points_per_pillar[count] = num_points_at_this_pillar;
    }
    dev_x_coors[count] = x;
    dev_y_coors[count] = y;
    dev_sparse_pillar_map[y * num_inds_for_scan + x] = 1;
  }
}

__global__ void make_pillar_feature_kernel(
    float* dev_pillar_point_feature_in_coors, float* dev_pillar_point_feature,
    float* dev_pillar_coors, int* dev_x_coors, int* dev_y_coors,
    float* dev_num_points_per_pillar, const int max_points,
    const int num_point_feature, const int grid_x_size) {
  int ith_pillar = blockIdx.x;
  int num_points_at_this_pillar = dev_num_points_per_pillar[ith_pillar];
  int ith_point = threadIdx.x;
  if (ith_point >= num_points_at_this_pillar) {
    return;
  }
  int x_ind = dev_x_coors[ith_pillar];
  int y_ind = dev_y_coors[ith_pillar];
  int pillar_ind = ith_pillar * max_points * num_point_feature +
                   ith_point * num_point_feature;
  int coors_ind = y_ind * grid_x_size * max_points * num_point_feature +
                  x_ind * max_points * num_point_feature +
                  ith_point * num_point_feature;
  #pragma unroll 
  for (int i = 0; i < num_point_feature; ++i) {
    dev_pillar_point_feature[pillar_ind + i] =
        dev_pillar_point_feature_in_coors[coors_ind + i];
  }

  float coor_x = static_cast<float>(x_ind);
  float coor_y = static_cast<float>(y_ind);
  dev_pillar_coors[ith_pillar * 4 + 0] = 0;  // batch idx
  dev_pillar_coors[ith_pillar * 4 + 1] = 0;  // z
  dev_pillar_coors[ith_pillar * 4 + 2] = coor_y;
  dev_pillar_coors[ith_pillar * 4 + 3] = coor_x;
}



__global__ void pillar_mean_kernel(
  float* dev_points_mean, 
  const int num_point_feature,
  const float* dev_pillar_point_feature, 
  const float* dev_num_points_per_pillar, 
  int max_pillars , 
  int max_points_per_pillar) {

    extern __shared__ float temp[];
    int ith_pillar = blockIdx.x; 
    int ith_point  = threadIdx.x;
    int axis = threadIdx.y;
  
    int reduce_size = max_points_per_pillar > 32 ? 64 : 32;
    temp[threadIdx.x * 3 + axis] =  dev_pillar_point_feature[ith_pillar * max_points_per_pillar * num_point_feature + ith_point * num_point_feature + axis];  
    if (threadIdx.x < reduce_size - max_points_per_pillar) {
        temp[(threadIdx.x + max_points_per_pillar) * 3 + axis] = 0.0f; //--> dummy placeholds will set as 0
    }
    __syncthreads();
    int num_points_at_this_pillar = dev_num_points_per_pillar[ith_pillar];

    if (ith_point >= num_points_at_this_pillar) {
          return;
    }

    for (unsigned int d = reduce_size >> 1 ; d > 0.6; d >>= 1) {
        if (ith_point < d) {
            temp[ith_point*3 +axis] += temp[(ith_point + d) * 3 + axis];
        }
        __syncthreads();
    }

    if (ith_point == 0) {
        dev_points_mean[ith_pillar * 3 + axis] = temp[ith_point + axis] / num_points_at_this_pillar ;
    }
}
















__device__ void warpReduce(volatile float* sdata , int ith_point , int axis) {
    sdata[ith_point * blockDim.y + axis] += sdata[(ith_point + 8) * blockDim.y + axis];
    sdata[ith_point * blockDim.y + axis] += sdata[(ith_point + 4) * blockDim.y + axis];
    sdata[ith_point * blockDim.y + axis] += sdata[(ith_point + 2) * blockDim.y + axis];
    sdata[ith_point * blockDim.y + axis] += sdata[(ith_point + 1) * blockDim.y + axis];
}





__global__ void make_pillar_mean_kernel(
  float* dev_points_mean, 
  const int num_point_feature,
  const float* dev_pillar_point_feature, 
  const float* dev_num_points_per_pillar, 
  int max_pillars , 
  int max_points_pre_pillar) {
    extern __shared__ float temp[];
    unsigned int ith_pillar = blockIdx.x;  // { 0 , 1, 2, ... , 10000+}
    unsigned int ith_point  = threadIdx.x; // { 0 , 1, 2, ...,9}
    unsigned int axis = threadIdx.y; 
    unsigned int idx_pre  = ith_pillar * max_points_pre_pillar * num_point_feature \
                     + ith_point  * num_point_feature;
    unsigned int idx_post = ith_pillar * max_points_pre_pillar * num_point_feature \
                     + (ith_point + blockDim.x)  * num_point_feature;

    temp[ith_point * blockDim.y + axis] = 0.0;
    unsigned int num_points_at_this_pillar = dev_num_points_per_pillar[ith_pillar];

    // if (ith_point < num_points_at_this_pillar / 2) {
      temp[ith_point * blockDim.y + axis] = dev_pillar_point_feature[idx_pre  + axis] 
                                          + dev_pillar_point_feature[idx_post + axis];
    // }
    __syncthreads();

    // do reduction in shared mem
    // Sequential addressing. This solves the bank conflicts as
    // the threads now access shared memory with a stride of one
    // 32-bit word (unsigned int) now, which does not cause bank 
    // conflicts
    warpReduce(temp , ith_point , axis);

	// // write result for this block to global mem
    if (ith_point == 0)
    dev_points_mean[ith_pillar * blockDim.y + axis] = temp[ith_point * blockDim.y + axis] / num_points_at_this_pillar ;
}


__global__ void gather_point_feature_kernel(
  const int max_num_pillars_,const int max_num_points_per_pillar,const int num_point_feature,
  const float min_x_range, const float min_y_range, const float min_z_range, 
  const float pillar_x_size,  const float pillar_y_size, const float pillar_z_size,
  const float* dev_pillar_point_feature, const float* dev_num_points_per_pillar, 
  const float* dev_pillar_coors,
  float* dev_points_mean, 
  float* dev_pfe_gather_feature_){

  int ith_pillar = blockIdx.x; 
  int ith_point = threadIdx.x;
  // int kNumPointFeature = 5;
  int num_gather_feature = 11;
  int num_points_at_this_pillar = dev_num_points_per_pillar[ith_pillar];

  if (ith_point >= num_points_at_this_pillar){
        return;
    }


    dev_pfe_gather_feature_[ith_pillar * max_num_points_per_pillar * num_gather_feature + ith_point * num_gather_feature + 0] 
    =  dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar * num_point_feature + ith_point * num_point_feature + 0]; 
  
    dev_pfe_gather_feature_[ith_pillar * max_num_points_per_pillar * num_gather_feature + ith_point * num_gather_feature + 1]  
    =  dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar * num_point_feature + ith_point * num_point_feature + 1];
  
    dev_pfe_gather_feature_[ith_pillar * max_num_points_per_pillar * num_gather_feature + ith_point * num_gather_feature + 2]  
    =  dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar * num_point_feature + ith_point * num_point_feature + 2];
  
    dev_pfe_gather_feature_[ith_pillar * max_num_points_per_pillar * num_gather_feature + ith_point * num_gather_feature + 3]  
    =  dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar * num_point_feature + ith_point * num_point_feature + 3];

    dev_pfe_gather_feature_[ith_pillar * max_num_points_per_pillar * num_gather_feature + ith_point * num_gather_feature + 4]  
    =  dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar * num_point_feature + ith_point * num_point_feature + 4];
  
    // dev_pfe_gather_feature_[ith_pillar * max_num_points_per_pillar * num_gather_feature + ith_point * num_gather_feature + 4]  =  0.0f;
    //   f_cluster = voxel_features[:, :, :3] - points_mean
    dev_pfe_gather_feature_[ith_pillar * max_num_points_per_pillar * num_gather_feature + ith_point * num_gather_feature + 5]  
    =  dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar * num_point_feature + ith_point * num_point_feature + 0] - dev_points_mean[ith_pillar * 3 + 0 ];

    dev_pfe_gather_feature_[ith_pillar * max_num_points_per_pillar * num_gather_feature + ith_point * num_gather_feature + 6] 
    =  dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar * num_point_feature + ith_point * num_point_feature + 1] - dev_points_mean[ith_pillar * 3 + 1 ];
  
    dev_pfe_gather_feature_[ith_pillar * max_num_points_per_pillar * num_gather_feature + ith_point * num_gather_feature + 7]  
    =  dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar * num_point_feature + ith_point * num_point_feature + 2] - dev_points_mean[ith_pillar * 3 + 2 ];

    // f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
    dev_pfe_gather_feature_[ith_pillar * max_num_points_per_pillar * num_gather_feature + ith_point * num_gather_feature + 8]  
    =  dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar * num_point_feature + ith_point * num_point_feature + 0] - (dev_pillar_coors[ith_pillar * 4 + 3] * pillar_x_size + (pillar_x_size/2 + min_x_range));
  
    dev_pfe_gather_feature_[ith_pillar * max_num_points_per_pillar * num_gather_feature + ith_point * num_gather_feature + 9]  
    =  dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar * num_point_feature + ith_point * num_point_feature + 1] - (dev_pillar_coors[ith_pillar * 4 + 2] * pillar_y_size + (pillar_y_size/2 + min_y_range));
  
    dev_pfe_gather_feature_[ith_pillar * max_num_points_per_pillar * num_gather_feature + ith_point * num_gather_feature + 10] 
    =  dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar * num_point_feature + ith_point * num_point_feature + 2] - (dev_pillar_coors[ith_pillar * 4 + 1] * pillar_z_size + (pillar_z_size/2 + min_z_range));

}




PreprocessPointsCuda::PreprocessPointsCuda(
    const int num_threads, const int max_num_pillars,
    const int max_points_per_pillar, const int num_point_feature,
    const int num_inds_for_scan, const int grid_x_size, const int grid_y_size,
    const int grid_z_size, const float pillar_x_size, const float pillar_y_size,
    const float pillar_z_size, const float min_x_range, const float min_y_range,
    const float min_z_range)
    : num_threads_(num_threads),
      max_num_pillars_(max_num_pillars),
      max_num_points_per_pillar_(max_points_per_pillar),
      num_point_feature_(num_point_feature),
      num_inds_for_scan_(num_inds_for_scan),
      grid_x_size_(grid_x_size),
      grid_y_size_(grid_y_size),
      grid_z_size_(grid_z_size),
      pillar_x_size_(pillar_x_size),
      pillar_y_size_(pillar_y_size),
      pillar_z_size_(pillar_z_size),
      min_x_range_(min_x_range),
      min_y_range_(min_y_range),
      min_z_range_(min_z_range) {
    
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_pillar_point_feature_in_coors_),
        grid_y_size_ * grid_x_size_ * max_num_points_per_pillar_ *  num_point_feature_ * sizeof(float)));
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_pillar_count_histo_),
        grid_y_size_ * grid_x_size_ * sizeof(int)));
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_counter_), sizeof(int)));
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_pillar_count_), sizeof(int)));    
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_points_mean_), max_num_pillars_ * 3 *sizeof(float)));  
    }

PreprocessPointsCuda::~PreprocessPointsCuda() {
    GPU_CHECK(cudaFree(dev_pillar_point_feature_in_coors_));
    GPU_CHECK(cudaFree(dev_pillar_count_histo_));
    GPU_CHECK(cudaFree(dev_counter_));
    GPU_CHECK(cudaFree(dev_pillar_count_));
    GPU_CHECK(cudaFree(dev_points_mean_));
  }


void PreprocessPointsCuda::DoPreprocessPointsCuda(
    const float* dev_points, const int in_num_points, 
    int* dev_x_coors,int* dev_y_coors, 
    float* dev_num_points_per_pillar,
    float* dev_pillar_point_feature, float* dev_pillar_coors,
    int* dev_sparse_pillar_map, int* host_pillar_count , float* dev_pfe_gather_feature) {
    // initialize paraments
    GPU_CHECK(cudaMemset(dev_pillar_point_feature_in_coors_, 0 , grid_y_size_ * grid_x_size_ * max_num_points_per_pillar_ *  num_point_feature_ * sizeof(float)));
    GPU_CHECK(cudaMemset(dev_pillar_count_histo_, 0 , grid_y_size_ * grid_x_size_ * sizeof(int)));
    GPU_CHECK(cudaMemset(dev_counter_, 0, sizeof(int)));
    GPU_CHECK(cudaMemset(dev_pillar_count_, 0, sizeof(int)));
    GPU_CHECK(cudaMemset(dev_points_mean_, 0,  max_num_pillars_ * 3 * sizeof(float)));
    int num_block = DIVUP(in_num_points , num_threads_);
    make_pillar_histo_kernel<<<num_block , num_threads_>>>(
        dev_points, dev_pillar_point_feature_in_coors_, dev_pillar_count_histo_,
        in_num_points, max_num_points_per_pillar_, grid_x_size_, grid_y_size_,
        grid_z_size_, min_x_range_, min_y_range_, min_z_range_, pillar_x_size_,
        pillar_y_size_, pillar_z_size_, num_point_feature_);
    
    make_pillar_index_kernel<<<grid_x_size_, grid_y_size_>>>(
        dev_pillar_count_histo_, dev_counter_, dev_pillar_count_, dev_x_coors,
        dev_y_coors, dev_num_points_per_pillar, dev_sparse_pillar_map,
        max_num_pillars_, max_num_points_per_pillar_, grid_x_size_,
        num_inds_for_scan_);  

    GPU_CHECK(cudaMemcpy(host_pillar_count, dev_pillar_count_, 1 * sizeof(int),
        cudaMemcpyDeviceToHost));
    make_pillar_feature_kernel<<<host_pillar_count[0],max_num_points_per_pillar_>>>(
        dev_pillar_point_feature_in_coors_, dev_pillar_point_feature,
        dev_pillar_coors, dev_x_coors, dev_y_coors, dev_num_points_per_pillar,
        max_num_points_per_pillar_, num_point_feature_, grid_x_size_);
    

    dim3 mean_block(max_num_points_per_pillar_,3); //(32,3)

    pillar_mean_kernel<<<host_pillar_count[0],mean_block,64 * 3 *sizeof(float)>>>(
      dev_points_mean_  ,num_point_feature_, dev_pillar_point_feature, dev_num_points_per_pillar, 
        max_num_pillars_ , max_num_points_per_pillar_);

    // dim3 mean_block(10,3); // Unrolling the Last Warp
    // make_pillar_mean_kernel<<<host_pillar_count[0], mean_block , 32 * 3 *sizeof(float)>>>(
    //       dev_points_mean_  ,num_point_feature_, dev_pillar_point_feature, dev_num_points_per_pillar, 
    //       max_num_pillars_ , max_num_points_per_pillar_);

    gather_point_feature_kernel<<<max_num_pillars_, max_num_points_per_pillar_>>>(
      max_num_pillars_,max_num_points_per_pillar_,num_point_feature_,
      min_x_range_, min_y_range_, min_z_range_,
      pillar_x_size_, pillar_y_size_, pillar_z_size_, 
      dev_pillar_point_feature, dev_num_points_per_pillar, dev_pillar_coors,
      dev_points_mean_,
      dev_pfe_gather_feature);

    // DEVICE_SAVE<float>(dev_pillar_point_feature , \
    //     max_num_pillars_ * max_num_points_per_pillar_ * num_point_feature_ , "dev_pillar_point_feature");
    // DEVICE_SAVE<float>(dev_num_points_per_pillar , \
    //   max_num_pillars_ , "dev_num_points_per_pillar");
    // DEVICE_SAVE<float>(dev_pfe_gather_feature , \
    //   max_num_pillars_ * 11, "dev_pfe_gather_feature");
}


