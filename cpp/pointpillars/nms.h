
/*
3D IoU Calculation and Rotated NMS(modified from 2D NMS written by others)
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/

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


class NmsCuda {
 private:
  const int num_threads_;
  const int num_box_corners_;
  const float nms_overlap_threshold_;

 public:
  /**
   * @brief Constructor
   * @param[in] num_threads Number of threads when launching cuda kernel
   * @param[in] num_box_corners Number of corners for 2D box
   * @param[in] nms_overlap_threshold IOU threshold for NMS
   * @details Captital variables never change after the compile, Non-captital
   * variables could be chaned through rosparam
   */
  NmsCuda(const int num_threads, const int num_box_corners,
          const float nms_overlap_threshold);

  /**
   * @brief GPU Non-Maximum Suppresion for network output
   * @param[in] host_filter_count Number of filtered output
   * @param[in] dev_sorted_box_for_nms Bounding box output sorted by score
   * @param[out] out_keep_inds Indexes of selected bounding box
   * @param[out] out_num_to_keep Number of kept bounding boxes
   * @details NMS in GPU and postprocessing for selecting box in CPU
   */
  void DoNmsCuda(const int host_filter_count, float* dev_sorted_box_for_nms,
                 long* out_keep_inds, int* out_num_to_keep);
};