#pragma once

#include "dl_detect.hpp"

namespace dl
{
    namespace detect
    {
        typedef struct
        {
            int stride_y;
            int stride_x;
            int offset_y;
            int offset_x;
            int min_input_size;
        } anchor_point_stage_t;

        template <typename model_input_t, typename model_output_t>
        class DetectAnchorPoint : public Detect<model_input_t, model_output_t>
        {
        public:
            std::vector<anchor_point_stage_t> stages;

            /**
             * @brief Construct a new Detect Anchor Point object
             * 
             * @param score_threshold 
             * @param nms_threshold 
             * @param with_keypoint 
             * @param top_k 
             * @param resize_scale 
             * @param stages 
             */
            DetectAnchorPoint(const float score_threshold,
                              const float nms_threshold,
                              const bool with_keypoint,
                              const int top_k,
                              const float resize_scale,
                              const std::vector<anchor_point_stage_t> stages);

            /**
             * @brief Destroy the Detect Anchor Point object
             * 
             */
            ~DetectAnchorPoint();

            /**
             * @brief 
             * 
             * @param score 
             * @param box 
             * @param stage_index 
             */
            void parse_stage(Tensor<model_output_t> &score, Tensor<model_output_t> &box, const int stage_index);

            /**
             * @brief 
             * 
             * @param score 
             * @param box 
             * @param keypoint 
             * @param stage_index 
             */
            void parse_stage(Tensor<model_output_t> &score, Tensor<model_output_t> &box, Tensor<model_output_t> &keypoint, const int stage_index);
        };
    }
}
