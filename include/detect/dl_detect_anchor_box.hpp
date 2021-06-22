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
            std::vector<std::vector<int>> anchor_shape;
        } anchor_box_stage_t;

        typedef enum
        {
            CENTER,
            LEFT_UP,
        } regress_type_t;

        typedef enum
        {
            SIGMOID,
            SOFTMAX,
        } score_type_t;

        template <typename model_input_t, typename model_output_t>
        class DetectAnchorBox : public Detect<model_input_t, model_output_t>
        {
        public:
            std::vector<anchor_box_stage_t> stages;
            const regress_type_t regress_type;
            const score_type_t score_type;

            /**
             * @brief Construct a new Detect Anchor Box object
             * 
             * @param score_threshold 
             * @param nms_threshold 
             * @param with_keypoint 
             * @param top_k 
             * @param resize_scale 
             * @param stages 
             * @param regress_type
             * @param score_type
             */
            DetectAnchorBox(const float score_threshold,
                            const float nms_threshold,
                            const bool with_keypoint,
                            const int top_k,
                            const float resize_scale,
                            const std::vector<anchor_box_stage_t> stages,
                            const regress_type_t regress_type = CENTER,
                            const score_type_t score_type = SIGMOID);

            /**
             * @brief Construct a new Detect Anchor Box object
             * 
             * @param score_threshold 
             * @param nms_threshold 
             * @param with_keypoint 
             * @param top_k 
             * @param resized_height 
             * @param resized_width 
             * @param stages 
             * @param regress_type
             * @param score_type
             */
            DetectAnchorBox(const float score_threshold,
                            const float nms_threshold,
                            const bool with_keypoint,
                            const int top_k,
                            const int resized_height,
                            const int resized_width,
                            const std::vector<anchor_box_stage_t> stages,
                            const regress_type_t regress_type = CENTER,
                            const score_type_t score_type = SIGMOID);

            /**
             * @brief Destroy the Detect Anchor Box object
             * 
             */
            ~DetectAnchorBox();

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