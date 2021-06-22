#pragma once

#include <vector>
#include <list>
#include <algorithm>
#include <math.h>

#include "dl_variable.hpp"
#include "dl_image.hpp"
#include "dl_define.hpp"
#include "dl_tool.hpp"

#if DL_LOG_DETECT_LATENCY
#define DL_LOG_DETECT_LATENCY_INIT() dl::tool::Latency latency
#define DL_LOG_DETECT_LATENCY_START() latency.start()
#define DL_LOG_DETECT_LATENCY_END(key) \
    latency.end();                     \
    latency.print("detect", key)
#else
#define DL_LOG_DETECT_LATENCY_INIT()
#define DL_LOG_DETECT_LATENCY_START()
#define DL_LOG_DETECT_LATENCY_END(key)
#endif

namespace dl
{
    namespace detect
    {
        typedef struct
        {
            int category;              /*<! category index */
            float score;               /*<! score in logit */
            std::vector<int> box;      /*<! [left_up_x, left_up_y, right_down_x, right_down_y] */
            std::vector<int> keypoint; /*<! [x1, y1, x2, y2, ...] */
        } result_t;

        /**
         * @brief 
         * 
         * @tparam model_input_t 
         * @tparam model_output_t 
         */
        template <typename model_input_t, typename model_output_t>
        class Detect
        {
        public:
            const float score_threshold;         /*<! Candidate box with lower score than score_threshold will be filtered */
            const float nms_threshold;           /*<! Candidate box with higher IoU than nms_threshold will be filtered */
            const bool with_keypoint;            /*<! true: detection with keypoint; false: detection without keypoint */
            const int top_k;                     /*<! Keep top_k number of candidate boxes */
            const float resize_scale;            /*<! resize scale */
            float resize_scale_y;                /*<! Resize scale in vertical */
            float resize_scale_x;                /*<! Resize scale in horizon */
            std::vector<int> input_shape;        /*<! The shape of input */
            std::list<result_t> box_list;        /*<! Detected box list */
            Tensor<model_input_t> resized_input; /*<! Resized input */
            int left_up_y;                       /*<! Cropped patch left up y */
            int left_up_x;                       /*<! Cropped patch left up x */

            /**
             * @brief Construct a new Detect object
             * 
             * @param score_threshold   Candidate box with lower score than score_threshold will be filtered
             * @param nms_threshold     Candidate box with higher IoU than nms_threshold will be filtered
             * @param with_keypoint     true: detection with keypoint; false: detection without keypoint
             * @param top_k             Keep top_k number of candidate boxes
             * @param resize_scale      resize scale
             */
            Detect(const float score_threshold,
                   const float nms_threshold,
                   const bool with_keypoint,
                   const int top_k,
                   const float resize_scale) : score_threshold(score_threshold),
                                               nms_threshold(nms_threshold),
                                               with_keypoint(with_keypoint),
                                               top_k(top_k),
                                               resize_scale(resize_scale),
                                               input_shape({0, 0, 0}),
                                               left_up_y(0),
                                               left_up_x(0) {}

            /**
             * @brief Construct a new Detect object
             * 
             * @param score_threshold   Candidate box with lower score than score_threshold will be filtered
             * @param nms_threshold     Candidate box with higher IoU than nms_threshold will be filtered
             * @param with_keypoint     true: detection with keypoint; false: detection without keypoint
             * @param top_k             Keep top_k number of candidate boxes
             * @param resized_height    Resized image height
             * @param resized_width     Resized image width
             */
            Detect(const float score_threshold,
                   const float nms_threshold,
                   const bool with_keypoint,
                   const int top_k,
                   const int resized_height,
                   const int resized_width) : score_threshold(score_threshold),
                                              nms_threshold(nms_threshold),
                                              with_keypoint(with_keypoint),
                                              top_k(top_k),
                                              resize_scale(0),
                                              input_shape({0, 0, 0}),
                                              left_up_y(0),
                                              left_up_x(0)
            {
                this->resized_input.set_shape({resized_height, resized_width, 3});
            }

            /**
             * @brief Destroy the Detect object
             * 
             */
            ~Detect() {}

            /**
             * @brief Parse the feature map
             * 
             * @param score 
             * @param box 
             * @param stage_index 
             */
            virtual void parse_stage(Tensor<model_output_t> &score, Tensor<model_output_t> &box, const int stage_index) = 0;

            /**
             * @brief Parse the feature map
             * 
             * @param score 
             * @param box 
             * @param keypoint 
             * @param stage_index 
             */
            virtual void parse_stage(Tensor<model_output_t> &score, Tensor<model_output_t> &box, Tensor<model_output_t> &keypoint, const int stage_index) = 0;

            /**
             * @brief Net forward and parse
             * 
             */
            virtual void call() = 0;

            /**
             * @brief Inference
             * 
             * @tparam T 
             * @param input_element 
             * @param input_shape 
             * @return std::list<result_t>& 
             */
            template <typename T>
            std::list<result_t> &infer(T *input_element, std::vector<int> input_shape)
            {
                DL_LOG_DETECT_LATENCY_INIT();

                DL_LOG_DETECT_LATENCY_START();
                if (this->input_shape[0] != input_shape[0] || this->input_shape[1] != input_shape[1] || this->input_shape[2] != input_shape[2])
                {
                    this->input_shape = input_shape;
                    if (this->resize_scale > 0)
                    {
                        int resized_height = int(this->input_shape[0] * this->resize_scale + 0.5);
                        int resized_width = int(this->input_shape[1] * this->resize_scale + 0.5);
                        this->resized_input.set_shape({resized_height, resized_width, input_shape[2]});
                    }
                    this->resize_scale_y = (float)this->input_shape[0] / this->resized_input.shape[0];
                    this->resize_scale_x = (float)this->input_shape[1] / this->resized_input.shape[1];
                }

                // resize
                this->resized_input.calloc_element();
                image::crop_and_resize(this->resized_input.element,
                                       this->resized_input.shape_with_padding[1],
                                       this->resized_input.shape[2],
                                       this->resized_input.padding[0], this->resized_input.padding[0] + this->resized_input.shape[0],
                                       this->resized_input.padding[2], this->resized_input.padding[2] + this->resized_input.shape[1],
                                       input_element,
                                       this->input_shape[0],
                                       this->input_shape[1],
                                       3,
                                       0, this->input_shape[0],
                                       0, this->input_shape[1]);
                DL_LOG_DETECT_LATENCY_END("resize");

                // call
                DL_LOG_DETECT_LATENCY_START();
                this->box_list.clear();
                this->call();
                DL_LOG_DETECT_LATENCY_END("call");

                // NMS
                DL_LOG_DETECT_LATENCY_START();
                int kept_number = 0;
                for (std::list<result_t>::iterator kept = this->box_list.begin(); kept != this->box_list.end(); kept++)
                {
                    kept_number++;

                    if (kept_number >= this->top_k)
                    {
                        this->box_list.erase(++kept, this->box_list.end());
                        break;
                    }

                    int kept_area = (kept->box[2] - kept->box[0] + 1) * (kept->box[3] - kept->box[1] + 1);

                    std::list<result_t>::iterator other = kept;
                    other++;
                    for (; other != this->box_list.end();)
                    {
                        int inter_lt_x = DL_MAX(kept->box[0], other->box[0]);
                        int inter_lt_y = DL_MAX(kept->box[1], other->box[1]);
                        int inter_rb_x = DL_MIN(kept->box[2], other->box[2]);
                        int inter_rb_y = DL_MIN(kept->box[3], other->box[3]);

                        int inter_height = inter_rb_y - inter_lt_y + 1;
                        int inter_width = inter_rb_x - inter_lt_x + 1;

                        if (inter_height > 0 && inter_width > 0)
                        {
                            int other_area = (other->box[2] - other->box[0] + 1) * (other->box[3] - other->box[1] + 1);
                            int inter_area = inter_height * inter_width;
                            float iou = (float)inter_area / (kept_area + other_area - inter_area);
                            if (iou > this->nms_threshold)
                            {
                                other = this->box_list.erase(other);
                                continue;
                            }
                        }
                        other++;
                    }
                }
                DL_LOG_DETECT_LATENCY_END("nms");
                return this->box_list;
            }

            /**
             * @brief Inference
             * 
             * @tparam T 
             * @param input_element 
             * @param candidate 
             * @return std::list<result_t>& 
             */
            template <typename T>
            std::list<result_t> &infer(T *input_element, std::vector<int> input_shape, std::list<result_t> &candidates)
            {
                DL_LOG_DETECT_LATENCY_INIT();

                if (this->resized_input.shape[2] != input_shape[2])
                    this->resized_input.set_shape({this->resized_input.shape[0], this->resized_input.shape[1], input_shape[2]});

                this->box_list.clear();

                for (std::list<result_t>::iterator candidate = candidates.begin(); candidate != candidates.end(); candidate++)
                {
                    DL_LOG_DETECT_LATENCY_START();
                    this->resized_input.calloc_element();

                    std::vector<int> &candidate_box = candidate->box;
                    int center_x = (candidate_box[0] + candidate_box[2]) >> 1;
                    int center_y = (candidate_box[1] + candidate_box[3]) >> 1;
                    int side = DL_MAX(candidate_box[2] - candidate_box[0], candidate_box[3] - candidate_box[1]);

                    this->resize_scale_y = (float)side / this->resized_input.shape[0];
                    this->resize_scale_x = (float)side / this->resized_input.shape[1];

                    // crop and resize
                    image::crop_and_resize(this->resized_input.element,
                                           this->resized_input.shape_with_padding[1],
                                           this->resized_input.shape[2],
                                           this->resized_input.padding[0], this->resized_input.padding[0] + this->resized_input.shape[0],
                                           this->resized_input.padding[2], this->resized_input.padding[2] + this->resized_input.shape[1],
                                           input_element,
                                           input_shape[0],
                                           input_shape[1],
                                           3,
                                           center_y - side / 2, center_y + side / 2,
                                           center_x - side / 2, center_x + side / 2);

                    this->left_up_y = center_y - side / 2;
                    this->left_up_x = center_x - side / 2;
                    DL_LOG_DETECT_LATENCY_END("resize");

                    // call
                    DL_LOG_DETECT_LATENCY_START();
                    this->call();
                    DL_LOG_DETECT_LATENCY_END("call");
                }

                // NMS
                DL_LOG_DETECT_LATENCY_START();
                int kept_number = 0;
                for (std::list<result_t>::iterator kept = this->box_list.begin(); kept != this->box_list.end(); kept++)
                {
                    kept_number++;

                    if (kept_number >= this->top_k)
                    {
                        this->box_list.erase(++kept, this->box_list.end());
                        break;
                    }

                    int kept_area = (kept->box[2] - kept->box[0] + 1) * (kept->box[3] - kept->box[1] + 1);

                    std::list<result_t>::iterator other = kept;
                    other++;
                    for (; other != this->box_list.end();)
                    {
                        int inter_lt_x = DL_MAX(kept->box[0], other->box[0]);
                        int inter_lt_y = DL_MAX(kept->box[1], other->box[1]);
                        int inter_rb_x = DL_MIN(kept->box[2], other->box[2]);
                        int inter_rb_y = DL_MIN(kept->box[3], other->box[3]);

                        int inter_height = inter_rb_y - inter_lt_y + 1;
                        int inter_width = inter_rb_x - inter_lt_x + 1;

                        if (inter_height > 0 && inter_width > 0)
                        {
                            int other_area = (other->box[2] - other->box[0] + 1) * (other->box[3] - other->box[1] + 1);
                            int inter_area = inter_height * inter_width;
                            float iou = (float)inter_area / (kept_area + other_area - inter_area);
                            if (iou > this->nms_threshold)
                            {
                                other = this->box_list.erase(other);
                                continue;
                            }
                        }
                        other++;
                    }
                }
                DL_LOG_DETECT_LATENCY_END("nms");
                return this->box_list;
            }
        };
    }
}
