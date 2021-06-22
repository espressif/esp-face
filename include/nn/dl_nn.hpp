#pragma once
#include <vector>
#include "dl_define.hpp"
#include "dl_tool.hpp"

namespace dl
{
    namespace nn
    {
        /**
         * @brief Get the output shape object
         * 
         * @param input_shape       input shape
         * @param filter_shape      filter shape
         * @param stride_y          y stride
         * @param stride_x          x stride
         * @param pad_type          padding type
         * @param depthwise         true: depthwise conv2d false: conv2d
         * @return std::vector<int> output shape 
         */
        std::vector<int> get_output_shape(const std::vector<int> &input_shape, const std::vector<int> &filter_shape, const int stride_y, const int stride_x, const padding_type_t pad_type, const bool depthwise);

        /**
         * @brief Get the pad size object
         * 
         * @param output_shape      output shape
         * @param input_shape       input shape
         * @param filter_shape      filter shape
         * @param stride_y          y stride
         * @param stride_x          x stride
         * @param pad_type          padding type
         * @return std::vector<int> padding size
         */
        std::vector<int> get_pad_size(const std::vector<int> &output_shape, const std::vector<int> &input_shape, const std::vector<int> &filter_shape, const int stride_y, const int stride_x, const padding_type_t pad_type);
    } // namespace nn
} // namespace dl

#if DL_LOG_NN_LATENCY
#define DL_LOG_NN_LATENCY_INIT() dl::tool::Latency latency
#define DL_LOG_NN_LATENCY_START() latency.start()
#define DL_LOG_NN_LATENCY_END(key) \
    latency.end();                 \
    latency.print("nn", key)
#else
#define DL_LOG_NN_LATENCY_INIT()
#define DL_LOG_NN_LATENCY_START()
#define DL_LOG_NN_LATENCY_END(key)
#endif
