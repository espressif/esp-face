#pragma once

#include "dl_constant.hpp"
#include "dl_variable.hpp"
#include "dl_nn.hpp"

namespace dl
{
    namespace nn
    {
        /**
         * @brief activate(depthwise_conv2d(input, filter) + bias)
         * 
         * @param output 
         * @param input 
         * @param padding 
         * @param filter 
         * @param stride_y 
         * @param stride_x 
         * @param bias 
         * @param activation 
         * @param assign_core 
         */
        void depthwise_conv2d(Tensor<int16_t> &output,
                              Tensor<int16_t> &input,
                              std::vector<int> &padding,
                              const Filter<int16_t> &filter,
                              const int stride_y,
                              const int stride_x,
                              const Bias<int16_t> *bias = NULL,
                              const Activation<int16_t> *activation = NULL,
                              const std::vector<int> &assign_core = CONFIG_DEFAULT_ASSIGN_CORE);

        void depthwise_conv2d(Tensor<int8_t> &output,
                              Tensor<int8_t> &input,
                              std::vector<int> &padding,
                              const Filter<int8_t> &filter,
                              const int stride_y,
                              const int stride_x,
                              const Bias<int8_t> *bias = NULL,
                              const Activation<int8_t> *activation = NULL,
                              const std::vector<int> &assign_core = CONFIG_DEFAULT_ASSIGN_CORE);

        /**
         * @brief Activation(DepthwiseConv2D(input, filter) + bias)
         * 
         * @tparam feature_t 
         * @param output_exponent 
         * @param input 
         * @param filter 
         * @param stride_y 
         * @param stride_x 
         * @param padding_type 
         * @param bias 
         * @param activation 
         * @return Tensor<feature_t> 
         */
        template <typename feature_t>
        Tensor<feature_t> depthwise_conv2d(const int output_exponent,
                                           Tensor<feature_t> &input,
                                           const Filter<feature_t> &filter,
                                           const int stride_y,
                                           const int stride_x,
                                           const padding_type_t padding_type,
                                           const Bias<feature_t> *bias,
                                           const Activation<feature_t> *activation,
                                           const std::vector<int> &assign_core = CONFIG_DEFAULT_ASSIGN_CORE)
        {
            DL_LOG_NN_LATENCY_INIT();

            DL_LOG_NN_LATENCY_START();
            std::vector<int> output_shape = get_output_shape(input.shape, filter.shape_with_dilation, stride_y, stride_x, padding_type, true);
            Tensor<feature_t> output;
            output.set_exponent(output_exponent).set_shape(output_shape).calloc_element();
            DL_LOG_NN_LATENCY_END("set_shape");

            DL_LOG_NN_LATENCY_START();
            if (padding_type == PADDING_SAME || padding_type == PADDING_SAME_MXNET)
            {
                std::vector<int> padding = get_pad_size(output_shape, input.shape, filter.shape_with_dilation, stride_y, stride_x, padding_type);
                input.set_padding(padding);
            }
            DL_LOG_NN_LATENCY_END("set_padding");

            DL_LOG_NN_LATENCY_START();
            depthwise_conv2d(output, input, input.padding, filter, stride_y, stride_x, bias, activation, assign_core);
            DL_LOG_NN_LATENCY_END("depthwise_conv2d");

            return output;
        }
    } // namespace nn
} // namespace dl