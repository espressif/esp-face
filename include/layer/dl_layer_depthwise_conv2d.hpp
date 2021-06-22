#pragma once

#include "dl_nn_depthwise_conv2d.hpp"
#include "dl_layer_base.hpp"

namespace dl
{
    namespace layer
    {
        /**
         * @brief Activation(DepthwiseConv2D(filter, input) + bias)
         * 
         * @tparam feature_t      type of input
         */
        template <typename feature_t>
        class DepthwiseConv2D : public Layer
        {
        private:
            const Filter<feature_t> *filter;         /*<! filter >*/
            const int stride_y;                      /*<! stride in height >*/
            const int stride_x;                      /*<! stride in width >*/
            const padding_type_t padding_type;       /*<! padding type >*/
            const Bias<feature_t> *bias;             /*<! bias >*/
            const Activation<feature_t> *activation; /*<! activation >*/
            std::vector<int> padding;                /*<! padding >*/

        public:
            Tensor<feature_t> output; /*<! output >*/

            /**
             * @brief Construct a new Depthwise Conv2D object.
             * 
             * @param output_exponent 
             * @param filter 
             * @param bias 
             * @param activation 
             * @param padding_type 
             * @param stride_y 
             * @param stride_x 
             * @param name 
             */
            DepthwiseConv2D(const int output_exponent,
                            const Filter<feature_t> *filter,
                            const Bias<feature_t> *bias = NULL,
                            const Activation<feature_t> *activation = NULL,
                            const padding_type_t padding_type = PADDING_VALID,
                            const int stride_y = 1,
                            const int stride_x = 1,
                            const char *name = NULL) : Layer(name),
                                                       filter(filter),
                                                       stride_y(stride_y),
                                                       stride_x(stride_x),
                                                       padding_type(padding_type),
                                                       bias(bias),
                                                       activation(activation)
            {
                this->output.set_exponent(output_exponent);
            }

            /**
             * @brief Destroy the Depthwise Conv2D object.
             * 
             */
            ~DepthwiseConv2D() {}

            /**
             * @brief update output shape and padding.
             * 
             * @param input 
             */
            void build(Tensor<feature_t> &input)
            {
                assert(input.shape[0] > 0);
                assert(input.shape[1] > 0);

                std::vector<int> output_shape = nn::get_output_shape(input.shape, this->filter->shape_with_dilation, this->stride_y, this->stride_x, this->padding_type, true);
                this->output.set_shape(output_shape);

                this->padding = nn::get_pad_size(output_shape, input.shape, this->filter->shape_with_dilation, this->stride_y, this->stride_x, this->padding_type);
                input.set_padding(this->padding);
            }

            /**
             * @brief calloc output's element if not calloc yet. Call depthwise_conv2d.
             * 
             * @param input 
             * @param autoload_enable true: autoload input and output from PSRAM to CACHE. false: do not. 
             * @return Tensor<feature_t>& 
             */
            Tensor<feature_t> &call(Tensor<feature_t> &input,
                                    bool autoload_enable = false,
                                    const std::vector<int> &assign_core = CONFIG_DEFAULT_ASSIGN_CORE)
            {
                DL_LOG_LAYER_LATENCY_INIT();

                DL_LOG_LAYER_LATENCY_START();
                this->output.calloc_element();
                DL_LOG_LAYER_LATENCY_END(this->name, "calloc");

                if (autoload_enable)
                {
                    dl::tool::cache::autoload_func((uint32_t)(this->output.element), this->output.get_size() * sizeof(feature_t),
                                                   (uint32_t)(input.element), input.get_size() * sizeof(feature_t));
                }

                DL_LOG_LAYER_LATENCY_START();
                nn::depthwise_conv2d(output, input, this->padding, *(this->filter), this->stride_y, this->stride_x, this->bias, this->activation, assign_core);
                DL_LOG_LAYER_LATENCY_END(this->name, "depthwise_conv2d");
                return this->output;
            }

            /**
             * @brief Preload the filter to Cache. 
             * 
             */
            void preload()
            {
                size_t size = sizeof(feature_t);
                int shape_size = this->filter->shape.size();
                for (int i = 0; i < shape_size; ++i)
                {
                    size *= filter->shape[i];
                }
                dl::tool::cache::preload_func((uint32_t)(this->filter->element), size);
            }
        };
    } // namespace layer
} // namespace dl
