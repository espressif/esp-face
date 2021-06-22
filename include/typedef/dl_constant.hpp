#pragma once

#include "dl_define.hpp"
#include <vector>

namespace dl
{
    /**
     * @brief 
     * 
     * @tparam T 
     */
    template <typename T>
    class Constant
    {
    public:
        const T *element;             /*<! The element of element */
        const int exponent;           /*<! The exponent of element */
        const std::vector<int> shape; /*<! The shape of element */

        Constant(const T *element, const int exponent, const std::vector<int> shape);
    };

    /**
     * @brief 
     * NOTE: The shape format of filter is fixed, but the element sequence depands on optimization.
     * For 1D: reserved
     * For 2D: shape format is [filter_height, filter_width, input_channel, output_channel]. dilation format is [height, width]
     *  
     * @tparam T 
     */
    template <typename T>
    class Filter : public Constant<T>
    {
    public:
        const std::vector<int> dilation;
        std::vector<int> shape_with_dilation;
        Filter(const T *element, const int exponent, const std::vector<int> shape, const std::vector<int> dilation = {1, 1});

        /**
         * @brief Print the n-th filter
         * 
         * @param n 
         * @param message 
         */
        void print2d_n(const int n, const char *message) const;
    };

    /**
     * @brief 
     * 
     * @tparam T 
     */
    template <typename T>
    class Bias : public Constant<T>
    {
    public:
        using Constant<T>::Constant;
    };

    /**
     * @brief
     * 
     * @tparam T 
     */
    template <typename T>
    class Activation : public Constant<T>
    {
    public:
        const activation_type_t type; /*<! The type of activation */

        /**
         * @brief Construct a new Activation object
         * 
         * @param type      Linear, ReLU, LeakyReLU, PReLU
         * @param element   element of activation
         * @param exponent  exponent of element
         * @param shape     shape of element
         */
        Activation(const activation_type_t type, const T *element = NULL, const int exponent = 0, const std::vector<int> shape = {0});
    };
} // namespace dl