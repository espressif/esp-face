#pragma once

#include <assert.h>
#include <vector>
#include "dl_constant.hpp"
#include "dl_variable.hpp"
#include "dl_tool.hpp"

namespace dl
{
    namespace nn
    {
        /**
         * @brief Concat(input_1, input_2, ...)
         * 
         * @tparam T 
         * @param output 
         * @param features 
         */
        template <typename T>
        void concat2d(Tensor<T> &output, std::vector<Tensor<T>> features);
    } // namespace nn
} // namespace dl