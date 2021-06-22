#pragma once

#include "dl_constant.hpp"
#include "dl_variable.hpp"

namespace dl
{
    namespace layer
    {
        /**
         * @brief Neural Network Model
         * 
         * @tparam feature_t 
         */
        template <typename feature_t>
        class Model
        {
        private:
            std::vector<int> input_shape; /*<! input shape >*/

        public:
            virtual ~Model() {}

            /**
             * @brief Build a model including update output shape and input padding of each layer
             * 
             * @param input 
             */
            virtual void build(Tensor<feature_t> &input) = 0;

            /**
             * @brief Call the model layer by layer
             * 
             * @param input 
             */
            virtual void call(Tensor<feature_t> &input) = 0;

            /**
             * @brief build() if input shape change. Then call()
             * 
             * @param input 
             */
            void forward(Tensor<feature_t> &input);
        };
    } // namespace layer
} // namespace dl
