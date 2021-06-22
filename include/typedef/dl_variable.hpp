#pragma once

#include "dl_define.hpp"
#include <stdio.h>
#include <vector>

namespace dl
{
    /**
     * @brief 
     * 
     * @tparam T 
     */
    template <typename T>
    class Tensor
    {
    private:
        int size;       /*<! size of element */
        bool auto_free; /*<! free element when destroy */

    public:
        T *element;                          /*<! element of feature */
        int exponent;                        /*<! exponent of element */
        std::vector<int> shape;              /*<! shape of feature */
        std::vector<int> shape_with_padding; /*<! shape of padded feature */
        std::vector<int> padding;            /*<! For 2D feature, padding format is [top, bottom, left, right] */

        /**
         * @brief Construct a new Tensor object
         * 
         */
        Tensor();

        /**
         * @brief Construct a new Tensor object by copying input feature
         * 
         * @param feature 
         * @param deep 
         */
        Tensor(Tensor<T> &feature, bool deep);

        /**
         * @brief Destroy the Tensor object
         * 
         */
        ~Tensor();

        /**
         * @brief Set the auto free object
         * 
         * @param auto_free 
         * @return Tensor<T>& 
         */
        Tensor<T> &set_auto_free(const bool auto_free)
        {
            this->auto_free = auto_free;
            return *this;
        }

        /**
         * @brief Set the element object
         * 
         * @param element 
         * @return Tensor& 
         */
        Tensor<T> &set_element(T *element, const bool auto_free = false);

        /**
         * @brief Set the exponent object
         * 
         * @param exponent 
         * @return Tensor& 
         */
        Tensor &set_exponent(const int exponent);

        /**
         * @brief Set the shape object
         * 
         * @param shape 
         * @return Tensor& 
         */
        Tensor &set_shape(const std::vector<int> shape);

        /**
         * @brief Set the pad object
         * 
         * If this->element != NULL, free this->element, new an element with padding for this->element
         * 
         * @param padding 
         * @return Tensor& 
         */
        Tensor &set_padding(std::vector<int> &padding);

        /**
         * @brief Get the element ptr object
         * 
         * @param padding 
         * @return T* 
         */
        T *get_element_ptr(const std::vector<int> padding = {0, 0, 0, 0});

        /**
         * @brief Get the element value object
         * 
         * @param index 
         * @param with_padding 
         * @return T& 
         */
        T &get_element_value(const std::vector<int> index, const bool with_padding = false);

        /**
         * @brief Get the size object
         * 
         * @return int 
         */
        int get_size();

        /**
         * @brief calloc element only if this->element == NULL
         * 
         */
        bool calloc_element(const bool auto_free = true);

        /**
         * @brief free element only if this->element != NULL
         * set this->element to NULL, after free
         */
        void free_element();

        /**
         * @brief print the shape of feature
         * 
         */
        void print_shape();

        /**
         * @brief Take numpy for example, this function print Constant[y_start:y_end, x_start:x_end, c]
         * 
         * @param y_start 
         * @param y_end 
         * @param x_start 
         * @param x_end 
         * @param c 
         * @param message 
         * @param with_padding 
         */
        void print2d(const int y_start, const int y_end, const int x_start, const int x_end, const int c, const char *message, const bool with_padding = false);

        /**
         * @brief Internal function for check the element value with simulated element value from Python
         * 
         * @param gt_element 
         * @param bias 
         * @param info 
         * @return true 
         * @return false 
         */
        bool check_element(T *gt_element, int bias = 2, bool info = true)
        {
            if (info)
                this->print_shape();
            int i = 0;
            for (int y = 0; y < this->shape[0]; y++)
            {
                for (int x = 0; x < this->shape[1]; x++)
                {
                    for (int c = 0; c < this->shape[2]; c++)
                    {
                        int a = this->get_element_value({y, x, c});
                        int b = gt_element[i];
                        int offset = DL_ABS(a - b);
                        if (offset > bias) // rounding mode is different between ESP32 and Python
                        {
                            printf("element[%d, %d, %d]: %d v.s. %d\n", y, x, c, a, b);
                            return false;
                        }
                        i++;
                    }
                }
            }

            if (info)
                printf("PASS\n");

            return true;
        }

        /**
         * @brief Internal function for check the input feature has the same shape
         * 
         * @param feature 
         * @return true 
         * @return false 
         */
        bool check_shape(Tensor<T> &feature)
        {
            if (feature.shape.size() != this->shape.size())
            {
                return false;
            }
            for (int i = 0; i < this->shape.size(); i++)
            {
                if (feature.shape[i] != this->shape[i])
                {
                    return false;
                }
            }
            return true;
        }
    };
} // namespace dl