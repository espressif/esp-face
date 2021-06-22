#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "dl_define.hpp"

namespace dl
{
    namespace image
    {
        typedef enum
        {
            IMAGE_RESIZE_BILINEAR = 0, /*<! Resize image by taking bilinear of four pixels */
            IMAGE_RESIZE_MEAN = 1,     /*<! Resize image by taking mean of four pixels */
            IMAGE_RESIZE_NEAREST = 2   /*<! Resize image by taking the nearest pixel */
        } resize_type_t;

        /**
         * @brief Convert RGB565 pixel to RGB888
         * 
         * @tparam T 
         * @param input     Pixel value in RGB565
         * @param output    Pixel value in RGB888
         */
        template <typename T>
        inline void convert_pixel_rgb565_to_rgb888(uint16_t input, T *output)
        {
            output[0] = (input & 0x1F00) >> 5;                           // blue
            output[1] = ((input & 0x7) << 5) | ((input & 0xE000) >> 11); // green
            output[2] = input & 0xF8;                                    // red
        }

        inline int convert_pixel_rgb_to_gray(int red, int green, int blue)
        {
            int temp = (red * 38 + green * 75 + blue * 15) >> 7;
            return DL_CLIP(temp, 0, 255);
        }

        /**
         * @brief Crop a patch from image and resize this patch.
         * If the cropping box is out of image, destination image will be padded with edge.
         * 
         * The outer rectangle is the entire output image.
         * The inner rectangle is where the resized image will be stored.
         * In other world, this function could help you do padding while resize image.
         *               ___________________________(dst_w)__________________
         *              |         ___________________________                |
         *              |        |(x_start, y_start)         |               | 
         *              |        |                           |               | 
         *              |        |                           |               | 
         *       (dst_h)|        |                           |               | 
         *              |        |                           |               | 
         *              |        |                           |               | 
         *              |        |___________________________|(x_end, y_end) | 
         *              |____________________________________________________| 
         * 
         * @tparam T 
         * @param dst_image     destination image
         * @param dst_width     destination image width
         * @param dst_channel   destination image channel number
         * @param dst_y_start   start y of resized image in destination image
         * @param dst_y_end     end y of resized image in destination image
         * @param dst_x_start   start x of resized image in destination image
         * @param dst_x_end     end x of resized image in destination image
         * @param src_image     source image
         * @param src_height    source image height
         * @param src_width     source image width
         * @param src_channel   source image channel
         * @param src_y_start   start y of resized image in source image
         * @param src_y_end     end y of resized image in source image
         * @param src_x_start   start x of resized image in source image
         * @param src_x_end     end x of resized image in source image
         * @param resize_type   resize type
         * @param shift_left    bit left shift number
         */
        template <typename T>
        void crop_and_resize(T *dst_image,
                             int dst_width,
                             int dst_channel,
                             int dst_y_start, int dst_y_end,
                             int dst_x_start, int dst_x_end,
                             uint16_t *src_image,
                             int src_height,
                             int src_width,
                             int src_channel,
                             int src_y_start, int src_y_end,
                             int src_x_start, int src_x_end,
                             resize_type_t resize_type = IMAGE_RESIZE_NEAREST,
                             int shift_left = 0);

        /**
         * @brief Crop a patch from image and resize this patch.
         * If the cropping box is out of image, destination image will be padded with edge.
         * 
         * The outer rectangle is the entire output image.
         * The inner rectangle is where the resized image will be stored.
         * In other world, this function could help you do padding while resize image.
         *               ___________________________(dst_w)__________________
         *              |         ___________________________                |
         *              |        |(x_start, y_start)         |               | 
         *              |        |                           |               | 
         *              |        |                           |               | 
         *       (dst_h)|        |                           |               | 
         *              |        |                           |               | 
         *              |        |                           |               | 
         *              |        |___________________________|(x_end, y_end) | 
         *              |____________________________________________________| 
         * 
         * @tparam T 
         * @param dst_image     destination image
         * @param dst_width     destination image width
         * @param dst_channel   destination image channel number
         * @param dst_y_start   start y of resized image in destination image
         * @param dst_y_end     end y of resized image in destination image
         * @param dst_x_start   start x of resized image in destination image
         * @param dst_x_end     end x of resized image in destination image
         * @param src_image     source image
         * @param src_height    source image height
         * @param src_width     source image width
         * @param src_channel   source image channel
         * @param src_y_start   start y of resized image in source image
         * @param src_y_end     end y of resized image in source image
         * @param src_x_start   start x of resized image in source image
         * @param src_x_end     end x of resized image in source image
         * @param resize_type   resize type
         * @param shift_left    bit left shift number
         */
        template <typename T>
        void crop_and_resize(T *dst_image,
                             int dst_width,
                             int dst_channel,
                             int dst_y_start, int dst_y_end,
                             int dst_x_start, int dst_x_end,
                             uint8_t *src_image,
                             int src_height,
                             int src_width,
                             int src_channel,
                             int src_y_start, int src_y_end,
                             int src_x_start, int src_x_end,
                             resize_type_t resize_type = IMAGE_RESIZE_NEAREST,
                             int shift_left = 0);

        /**
         * @brief Draw a rectangle on RGB565 image
         * 
         * @param src_image image 
         * @param x1        left-up corner x
         * @param y1        left-up corner y
         * @param x2        right-bottom corner x
         * @param y2        right-bottom corner y
         * @param height    image height
         * @param width     image width
         * @param color     rectangle color
         */
        void draw_rectangle(uint16_t *src_image, int x1, int y1, int x2, int y2, int height, int width, uint16_t color);

        /**
         * @brief Draw a point on RGB565 image
         * 
         * @param src_image image
         * @param x         point x
         * @param y         point y
         * @param height    image height
         * @param width     image width
         * @param color     point color
         * @param size      point size
         */
        void draw_point(uint16_t *src_image, int x, int y, int height, int width, uint16_t color, int size);
    } // namespace image
} // namespace dl
