#pragma once

#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dl_variable.hpp"
#include "dl_tool_isa.hpp"

#include "esp_system.h"
#include "esp_timer.h"

#if DL_SPIRAM_SUPPORT
#include "freertos/FreeRTOS.h"
#endif

namespace dl
{
    namespace tool
    {
        /**
         * @brief Allocate a zero-initialized space. Must use 'dl_lib_free' to free the memory.
         * 
         * @param cnt  Count of units.
         * @param size Size of unit.
         * @param align Align of memory. If not required, set 0.
         * @return void* Pointer of allocated memory. Null for failed.
         */
        inline void *calloc_aligned(int cnt, int size, int align = 0)
        {
            int n = cnt * size;
            n >>= 4;
            n += 2;
            n <<= 4;
            int total_size = n + align + sizeof(void *);
            void *res = malloc(total_size);
            if (NULL == res)
            {
#if DL_SPIRAM_SUPPORT
                // printf("Size need: %d, left: %d\n", total_size, heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL));
                // heap_caps_print_heap_info(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
                res = heap_caps_malloc(total_size, MALLOC_CAP_SPIRAM);
            }

            if (NULL == res)
            {
                printf("Item psram alloc failed. Size: %d = %d x %d + %d + %d\n", total_size, cnt, size, align, sizeof(void *));
                printf("Available: %d\n", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
#else
                printf("Item alloc failed. Size: %d = %d x %d + %d + %d, SPIRAM_FLAG: %d\n", total_size, cnt, size, align, sizeof(void *), DL_SPIRAM_SUPPORT);
#endif
                return NULL;
            }
            void **data = (void **)res + 1;
            void **aligned;
            if (align)
                aligned = (void **)(((size_t)data + (align - 1)) & -align);
            else
                aligned = data;

            aligned[-1] = res;

#if CONFIG_TIE728_BOOST
            dl_tie728_bzero_128b(aligned, n);
#else
            bzero(aligned, n);
#endif

            // printf("RAM size: %dKB\n", heap_caps_get_free_size(MALLOC_CAP_8BIT) / 1024);
            return (void *)aligned;
        }

        /**
         * @brief Allocate a un-initialized space. Must use 'free_aligned' to free the memory.
         * 
         * @param cnt  Count of units.
         * @param size Size of unit.
         * @param align Align of memory. If not required, set 0.
         * @return void* Pointer of allocated memory. Null for failed.
         */
        inline void *malloc_aligned(int cnt, int size, int align = 0)
        {
            int total_size = cnt * size + align + sizeof(void *);
            void *res = malloc(total_size);
            if (NULL == res)
            {
#if DL_SPIRAM_SUPPORT
                //printf("Size need: %d, left: %d\n", total_size, heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL));
                //heap_caps_print_heap_info(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
                res = heap_caps_malloc(total_size, MALLOC_CAP_SPIRAM);
            }
            if (NULL == res)
            {
                printf("Item psram alloc failed. Size: %d = %d x %d + %d + %d\n", total_size, cnt, size, align, sizeof(void *));
                printf("Available: %d\n", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
#else
                printf("Item alloc failed. Size: %d = %d x %d + %d + %d, SPIRAM_FLAG: %d\n", total_size, cnt, size, align, sizeof(void *), DL_SPIRAM_SUPPORT);
#endif
                return NULL;
            }
            void **data = (void **)res + 1;
            void **aligned;
            if (align)
                aligned = (void **)(((size_t)data + (align - 1)) & -align);
            else
                aligned = data;

            aligned[-1] = res;

            // printf("RAM size: %dKB\n", heap_caps_get_free_size(MALLOC_CAP_8BIT) / 1024);
            return (void *)aligned;
        }

        /**
         * @brief free the calloc_aligned() and malloc_aligned() memory
         * 
         * @param address 
         */
        inline void free_aligned(void *address)
        {
            if (NULL == address)
                return;

            free(((void **)address)[-1]);
        }

        /**
         * @brief truncate the input into range
         * 
         * @param output 
         * @param input 
         */
        inline void truncate(int16_t &output, int input)
        {
            if (input >= DL_Q16_MAX)
                output = DL_Q16_MAX;
            else if (input <= DL_Q16_MIN)
                output = DL_Q16_MIN;
            else
                output = input;
        }

        /**
         * @brief truncate the input into range
         * 
         * @param output 
         * @param input 
         */
        inline void truncate(int8_t &output, int input)
        {
            if (input >= DL_Q8_MAX)
                output = DL_Q8_MAX;
            else if (input <= DL_Q8_MIN)
                output = DL_Q8_MIN;
            else
                output = input;
        }

        /**
         * @brief truncate the input into range
         * 
         * @param output 
         * @param input 
         */
        inline void truncate(int16_t &output, int64_t input)
        {
            if (input >= DL_Q16_MAX)
                output = DL_Q16_MAX;
            else if (input <= DL_Q16_MIN)
                output = DL_Q16_MIN;
            else
                output = input;
        }

        /**
         * @brief truncate the input into range
         * 
         * @param output 
         * @param input 
         */
        inline void truncate(int8_t &output, int64_t input)
        {
            if (input >= DL_Q8_MAX)
                output = DL_Q8_MAX;
            else if (input <= DL_Q8_MIN)
                output = DL_Q8_MIN;
            else
                output = input;
        }

        /**
         * @brief print vector
         * 
         * @param array 
         */
        inline void print_vector(std::vector<int> &array, const char *message = NULL)
        {
            if (message)
                printf("%s: ", message);

            printf("[");
            for (int i = 0; i < array.size(); i++)
            {
                printf(", %d" + (i ? 0 : 2), array[i]);
            }
            printf("]\n");
        }

        /**
         * @brief check whether two feature are same
         * 
         * @tparam T 
         * @param a 
         * @param b 
         * @return true 
         * @return false 
         */
        template <typename T>
        bool equal(Tensor<T> &a, Tensor<T> &b)
        {
            if (a.exponent != b.exponent)
            {
                printf("Exponent: %d v.s. %d\n", a.exponent, b.exponent);
                return false;
            }

            if (a.shape.size() != b.shape.size())
            {
                printf("Dimension: %d v.s. %d\n", (int)a.shape.size(), (int)b.shape.size());
                return false;
            }

            bool failed = false;
            for (int i = 0; i < a.shape.size(); i++)
            {
                if (a.shape[i] != b.shape[i])
                {
                    failed = true;
                    break;
                }
            }
            if (failed)
            {
                printf("Shape: ");

                for (int i = 0; i < a.shape.size(); i++)
                    printf("%d, ", a.shape[i]);

                printf("v.s. ");

                for (int i = 0; i < b.shape.size(); i++)
                    printf("%d, ", b.shape[i]);

                printf("\n");

                return false;
            }

            for (int y = 0; y < a.shape[0]; y++)
            {
                for (int x = 0; x < a.shape[1]; x++)
                {
                    for (int c = 0; c < a.shape[2]; c++)
                    {
                        int a_i = a.get_element_value({y, x, c});
                        int b_i = b.get_element_value({y, x, c});
                        int offset = DL_ABS(a_i - b_i);
                        if (offset > 2) // rounding mode is different between ESP32 and Python
                        {
                            printf("element[%d, %d, %d]: %d v.s. %d\n", y, x, c, a_i, b_i);
                            return false;
                        }
                    }
                }
            }

            return true;
        }

        class Latency
        {
        private:
            int64_t __start;
            int64_t __end;

        public:
            void start()
            {
                this->__start = esp_timer_get_time();
                // RSR(CCOUNT, this->__start);
            }

            void end()
            {
                this->__end = esp_timer_get_time();
                // RSR(CCOUNT, this->__end);
            }

            int period()
            {
                return (int)(this->__end - this->__start);
            }

            void print()
            {
                printf("latency: %15d us\n", this->period());
            }

            void print(const char *message)
            {
                printf("%s: %15d us\n", message, this->period());
            }

            void print(const char *prefix, const char *key)
            {
                printf("%s::%s: %d us\n", prefix, key, this->period());
            }
        };

        /**
         * @brief           Init preload. call this function to turn on or turn off the preload.  
         * 
         * @param preload   1: turn on the preload. 0: turn off the preload.
         * @return int8_t 
         *                  1: Init sucessfully.
         *                  0: Init suceesfully, autoload has been turned off.
         *                  -1: Init failed, the chip does not support preload.
         */
        int8_t preload_init(uint8_t preload = 1);

        /**
         * @brief           Call preload.
         * 
         * @param addr      The start address of data to be preloaded.
         * @param size      The size(btyes) of the data to be preloaded.
         */
        void preload_func(uint32_t addr, uint32_t size);

        /**
         * @brief           Init autoload. call this function to turn on or turn off the autoload.  
         * 
         * @param autoload  1: turn on the autoload. 0: turn off the autoload.
         * @param trigger   0: miss. 1: hit. 2: both
         * @param linesize  the number of cache lines to be autoloaded.
         * @return int8_t  
         *                  1: Init sucessfully.
         *                  0: Init suceesfully, preload has been turned off.
         *                  -1: Init failed, the chip does not support autoload.
         */
        int8_t autoload_init(uint8_t autoload = 1, uint8_t trigger = 2, uint8_t linesize = 0);

        /**
         * @brief           Call autoload.           
         * 
         * @param addr1     The start address of data1 to be autoloaded.
         * @param size1     The size(btyes) of the data1 to be preloaded.
         * @param addr2     The start address of data2 to be autoloaded.
         * @param size2     The size(btyes) of the data2 to be preloaded.
         */
        void autoload_func(uint32_t addr1, uint32_t size1, uint32_t addr2, uint32_t size2);

        /**
         * @brief           Call autoload. 
         * 
         * @param addr1     The start address of data1 to be autoloaded.
         * @param size1     The size(btyes) of the data1 to be preloaded.
         */
        void autoload_func(uint32_t addr1, uint32_t size1);
    } // namespace tool
} // namespace dl