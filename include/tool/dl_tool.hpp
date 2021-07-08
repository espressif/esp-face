#pragma once

#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dl_variable.hpp"

#include "esp_system.h"
#include "esp_timer.h"

#if DL_SPIRAM_SUPPORT
#include "freertos/FreeRTOS.h"
#endif

extern "C"
{
#if CONFIG_TIE728_BOOST
    void dl_tie728_memset_8b(void *ptr, const int value, const int n);
    void dl_tie728_memset_16b(void *ptr, const int value, const int n);
    void dl_tie728_memset_32b(void *ptr, const int value, const int n);
#endif
}

namespace dl
{
    namespace tool
    {
        /**
         * @brief Set memory zero.
         * 
         * @param ptr pointer of memory
         * @param n   byte number
         */
        void set_zero(void *ptr, const int n);

        /**
         * @brief Set array value.
         * 
         * @tparam T supports all data type, sizeof(T) equals to 1, 2 and 4 will boost by instruction 
         * @param ptr   pointer of array
         * @param value value to set
         * @param len   length of array
         */
        template <typename T>
        void set_value(T *ptr, const T value, const int len)
        {
#if CONFIG_TIE728_BOOST
            int *temp = (int *)&value;
            if (sizeof(T) == 1)
                dl_tie728_memset_8b(ptr, *temp, len);
            else if (sizeof(T) == 2)
                dl_tie728_memset_16b(ptr, *temp, len);
            else if (sizeof(T) == 4)
                dl_tie728_memset_32b(ptr, *temp, len);
            else
#endif
                for (size_t i = 0; i < len; i++)
                    ptr[i] = value;
        }

        /**
         * @brief Copy memory.
         * 
         * @param dst pointer of destination
         * @param src pointer of source
         * @param n   byte number
         */
        void copy_memory(void *dst, void *src, const int n);

        /**
         * @brief Apply memory with zero-initialized. Must use dl_lib_free() to free the memory.
         * 
         * @param number number of elements
         * @param size   size of element
         * @param align  number of aligned, e.g., 16 means 16-byte aligned
         * @return pointer of allocated memory. NULL for failed
         */
        inline void *calloc_aligned(int number, int size, int align = 0)
        {
            int n = number * size;
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
                printf("Item psram alloc failed. Size: %d = %d x %d + %d + %d\n", total_size, number, size, align, sizeof(void *));
                printf("Available: %d\n", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
#else
                printf("Item alloc failed. Size: %d = %d x %d + %d + %d, SPIRAM_FLAG: %d\n", total_size, number, size, align, sizeof(void *), DL_SPIRAM_SUPPORT);
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

            set_zero(aligned, n);

            return (void *)aligned;
        }

        /**
         * @brief Apply memory without initialized. Must use free_aligned() to free the memory.
         * 
         * @param number number of elements
         * @param size   size of element
         * @param align  number of aligned, e.g., 16 means 16-byte aligned
         * @return pointer of allocated memory. NULL for failed
         */
        inline void *malloc_aligned(int number, int size, int align = 0)
        {
            int total_size = number * size + align + sizeof(void *);
            void *res = malloc(total_size);
            if (NULL == res)
            {
#if DL_SPIRAM_SUPPORT
                res = heap_caps_malloc(total_size, MALLOC_CAP_SPIRAM);
            }
            if (NULL == res)
            {
                printf("Item psram alloc failed. Size: %d = %d x %d + %d + %d\n", total_size, number, size, align, sizeof(void *));
                printf("Available: %d\n", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
#else
                printf("Item alloc failed. Size: %d = %d x %d + %d + %d, SPIRAM_FLAG: %d\n", total_size, number, size, align, sizeof(void *), DL_SPIRAM_SUPPORT);
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

            return (void *)aligned;
        }

        /**
         * @brief Free the calloc_aligned() and malloc_aligned() memory
         * 
         * @param address pointer of memory to free
         */
        inline void free_aligned(void *address)
        {
            if (NULL == address)
                return;

            free(((void **)address)[-1]);
        }

        /**
         * @brief Truncate the input into int8_t range.
         * 
         * @tparam T supports all integer types
         * @param output as an output
         * @param input  as an input
         */
        template <typename T>
        void truncate(int8_t &output, T input)
        {
            if (input >= DL_Q8_MAX)
                output = DL_Q8_MAX;
            else if (input <= DL_Q8_MIN)
                output = DL_Q8_MIN;
            else
                output = input;
        }

        /**
         * @brief Truncate the input into int16_t range.
         * 
         * @tparam T supports all integer types
         * @param output as an output
         * @param input  as an input
         */
        template <typename T>
        void truncate(int16_t &output, T input)
        {
            if (input >= DL_Q16_MAX)
                output = DL_Q16_MAX;
            else if (input <= DL_Q16_MIN)
                output = DL_Q16_MIN;
            else
                output = input;
        }

        /**
         * @brief Print vector in format "[x1, x2, ...]\n".
         * 
         * @param array to print
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
         * @brief Get the cycle object
         * 
         * @return cycle count
         */
        inline uint32_t get_cycle()
        {
            uint32_t ccount;
            __asm__ __volatile__("rsr %0, ccount"
                                 : "=a"(ccount)
                                 :
                                 : "memory");
            return ccount;
        }

        class Latency
        {
        private:
            uint32_t __start; /*<! record the start >*/
            uint32_t __end;   /*<! record the end >*/

        public:
            /**
             * @brief Record the start time.
             * 
             */
            void start()
            {
#if DL_LOG_LATENCY_UNIT
                this->__start = get_cycle();
#else
                this->__start = esp_timer_get_time();
#endif
            }

            /**
             * @brief Record the end time.
             * 
             */
            void end()
            {
#if DL_LOG_LATENCY_UNIT
                this->__end = get_cycle();
#else
                this->__end = esp_timer_get_time();
#endif
            }

            /**
             * @brief Return the period.
             * 
             * @return this->__end - this->__start
             */
            int period()
            {
                return (int)(this->__end - this->__start);
            }

            /**
             * @brief Print in format "latency: {this->period} {unit}\n".
             */
            void print()
            {
#if DL_LOG_LATENCY_UNIT
                printf("latency: %15u cycle\n", this->period());
#else
                printf("latency: %15u us\n", this->period());
#endif
            }

            /**
             * @brief Print in format "{message}: {this->period} {unit}\n".
             * 
             * @param message message of print
             */
            void print(const char *message)
            {
#if DL_LOG_LATENCY_UNIT
                printf("%s: %15u cycle\n", message, this->period());
#else
                printf("%s: %15u us\n", message, this->period());
#endif
            }

            /**
             * @brief Print in format "{prefix}::{key}: {this->period} {unit}\n".
             * 
             * @param prefix prefix of print
             * @param key    key of print
             */
            void print(const char *prefix, const char *key)
            {
#if DL_LOG_LATENCY_UNIT
                printf("%s::%s: %u cycle\n", prefix, key, this->period());
#else
                printf("%s::%s: %u us\n", prefix, key, this->period());
#endif
            }
        };
    } // namespace tool
} // namespace dl