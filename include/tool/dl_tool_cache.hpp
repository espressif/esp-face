#pragma once

#include <stdint.h>

#if CONFIG_IDF_TARGET_ESP32S3
#include "esp32s3/rom/cache.h"
#include "soc/extmem_reg.h"
#endif

namespace dl
{
    namespace tool
    {
        namespace cache
        {
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
        }
    } // namespace tool
} // namespace dl