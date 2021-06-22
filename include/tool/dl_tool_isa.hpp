#pragma once

extern "C"
{
#if CONFIG_XTENSA_BOOST
    void dl_xtensa_bzero_32b(void *ptr, const int n);
#endif

#if CONFIG_TIE728_BOOST
    void dl_tie728_bzero_128b(void *ptr, const int n);
#endif
}
