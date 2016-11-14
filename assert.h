#pragma once

#include <stdexcept>

#undef assert

#if 1
        inline void assert(bool expression)
        {
            if (!expression)
                throw std::runtime_error("unknown error");
        }

        inline void assert(bool expression, const char* message)
        {
            if (!expression)
                throw std::runtime_error(message);
        }
#else
        inline void assert(bool) {}
        
        inline void assert(bool, const char*) {}
#endif
