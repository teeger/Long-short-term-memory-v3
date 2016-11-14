#pragma once

#include <stdexcept>

inline void runtime_assert(bool expression)
{
    if (!expression)
        throw std::runtime_error("runtime error.");
}

inline void runtime_assert(bool expression, const char *message)
{
    if (!expression)
        throw std::runtime_error(message);
}
