#pragma once

#include <string>
#include <iostream>

#include "format.h"

template <
    typename Stream,
    typename Arg0 = int,
    typename Arg1 = int,
    typename Arg2 = int,
    typename Arg3 = int,
    typename Arg4 = int,
    typename Arg5 = int,
    typename Arg6 = int,
    typename Arg7 = int>
void print(
    Stream &stream,
    const std::string &_format,
    const Arg0 &arg0 = 0,
    const Arg1 &arg1 = 0,
    const Arg2 &arg2 = 0,
    const Arg3 &arg3 = 0,
    const Arg4 &arg4 = 0,
    const Arg5 &arg5 = 0,
    const Arg6 &arg6 = 0,
    const Arg7 &arg7 = 0)
{
    stream << format(_format, arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7);
}

template <typename... Args>
void print(const std::string &_format, const Args &... args)
{
    print(std::cout, _format, args...);
}

inline void print(const std::string &text)
{
    print(std::cout, text);
}

template <
    typename Stream,
    typename Arg0 = int,
    typename Arg1 = int,
    typename Arg2 = int,
    typename Arg3 = int,
    typename Arg4 = int,
    typename Arg5 = int,
    typename Arg6 = int,
    typename Arg7 = int>
void println(
    Stream &stream,
    const std::string &_format,
    const Arg0 &arg0 = 0,
    const Arg1 &arg1 = 0,
    const Arg2 &arg2 = 0,
    const Arg3 &arg3 = 0,
    const Arg4 &arg4 = 0,
    const Arg5 &arg5 = 0,
    const Arg6 &arg6 = 0,
    const Arg7 &arg7 = 0)
{
    stream << format(_format, arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7) << std::endl;
}

template <typename... Args>
void println(const std::string &_format, const Args &... args)
{
    println(std::cout, _format, args...);
}

inline void println(const std::string &text)
{
    println(std::cout, text);
}
