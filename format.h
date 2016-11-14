#pragma once

#include <sstream>
#include <string>

template <
    typename Arg0 = int,
    typename Arg1 = int,
    typename Arg2 = int,
    typename Arg3 = int,
    typename Arg4 = int,
    typename Arg5 = int,
    typename Arg6 = int,
    typename Arg7 = int>
std::string format(
    const std::string &format,
    const Arg0 &arg0 = 0,
    const Arg1 &arg1 = 0,
    const Arg2 &arg2 = 0,
    const Arg3 &arg3 = 0,
    const Arg4 &arg4 = 0,
    const Arg5 &arg5 = 0,
    const Arg6 &arg6 = 0,
    const Arg7 &arg7 = 0)
{
    using std::string;
    using std::stringstream;

    stringstream ss;

    auto len = format.length();
    for (size_t i = 0; i < len; i++)
    {
        if (format[i] == '{' && isdigit(format[i + 1]) && format[i + 2] == '}')
        {
            int n = format[i + 1] - '0';
            switch (n)
            {
            case 0:
                ss << arg0;
                break;
            case 1:
                ss << arg1;
                break;
            case 2:
                ss << arg2;
                break;
            case 3:
                ss << arg3;
                break;
            case 4:
                ss << arg4;
                break;
            case 5:
                ss << arg5;
                break;
            case 6:
                ss << arg6;
                break;
            case 7:
                ss << arg7;
                break;
            default:
                break;
            }
            i += 2;
        }
        else
        {
            ss << format[i];
        }
    }

    return ss.str();
}
