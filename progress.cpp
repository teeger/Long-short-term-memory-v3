#include <iostream>

#include "progress.h"

void progress(int percent)
{
    int count = percent / 5;

    std::cerr << "\r[";
    for (int i = 0; i < 20; i++)
    {
        if (i < count)
            std::cerr << '=';
        else if (i == count)
            std::cerr << '>';
        else
            std::cerr << ' ';
    }
    std::cerr << "]" << percent << '%';

    std::cerr.flush();
}
