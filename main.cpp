#include <iostream>
#include <fstream>

#include "format.h"
#include "print.h"
#include "progress.h"
#include "kbhit.h"

#include "LSTMNetwork.h"

using namespace app;
using namespace std;

int main()
{
    constexpr int D = 13;       // Input dimensions.
    constexpr int T = 100;      // Input time length.
    constexpr int K = 10;       // Number of classes.
    constexpr int N = 6600;     // Number of train samples per class.
    constexpr int M = 2200;     // Number of test samples per class.

#pragma region Create training samples.

    vector<vector<VectorXd>> train;
    train.resize(N);
    {
        ifstream fs("Train_Arabic_Digit.txt");
        string line;
        getline(fs, line);

        for (int n = 0; n < N; n++)
        {
            train[n].reserve(T);

            while (!fs.eof())
            {
                getline(fs, line);
                if (line == "            ")
                {
                    break;
                }

                VectorXd v(D);
                sscanf(
                    line.c_str(),
                    "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                    &v(0), &v(1), &v(2), &v(3), &v(4), &v(5), &v(6), &v(7), &v(8), &v(9), &v(10), &v(11), &v(12));
                train[n].emplace_back(v);
            }
        }
    }
#pragma endregion

#pragma region Create test samples.

    vector<vector<VectorXd>> test;
    test.resize(M);
    {
        ifstream fs("Test_Arabic_Digit.txt");
        string line;
        getline(fs, line);

        for (int m = 0; m < M; m++)
        {
            test[m].reserve(T);

            while (!fs.eof())
            {
                getline(fs, line);
                if (line == "            ")
                {
                    break;
                }

                VectorXd v(D);
                sscanf(
                    line.c_str(),
                    "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                    &v(0), &v(1), &v(2), &v(3), &v(4), &v(5), &v(6), &v(7), &v(8), &v(9), &v(10), &v(11), &v(12));
                test[m].emplace_back(v);
            }
        }
    }
#pragma endregion

#pragma region Show examples.
    if (false)
    {
        constexpr int D = 5;
        constexpr int T = 70;
        for (int k = 0; k < K; k++)
        {
            ofstream fs("sample example (class " + to_string(k) + ").txt");
            for (int t = 0; t < T; t++)
            {
                for (int i = 0; i < 10; i++)
                {
                    const int n = k * 660 + i * 66;
                    const auto& sample = train[n];
                    const int T = (int)sample.size();
                    for (int d = 0; d < D; d++)
                    {
                        if (t < T)
                            fs << sample[t](d);
                        fs << '\t';
                    }
                }
                fs << endl;
            }
        }
        for (int k = 0; k < K; k++)
        {
            ofstream fs("sample example (class " + to_string(k) + ").plt");
            fs << "unset key" << endl;
            fs << "set xlabel 't'" << endl;
            fs << "set ylabel 'x(t)'" << endl;
            fs << "set xrange [0:70]" << endl;
            fs << "set yrange [-10:10]" << endl;
            fs << "plot";
            for (int i = 0; i < 10; i++)
            {
                for (int d = 0; d < D; d++)
                {
                    fs << " \"sample example (class " + to_string(k) + ").txt\"";
                    fs << "using 0:" << (i * D + d + 1) << ' ';
                    fs << "with lines ";
                    fs << "lc " << d + 1 << ' ';
                    fs << ",\\" << endl;
                }
            }
        }
        return 0;
    }
#pragma endregion

#pragma region Train network.
    LSTMNetwork network;
    {
        ofstream fs("result.txt");

        vector<VectorXd> teacher;
        teacher.reserve(T);

        VectorXd vote = VectorXd::Zero(K);

        printf("%16s%16s%16s%16s\n", "error(train)", "accuracy(train)", "terror(test)", "accuracy(test)");

        while (!(kbhit() && getchar() == 'q'))
        {
            double train_error = 0, test_error = 0;
            double train_accuracy = 0, test_accuracy = 0;

            for (int n = 0; n < N; n++)
            {
                double& error = train_error;
                double& accuracy = train_accuracy;

                auto& sample = train[n];
                int k = K * n / N;
                int T = (int)sample.size();

                network.forward(sample);
                teacher.resize(T);
                for (int t = 0; t < T; t++)
                {
                    teacher[t].resize(K);
                    teacher[t].fill(0);
                    teacher[t][k] = 1;
                }
                network.backward(teacher);
                network.update(T);

                for (int t = 0; t < T; t++)
                {
                    error += -log(network.w[t](k));
                }

                vote.fill(0);
                for (int t = 0; t < T; t++)
                {
                    vote += network.w[t];
                }
                int estimate_k = 0;
                vote.maxCoeff(&estimate_k);

                if (k == estimate_k)
                {
                    accuracy += 1.0;
                }

                progress(100 * n / N);
            }

            for (int m = 0; m < M; m++)
            {
                double& error = test_error;
                double& accuracy = test_accuracy;

                auto& sample = test[m];
                int k = K * m / M;
                int T = (int)sample.size();

                network.forward(sample);

                for (int t = 0; t < T; t++)
                {
                    error += -log(network.w[t](k));
                }

                vote.fill(0);
                for (int t = 0; t < T; t++)
                {
                    vote += network.w[t];
                }
                int estimate_k = 0;
                vote.maxCoeff(&estimate_k);

                if (k == estimate_k)
                {
                    accuracy += 1.0;
                }

                progress(100 * m / M);
            }

            static int i = 0;
            train_error /= N * T;
            train_accuracy *= 100.0 / N;
            test_error /= M * T;
            test_accuracy *= 100.0 / M;
            // println("\r[{4}]\t{0}\t{1}%\t{2}\t{3}%", train_error, train_accuracy, test_error, test_accuracy, ++i);
            printf("\r[%2d]%12.4lf%16.4lf%%%16.4lf%16.4lf\n", ++i, train_error, train_accuracy, test_error, test_accuracy);
            println(fs, "{0}\t{1}\t{2}\t{3}", train_error, train_accuracy, test_error, test_accuracy);
        }
    }
#pragma endregion
}
