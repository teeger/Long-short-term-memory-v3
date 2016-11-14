#include <algorithm>

#include <core/runtime_assert.h>

#include "LSTMNetwork.h"

using namespace std;
using namespace helper::core;

namespace app
{
    template <typename T1, typename T2>
    static auto hadamard(const T1& a, const T2& b)
    {
        return a.array() * b.array();
    }

    template <typename T1, typename T2, typename ...Args>
    static auto hadamard(const T1& a, const T2& b, const Args& ...c)
    {
        return hadamard(a.array() * b.array(), c...);
    }

    template <typename T>
    static auto sigmoid(const T& x)
    {
        static const auto sigmoid = [](double x) { return 1.0 / (1 + exp(-x)); };
        return x.unaryExpr(sigmoid);
    }

    template <typename T>
    static auto sigmoid_diff(const T& x)
    {
        static const auto sigmoid = [](double x) { return 1.0 / (1 + exp(-x)); };
        static const auto sigmoid_diff = [](double x) { return (1 - sigmoid(x)) * sigmoid(x); };
        return x.unaryExpr(sigmoid_diff);
    }

    template <typename T>
    static auto active(const T& x)
    {
        return sigmoid(x);
    }

    template <typename T>
    static auto active_diff(const T& x)
    {
        return sigmoid_diff(x);
    }

    template <typename T>
    static auto softmax(const T& x)
    {
        T y = x.unaryExpr([](double x) { return exp(x); });
        y.array() /= y.sum();
        return y;
    }

    LSTMNetwork::LSTMNetwork()
    {
        for (int t = 0; t < T; t++)
        {
            x[t].resize(I);
            z_[t].resize(J);
            z[t].resize(J);
            y[t].resize(J);
            w_[t].resize(K);
            w[t].resize(K);

            c[t].resize(J);

            i_[t].resize(J);
            i[t].resize(J);
            f_[t].resize(J);
            f[t].resize(J);
            o_[t].resize(J);
            o[t].resize(J);

            δy[t].resize(J);
            δo[t].resize(J);
            δc[t].resize(J);
            δz[t].resize(J);
            δf[t].resize(J);
            δi[t].resize(J);

            δw[t].resize(K);
        }
    }

    void LSTMNetwork::forward(const array<VectorXd>& input)
    {
        const int T = (int)input.size();

        // for t = 0, ...
        for (int t = 0; t < T; t++)
        {
            if (t == 0)
            {
                x[t] = input[t];

                z_[t] = Wz * x[t];

                z[t] = active(z_[t]);

                i_[t] = Win * x[t];

                i[t] = sigmoid(i_[t]);

                f_[t] = Wfor * x[t];

                f[t] = sigmoid(f_[t]);

                c[t] = i[t].array() * active(z[t]).array();

                o_[t] = Wout * x[t] + c[t];

                o[t] = sigmoid(o_[t]);

                y[t] = o[t].array() * active(c[t]).array();

                w_[t] = Ww * y[t];

                w[t] = softmax(w_[t]);

                runtime_assert(!x[t].hasNaN());
                runtime_assert(!z[t].hasNaN());
                runtime_assert(!i[t].hasNaN());
                runtime_assert(!f[t].hasNaN());
                runtime_assert(!c[t].hasNaN());
                runtime_assert(!o[t].hasNaN());
                runtime_assert(!y[t].hasNaN());
                runtime_assert(!w_[t].hasNaN());
                runtime_assert(!w[t].hasNaN());
            }
            else
            {
                x[t] = input[t];

                z_[t] = Wz * x[t] + Rz * y[t - 1];

                z[t] = active(z_[t]);

                i_[t] = Win * x[t] + Rin * y[t - 1] + c[t - 1];

                i[t] = sigmoid(i_[t]);

                f_[t] = Wfor * x[t] + Rfor * y[t - 1] + c[t - 1];

                f[t] = sigmoid(f_[t]);

                c[t] = f[t].array() * c[t - 1].array() + i[t].array() * active(z[t]).array();

                o_[t] = Wout * x[t] + Rout * y[t - 1] + c[t];

                o[t] = sigmoid(o_[t]);

                y[t] = o[t].array() * active(c[t]).array();

                w_[t] = Ww * y[t];

                w[t] = softmax(w_[t]);

                runtime_assert(!x[t].hasNaN());
                runtime_assert(!z[t].hasNaN());
                runtime_assert(!i[t].hasNaN());
                runtime_assert(!f[t].hasNaN());
                runtime_assert(!c[t].hasNaN());
                runtime_assert(!o[t].hasNaN());
                runtime_assert(!y[t].hasNaN());
                runtime_assert(!w[t].hasNaN());
            }
        }
    }

    void LSTMNetwork::backward(const array<VectorXd>& teacher)
    {
        const int T = (int)teacher.size();

        const array<VectorXd>& d = teacher;

        // for t = T, ...
        for (int t = T - 1; t >= 0; t--)
        {
            if (t == T - 1)
            {
                δw[t] = w[t] - d[t];

                δy[t] = Ww.transpose() * δw[t];

                δo[t] = hadamard(sigmoid_diff(o_[t]), active(c[t]), δy[t]);

                δc[t] = hadamard(δy[t], o[t], active_diff(c[t])) + δo[t].array();

                δf[t] = hadamard(sigmoid_diff(f_[t]), c[t - 1], δc[t]);

                δi[t] = hadamard(sigmoid_diff(i_[t]), z[t], δc[t]);

                δz[t] = hadamard(δc[t], i[t], active_diff(z_[t]));
            }
            else
            {
                δw[t] = w[t] - d[t];

                δy[t] = Ww.transpose() * δw[t] + Rz.transpose() * δz[t + 1] + Rin.transpose() * δi[t + 1] + Rfor.transpose() * δf[t + 1] + Rout.transpose() * δo[t + 1];

                δo[t] = hadamard(sigmoid_diff(o_[t]), active(c[t]), δy[t]);

                δc[t] = hadamard(δy[t], o[t], active_diff(c[t]))
                    + δo[t].array()
                    + δi[t + 1].array()
                    + δf[t + 1].array()
                    + f[t + 1].array() * δc[t + 1].array();

                if (t > 0)
                    δf[t] = hadamard(sigmoid_diff(f_[t]), c[t - 1], δc[t]);
                else
                    δf[t].fill(0);

                δi[t] = hadamard(sigmoid_diff(i_[t]), z[t], δc[t]);

                δz[t] = hadamard(δc[t], i[t], active_diff(z_[t]));
            }
        }
    }

    void LSTMNetwork::update(int length)
    {
        const int T = length;

        constexpr double ε = 0.000005;

        for (int t = 0; t < T; t++)
        {
            if (t == 0)
            {
                Wz -= ε * δz[t] * x[t].transpose();

                Ww -= ε * δw[t] * y[t].transpose();

                Win -= ε * δi[t] * x[t].transpose();

                Wfor -= ε * δf[t] * x[t].transpose();

                Wout -= ε * δo[t] * x[t].transpose();
            }
            else
            {
                Wz -= ε * δz[t] * x[t].transpose();

                Rz -= 0.1 * ε * δz[t] * y[t - 1].transpose();

                Ww -= ε * δw[t] * y[t].transpose();

                Win -= ε * δi[t] * x[t].transpose();

                Rin -= 0.1 * ε * δi[t] * y[t - 1].transpose();

                Wfor -= ε * δf[t] * x[t].transpose();

                Rfor -= 0.1 * ε * δf[t] * y[t - 1].transpose();

                Wout -= ε * δo[t] * x[t].transpose();

                Rout -= 0.1 * ε * δo[t] * y[t - 1].transpose();
            }
        }

        runtime_assert(!Wz.hasNaN());
        runtime_assert(!Rz.hasNaN());
        runtime_assert(!Ww.hasNaN());
        runtime_assert(!Win.hasNaN());
        runtime_assert(!Rin.hasNaN());
        runtime_assert(!Wfor.hasNaN());
        runtime_assert(!Rfor.hasNaN());
        runtime_assert(!Wout.hasNaN());
        runtime_assert(!Rout.hasNaN());
    }
}
