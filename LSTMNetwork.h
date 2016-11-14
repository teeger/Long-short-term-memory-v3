#pragma once

#include <vector>

#include <Eigen/Dense>

namespace app
{
    using namespace Eigen;

    template <typename T>
    using array = std::vector<T>;

    class LSTMNetwork
    {
    public:

        LSTMNetwork();

        void forward(const array<VectorXd>& input);

        void backward(const array<VectorXd>& teacher);

        void update(int length);

        static constexpr int I = 13;            // Input unit count.
        static constexpr int J = 40;            // Hidden unit count.
        static constexpr int K = 10;            // Output unit count.
        static constexpr int T = 100;           // Maximum time.

        MatrixXd Wz = MatrixXd::Random(J, I);       // Weight of hidden layer from input-layer.
        MatrixXd Rz = MatrixXd::Random(J, J) / 10;  // Weight of inside of hidden-layer.
        MatrixXd Win = MatrixXd::Random(J, I);      // Weight of input gate from input-layer.
        MatrixXd Rin = MatrixXd::Random(J, J) / 10; // Weight of input gate from hidden-layer.
        MatrixXd Wfor = MatrixXd::Random(J, I);     // Weight of forget gate from input-layer.
        MatrixXd Rfor = MatrixXd::Random(J, J) / 10;// Weight of forget gate from hidden-layer.
        MatrixXd Wout = MatrixXd::Random(J, I);     // Weight of output gate from input-layer.
        MatrixXd Rout = MatrixXd::Random(J, J) / 10;// Weight of output gate from hidden-layer.
        MatrixXd Ww = MatrixXd::Random(K, J);       // Weight of output layer from hidden-layer.

        array<VectorXd> x = array<VectorXd>(T);     // Output of input-layer.
        array<VectorXd> z_ = array<VectorXd>(T);    // Input of hidden-layer.
        array<VectorXd> z = array<VectorXd>(T);     // Input of hidden-layer (activated).
        array<VectorXd> y = array<VectorXd>(T);     // Output of hidden-later (activated).
        array<VectorXd> w_ = array<VectorXd>(T);    // Input of output-layer.
        array<VectorXd> w = array<VectorXd>(T);     // Output of output-later (activated).

        array<VectorXd> c = array<VectorXd>(T);     // Output of memory-cell.
        array<VectorXd> i_ = array<VectorXd>(T);    // Input of input-gate.
        array<VectorXd> i = array<VectorXd>(T);     // Output of input-gate (activated) .
        array<VectorXd> f_ = array<VectorXd>(T);    // Input of forget-gate.
        array<VectorXd> f = array<VectorXd>(T);     // Output of forget-gate (activated).
        array<VectorXd> o_ = array<VectorXd>(T);    // Input of output-gate.
        array<VectorXd> o = array<VectorXd>(T);     // Output of output-gate (activated).

        array<VectorXd> δz = array<VectorXd>(T);    // Delta of input unit of hidden-layer.
        array<VectorXd> δc = array<VectorXd>(T);    // Delta of memory cell of hidden-layer..
        array<VectorXd> δi = array<VectorXd>(T);    // Delta of input gate of hidden-layer.
        array<VectorXd> δf = array<VectorXd>(T);    // Delta of forget gate of hidden-layer.
        array<VectorXd> δo = array<VectorXd>(T);    // Delta of output gate unit of hidden-layer..
        array<VectorXd> δy = array<VectorXd>(T);    // Delta of output unit of hidden-layer.
        array<VectorXd> δw = array<VectorXd>(T);    // Delta of output-layer.
    };
}
