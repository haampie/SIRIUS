#include "multi_cg/multi_cg.hpp"

#include <Eigen/Core>

#include <iostream>
#include <complex>

using namespace Eigen;

template<class NumberType>
struct BlockVector {
    Matrix<NumberType,Dynamic,Dynamic> vec;

    typedef NumberType value_type;
    typedef size_t size_type;

    void block_axpy(std::vector<NumberType> alphas, BlockVector const &X, size_t num) {
        DiagonalMatrix<NumberType,Dynamic,Dynamic> D = Map<Matrix<NumberType,Dynamic,1>>(alphas.data(), num).asDiagonal();
        vec.leftCols(num) += X.vec.leftCols(num) * D;
    }

    void block_axpy_scatter(std::vector<NumberType> alphas, BlockVector const &X, std::vector<size_t> ids, size_t num) {
        for (size_t i = 0; i < num; ++i) {
            vec.col(ids[i]) += alphas[i] * X.vec.col(i);
        }
    }

    // rhos[i] = dot(X[i], Y[i])
    void block_dot(BlockVector const &Y, std::vector<NumberType> &rhos, size_t num) {
        Matrix<NumberType,Dynamic,1> result = (vec.leftCols(num).transpose() * Y.vec.leftCols(num)).diagonal();
        Matrix<NumberType,Dynamic,1>::Map(rhos.data(), result.size()) = result;
    }

    // X[:, i] = Z[:, i] + alpha[i] * X[:, i] for i < num_unconverged
    void block_xpby(BlockVector const &Z, std::vector<NumberType> alphas, size_t num) {
        DiagonalMatrix<NumberType,Dynamic,Dynamic> D = Map<Matrix<NumberType,Dynamic,1>>(alphas.data(), num).asDiagonal();
        vec.leftCols(num) = Z.vec.leftCols(num) + vec.leftCols(num) * D;
    }

    void copy(BlockVector const &X, size_t num) {
        vec.leftCols(num) = X.vec.leftCols(num);
    }

    void fill(NumberType val) {
        vec.fill(val);
    }

    auto cols() {
        return vec.cols();
    }

    void repack(std::vector<size_t> const &ids) {
        for (size_t i = 0; i < ids.size(); ++i) {
            auto j = ids[i];
            if (j != i) {
                vec.col(i) = vec.col(j);
            }
        }
    }
};

// This is a linear but special operator A(X)
// producing AX + XD where D_ii = shifts[i] is a diagonal matrix.
// So column-wise it performs (A + shift[i])X[:, i]
// the multiply function basically does a gemv on every column with a different shift
// so alpha * A(X) + beta * Y.
template<class NumberType>
struct PosDefMatrixShifted {
    DiagonalMatrix<NumberType, Dynamic, Dynamic> A;
    Matrix<NumberType, Dynamic, 1> shifts;

    void multiply(NumberType alpha, BlockVector<NumberType> const &u, NumberType beta, BlockVector<NumberType> &v, size_t num) {
        v.vec.leftCols(num) = alpha * A * u.vec.leftCols(num) + alpha * u.vec.leftCols(num) * shifts.head(num).asDiagonal() + beta * v.vec.leftCols(num);
    }

    void repack(std::vector<size_t> const &ids) {
        for (size_t i = 0; i < ids.size(); ++i) {
            auto j = ids[i];
            if (j != i) {
                shifts[i] = shifts[j];
            }
        }
    }
};

template <class NumberType>
struct RandomDiagonalPreconditioner {
    DiagonalMatrix<NumberType, Dynamic, Dynamic> P;

    void apply(BlockVector<NumberType> &C, BlockVector<NumberType> const &B, size_t num) {
        C.vec.leftCols(num) = P * B.vec.leftCols(num);
    }
    void repack(std::vector<size_t> const &ids) {
        // nothing to do;
    }
};

int main() {
    size_t m = 40;
    size_t n = 10;

    // A is just a real diagonal matrix, so it is Hermitian too.
    auto A_shifts = VectorXcd::LinSpaced(n, 1, n);
    auto A_diag = VectorXcd::LinSpaced(m, 1, m);

    auto A = PosDefMatrixShifted<std::complex<double>>{
        A_diag.asDiagonal(),
        A_shifts
    };

    // Also this guy should be Hermitian.
    auto P = RandomDiagonalPreconditioner<std::complex<double>>{
        VectorXcd::LinSpaced(m, 1, 2).asDiagonal()
    };

    auto U = BlockVector<std::complex<double>>{MatrixXcd::Zero(m, n)};
    auto C = BlockVector<std::complex<double>>{MatrixXcd::Zero(m, n)};
    auto X = BlockVector<std::complex<double>>{MatrixXcd::Zero(m, n)};
    auto B = BlockVector<std::complex<double>>{MatrixXcd::Random(m, n)};
    auto R = B;

    auto tol = 1e-10;

    auto resnorms = sirius::cg::multi_cg(
        A, P,
        X, R, U, C,
        100, tol
    );

    // check the residual norms according to the algorithm
    for (size_t i = 0; i < resnorms.size(); ++i) {
        std::cout << "shift " << i << " needed " << resnorms[i].size() << " iterations " << resnorms[i].back() << "\n";

        if (resnorms[i].back() > tol) {
            return 1;
        }
    }

    // True residual norms might be different! because of rounding errors.
    VectorXd true_resnorms = (A_diag.asDiagonal() * X.vec + X.vec * A_shifts.asDiagonal() - B.vec).colwise().norm();

    int return_code = 0;

    for (Eigen::Index i = 0; i < true_resnorms.size(); ++i) {
        std::cout << "true resnorm " << i << ": " << true_resnorms[i] << "\n";
        if (true_resnorms[i] > tol * 500) {
            return_code = 2;
            std::cerr << "ERROR: true resnom > tol * 500\n";
        }
    }

    return return_code;
}