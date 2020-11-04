// Copyright (c) 2013-2019 Simon Frasch, Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that
// the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file broyden1_mixer.hpp
 *
 *   \brief Contains definition and implementation sirius::Broyden1.
 */

#ifndef __BROYDEN1_MIXER_HPP__
#define __BROYDEN1_MIXER_HPP__

#include <tuple>
#include <functional>
#include <utility>
#include <vector>
#include <limits>
#include <memory>
#include <exception>
#include <cmath>
#include <numeric>

#include "SDDK/memory.hpp"
#include "mixer/mixer.hpp"
#include "linalg/linalg.hpp"

namespace sirius {
namespace mixer {

/// Broyden mixer.
/** First version of the Broyden mixer, which requires inversion of the Jacobian matrix.
 *  Reference paper: "Robust acceleration of self consistent field calculations for
 *  density functional theory", Baarman K, Eirola T, Havu V., J Chem Phys. 134, 134109 (2011)
 */
template <typename... FUNCS>
class Broyden1 : public Mixer<FUNCS...>
{
  private:
    double beta_;
    double beta0_;
    double beta_scaling_factor_;
    std::size_t history_size_;
    mdarray<double, 2> S_old_;
  public:
    Broyden1(std::size_t max_history, double beta, double beta0, double beta_scaling_factor)
        : Mixer<FUNCS...>(max_history)
        , beta_(beta)
        , beta0_(beta0)
        , beta_scaling_factor_(beta_scaling_factor)
        , history_size_(0)
    {
        S_old_ = mdarray<double, 2>(max_history, max_history);
    }

    void mix_impl() override
    {
        const auto idx_step      = this->idx_hist(this->step_);
        const auto idx_next_step = this->idx_hist(this->step_ + 1);
        const auto idx_step_prev = this->idx_hist(this->step_ - 1);

        this->history_size_ = static_cast<int>(std::min(this->history_size_, this->max_history_ - 1));

        bool should_restart = this->step_ > 1 && this->rmse_history_[idx_step_prev] < 0.5 * this->rmse_history_[idx_step];

        if (this->step_ > 0) {
            std::printf("Rank %d. RMSE ratio = %1.4f\n", Communicator::world().rank(), this->rmse_history_[idx_step_prev] / this->rmse_history_[idx_step]);
        }

        if (should_restart) {
            std::printf("Restarting Anderson at %d step %llu\n", Communicator::world().rank(), this->step_);
            this->history_size_ = 0;
        } else {
            std::printf("Not restarting Anderson at %d step %llu\n", Communicator::world().rank(), this->step_);
        }

        const bool normalize = false;

        // beta scaling
        if (this->step_ > this->max_history_) {
            // const double rmse_avg = std::accumulate(this->rmse_history_.begin(), this->rmse_history_.end(), 0.0) /
            //                         this->rmse_history_.size();
            // if (this->rmse_history_[idx_step] > rmse_avg) {
            //     this->beta_ = std::max(beta0_, this->beta_ * beta_scaling_factor_);
            // }
        }

        // set input to 0, to use as buffer
        this->scale(0.0, this->input_);

        if (this->history_size_ > 0) {
            std::cout << "hitting history size > 0\n";
            sddk::mdarray<double, 2> S(this->history_size_, this->history_size_);
            S.zero();
            /* restore S from the previous step */
            for (int j1 = 0; j1 < this->history_size_ - 1; j1++) {
                for (int j2 = 0; j2 < this->history_size_ - 1; j2++) {
                    S(j1 + 1, j2 + 1) = this->S_old_(j1, j2);
                }
            }

            this->copy(this->residual_history_[idx_step], this->tmp2_);
            this->axpy(-1.0, this->residual_history_[idx_step_prev], this->tmp2_);

            for (int j1 = 0; j1 < this->history_size_; j1++) {
                int i1 = this->idx_hist(this->step_ - j1);
                int i2 = this->idx_hist(this->step_ - j1 - 1);
                this->copy(this->residual_history_[i1], this->tmp1_);
                this->axpy(-1.0, this->residual_history_[i2], this->tmp1_);
                S(0, j1) = S(j1, 0) = this->template inner_product<normalize>(this->tmp1_, this->tmp2_);
            }

            for (int j1 = 0; j1 < this->history_size_; j1++) {
                for (int j2 = 0; j2 < this->history_size_; j2++) {
                    S_old_(j1, j2) = S(j1, j2);
                }
            }

            /* invert matrix */
            sddk::linalg(sddk::linalg_t::lapack).syinv(this->history_size_, S);
            /* restore lower triangular part */
            for (int j1 = 0; j1 < this->history_size_; j1++) {
                for (int j2 = 0; j2 < j1; j2++) {
                    S(j1, j2) = S(j2, j1);
                }
            }

            if (Communicator::world().rank() == 0) {
                std::stringstream ss;
                ss.precision(std::numeric_limits<double>::max_digits10);

                ss << "\n\nGram matrix dF' * dF\n";
                for (int j1 = 0; j1 < this->history_size_; j1++) {
                    for (int j2 = 0; j2 < this->history_size_; j2++) {
                        ss << std::setw(18) << S(j2, j1) << ' ';
                    }
                    ss << '\n';
                }
                ss << '\n';

                std::cout << ss.str();
            }

            sddk::mdarray<double, 1> c(this->history_size_);
            c.zero();
            for (int j = 0; j < this->history_size_; j++) {
                int i1 = this->idx_hist(this->step_ - j);
                int i2 = this->idx_hist(this->step_ - j - 1);

                this->copy(this->residual_history_[i1], this->tmp1_);
                this->axpy(-1.0, this->residual_history_[i2], this->tmp1_);

                c(j) = this->template inner_product<normalize>(this->tmp1_, this->residual_history_[idx_step]);
            }

            for (int j = 0; j < this->history_size_; j++) {
                double gamma = 0;
                for (int i = 0; i < this->history_size_; i++) {
                    gamma += c(i) * S(i, j);
                }

                int i1 = this->idx_hist(this->step_ - j);
                int i2 = this->idx_hist(this->step_ - j - 1);

                this->copy(this->residual_history_[i1], this->tmp1_);
                this->axpy(-1.0, this->residual_history_[i2], this->tmp1_);

                this->copy(this->output_history_[i1], this->tmp2_);
                this->axpy(-1.0, this->output_history_[i2], this->tmp2_);

                this->axpy(this->beta_, this->tmp1_, this->tmp2_);
                this->axpy(-gamma, this->tmp2_, this->input_);
            }
        }
        this->copy(this->output_history_[idx_step], this->output_history_[idx_next_step]);
        this->axpy(this->beta_, this->residual_history_[idx_step], this->output_history_[idx_next_step]);
        this->axpy(1.0, this->input_, this->output_history_[idx_next_step]);

        ++this->history_size_;
    }
};
} // namespace mixer
} // namespace sirius

#endif // __MIXER_HPP__
