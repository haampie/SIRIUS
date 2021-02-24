// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
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

/** \file wave_functions.cpp
 *
 *  \brief Definitions.
 *
 */
#include "wave_functions.hpp"

namespace sddk {

Wave_functions::Wave_functions(const Gvec_partition& gkvecp__, int num_wf__, memory_t preferred_memory_t__,
                               int num_sc__)
    : comm_(gkvecp__.gvec().comm())
    , gkvecp_(gkvecp__)
    , num_wf_(num_wf__)
    , num_sc_(num_sc__)
    , preferred_memory_t_(preferred_memory_t__)
{
    if (!(num_sc__ == 1 || num_sc__ == 2)) {
        TERMINATE("wrong number of spin components");
    }

    for (int ispn = 0; ispn < num_sc_; ispn++) {
        pw_coeffs_[ispn] = std::unique_ptr<matrix_storage<double_complex, matrix_storage_t::slab>>(
            new matrix_storage<double_complex, matrix_storage_t::slab>(gkvecp_, num_wf_));
    }
}

Wave_functions::Wave_functions(memory_pool& mp__, const Gvec_partition& gkvecp__, int num_wf__,
                               memory_t preferred_memory_t__, int num_sc__)
    : comm_(gkvecp__.gvec().comm())
    , gkvecp_(gkvecp__)
    , num_wf_(num_wf__)
    , num_sc_(num_sc__)
    , preferred_memory_t_(preferred_memory_t__)
{
    if (!(num_sc__ == 1 || num_sc__ == 2)) {
        TERMINATE("wrong number of spin components");
    }
    if (!is_host_memory(mp__.memory_type())) {
        TERMINATE("this is not a host memory pool");
    }

    for (int ispn = 0; ispn < num_sc_; ispn++) {
        pw_coeffs_[ispn] = std::unique_ptr<matrix_storage<double_complex, matrix_storage_t::slab>>(
            new matrix_storage<double_complex, matrix_storage_t::slab>(mp__, gkvecp_, num_wf_));
    }
}

Wave_functions::Wave_functions(const Gvec_partition& gkvecp__, int num_atoms__, std::function<int(int)> mt_size__,
                               int num_wf__, memory_t preferred_memory_t__, int num_sc__)
    : comm_(gkvecp__.gvec().comm())
    , gkvecp_(gkvecp__)
    , num_wf_(num_wf__)
    , num_sc_(num_sc__)
    , has_mt_(true)
    , preferred_memory_t_(preferred_memory_t__)
{
    if (!(num_sc__ == 1 || num_sc__ == 2)) {
        TERMINATE("wrong number of spin components");
    }

    for (int ispn = 0; ispn < num_sc_; ispn++) {
        pw_coeffs_[ispn] = std::unique_ptr<matrix_storage<double_complex, matrix_storage_t::slab>>(
            new matrix_storage<double_complex, matrix_storage_t::slab>(gkvecp_, num_wf_));
    }

    spl_num_atoms_   = splindex<splindex_t::block>(num_atoms__, comm_.size(), comm_.rank());
    mt_coeffs_distr_ = block_data_descriptor(comm_.size());

    for (int ia = 0; ia < num_atoms__; ia++) {
        int rank = spl_num_atoms_.local_rank(ia);
        if (rank == comm_.rank()) {
            offset_mt_coeffs_.push_back(mt_coeffs_distr_.counts[rank]);
        }
        mt_coeffs_distr_.counts[rank] += mt_size__(ia);
    }
    mt_coeffs_distr_.calc_offsets();

    num_mt_coeffs_ = mt_coeffs_distr_.offsets.back() + mt_coeffs_distr_.counts.back();

    for (int ispn = 0; ispn < num_sc_; ispn++) {
        mt_coeffs_[ispn] = std::unique_ptr<matrix_storage<double_complex, matrix_storage_t::slab>>(
            new matrix_storage<double_complex, matrix_storage_t::slab>(mt_coeffs_distr_.counts[comm_.rank()], num_wf_));
    }
}

void Wave_functions::copy_from(device_t pu__, int n__, const Wave_functions& src__, int ispn__, int i0__,
                               int jspn__, int j0__)
{
    assert(ispn__ == 0 || ispn__ == 1);
    assert(jspn__ == 0 || jspn__ == 1);

    int ngv = pw_coeffs(jspn__).num_rows_loc();
    int nmt = has_mt() ? mt_coeffs(jspn__).num_rows_loc() : 0;

    switch (pu__) {
        case device_t::CPU: {
            /* copy PW part */
            std::copy(src__.pw_coeffs(ispn__).prime().at(memory_t::host, 0, i0__),
                      src__.pw_coeffs(ispn__).prime().at(memory_t::host, 0, i0__) + ngv * n__,
                      pw_coeffs(jspn__).prime().at(memory_t::host, 0, j0__));
            /* copy MT part */
            if (has_mt()) {
                std::copy(src__.mt_coeffs(ispn__).prime().at(memory_t::host, 0, i0__),
                          src__.mt_coeffs(ispn__).prime().at(memory_t::host, 0, i0__) + nmt * n__,
                          mt_coeffs(jspn__).prime().at(memory_t::host, 0, j0__));
            }
            break;
        }
        case device_t::GPU: {
            /* copy PW part */
            acc::copy(pw_coeffs(jspn__).prime().at(memory_t::device, 0, j0__),
                      src__.pw_coeffs(ispn__).prime().at(memory_t::device, 0, i0__), ngv * n__);
            /* copy MT part */
            if (has_mt()) {
                acc::copy(mt_coeffs(jspn__).prime().at(memory_t::device, 0, j0__),
                          src__.mt_coeffs(ispn__).prime().at(memory_t::device, 0, i0__), nmt * n__);
            }
            break;
        }
    }
}

void Wave_functions::copy_from(const Wave_functions& src__, int n__, int ispn__, int i0__, int jspn__, int j0__)
{
    assert(ispn__ == 0 || ispn__ == 1);
    assert(jspn__ == 0 || jspn__ == 1);

    int ngv = pw_coeffs(jspn__).num_rows_loc();
    int nmt = has_mt() ? mt_coeffs(jspn__).num_rows_loc() : 0;

    copy(src__.preferred_memory_t(), src__.pw_coeffs(ispn__).prime().at(src__.preferred_memory_t(), 0, i0__),
         preferred_memory_t(), pw_coeffs(jspn__).prime().at(preferred_memory_t(), 0, j0__), ngv * n__);
    if (has_mt()) {
        copy(src__.preferred_memory_t(), src__.mt_coeffs(ispn__).prime().at(src__.preferred_memory_t(), 0, i0__),
             preferred_memory_t(), mt_coeffs(jspn__).prime().at(preferred_memory_t(), 0, j0__), nmt * n__);
    }
}

double_complex Wave_functions::checksum_pw(device_t pu__, int ispn__, int i0__, int n__) const
{
    assert(n__ != 0);
    double_complex cs(0, 0);
    for (int s = s0(ispn__); s <= s1(ispn__); s++) {
        cs += pw_coeffs(s).checksum(pu__, i0__, n__);
    }
    comm_.allreduce(&cs, 1);
    return cs;
}

double_complex Wave_functions::checksum_mt(device_t pu__, int ispn__, int i0__, int n__) const
{
    assert(n__ != 0);
    double_complex cs(0, 0);
    if (!this->has_mt_) {
        return cs;
    }
    for (int s = s0(ispn__); s <= s1(ispn__); s++) {
        if (mt_coeffs_distr_.counts[comm_.rank()]) {
            cs += mt_coeffs(s).checksum(pu__, i0__, n__);
        }
    }
    comm_.allreduce(&cs, 1);
    return cs;
}

void Wave_functions::print_checksum(device_t pu__, std::string label__, int N__, int n__) const
{
    for (int ispn = 0; ispn < num_sc(); ispn++) {
        auto cs1 = this->checksum_pw(pu__, ispn, N__, n__);
        auto cs2 = this->checksum_mt(pu__, ispn, N__, n__);
        if (this->comm().rank() == 0) {
            std::stringstream s;
            s << ispn;
            utils::print_checksum(label__ + "_pw_" + s.str(), cs1);
            if (this->has_mt_) {
                utils::print_checksum(label__ + "_mt_" + s.str(), cs2);
            }
            utils::print_checksum(label__ + "_" + s.str(), cs1 + cs2);
        }
    }
}

void Wave_functions::zero_pw(device_t pu__, int ispn__, int i0__, int n__) // TODO: pass memory_t
{
    for (int s = s0(ispn__); s <= s1(ispn__); s++) {
        switch (pu__) {
            case device_t::CPU: {
                pw_coeffs(s).zero(memory_t::host, i0__, n__);
                break;
            }
            case device_t::GPU: {
                pw_coeffs(s).zero(memory_t::device, i0__, n__);
                break;
            }
        }
    }
}

void Wave_functions::zero_mt(device_t pu__, int ispn__, int i0__, int n__) // TODO: pass memory_t
{
    if (!has_mt()) {
        return;
    }
    for (int s = s0(ispn__); s <= s1(ispn__); s++) {
        switch (pu__) {
            case device_t::CPU: {
                mt_coeffs(s).zero(memory_t::host, i0__, n__);
                break;
            }
            case device_t::GPU: {
                mt_coeffs(s).zero(memory_t::device, i0__, n__);
                break;
            }
        }
    }
}

void Wave_functions::scale(memory_t mem__, int ispn__, int i0__, int n__, double beta__)
{
    for (int s = s0(ispn__); s <= s1(ispn__); s++) {
        pw_coeffs(s).scale(mem__, i0__, n__, beta__);
        if (has_mt()) {
            mt_coeffs(s).scale(mem__, i0__, n__, beta__);
        }
    }
}

mdarray<double, 1>
Wave_functions::l2norm(device_t pu__, spin_range spins__, int n__) const
{
    assert(n__ != 0);

    auto norm = sumsqr(pu__, spins__, n__);
    for (int i = 0; i < n__; i++) {
        norm[i] = std::sqrt(norm[i]);
    }

    return norm;
}

void Wave_functions::normalize(device_t pu__, spin_range spins__, int n__)
{
    auto norm = this->l2norm(pu__, spins__, n__);
    for (int i = 0; i < n__; i++) {
        norm[i] = 1.0 / norm[i];
    }
    if (pu__ == device_t::GPU) {
        norm.copy_to(memory_t::device);
    }
    for (int ispn : spins__) {
        switch (pu__) {
            case device_t::CPU: {
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < n__; i++) {
                    for (int ig = 0; ig < this->pw_coeffs(ispn).num_rows_loc(); ig++) {
                        this->pw_coeffs(ispn).prime(ig, i) *= norm[i];
                    }
                    if (this->has_mt()) {
                        for (int j = 0; j < this->mt_coeffs(ispn).num_rows_loc(); j++) {
                            this->mt_coeffs(ispn).prime(j, i) *= norm[i];
                        }
                    }
                }
                break;
            }
            case device_t::GPU: {
#if defined(SIRIUS_GPU)
                scale_matrix_columns_gpu(this->pw_coeffs(ispn).num_rows_loc(), n__,
                                         (acc_complex_double_t*)this->pw_coeffs(ispn).prime().at(memory_t::device),
                                         norm.at(memory_t::device));

                if (this->has_mt()) {
                    scale_matrix_columns_gpu(this->mt_coeffs(ispn).num_rows_loc(), n__,
                                             (acc_complex_double_t*)this->mt_coeffs(ispn).prime().at(memory_t::device),
                                             norm.at(memory_t::device));
                }
#endif
            } break;
        }
    }
}

void Wave_functions::allocate(spin_range spins__, memory_t mem__)
{
    for (int s : spins__) {
        pw_coeffs(s).allocate(mem__);
        if (has_mt()) {
            mt_coeffs(s).allocate(mem__);
        }
    }
}

void Wave_functions::allocate(spin_range spins__, memory_pool& mp__)
{
    for (int s : spins__) {
        pw_coeffs(s).allocate(mp__);
        if (has_mt()) {
            mt_coeffs(s).allocate(mp__);
        }
    }
}

void Wave_functions::deallocate(spin_range spins__, memory_t mem__)
{
    for (int s : spins__) {
        pw_coeffs(s).deallocate(mem__);
        if (has_mt()) {
            mt_coeffs(s).deallocate(mem__);
        }
    }
}

void Wave_functions::copy_to(spin_range spins__, memory_t mem__, int i0__, int n__)
{
    for (int s : spins__) {
        pw_coeffs(s).copy_to(mem__, i0__, n__);
        if (has_mt()) {
            mt_coeffs(s).copy_to(mem__, i0__, n__);
        }
    }
}

mdarray<double_complex, 1>
Wave_functions::dot(device_t pu__, spin_range spins__, Wave_functions const &phi, int n__) const
{
    mdarray<double_complex, 1> s(n__, memory_t::host, "dot");
    s.zero();
    if (pu__ == device_t::GPU) {
        s.allocate(memory_t::device).zero(memory_t::device);
    }

    for (int is : spins__) {
        switch (pu__) {
            case device_t::CPU: {
                #pragma omp parallel for
                for (int i = 0; i < n__; i++) {
                    for (int ig = 0; ig < pw_coeffs(is).num_rows_loc(); ig++) {
                        auto x = pw_coeffs(is).prime(ig, i);
                        auto y = phi.pw_coeffs(is).prime(ig, i);
                        s[i] += std::conj(x) * y;
                    }
                    // todo, do something here.
                    // if (gkvecp_.gvec().reduced()) {
                    //     if (comm_.rank() == 0) {
                    //         s[i] = 2 * s[i] - std::pow(pw_coeffs(is).prime(0, i).real(), 2);
                    //     } else {
                    //         s[i] *= 2;
                    //     }
                    // }
                    if (has_mt()) {
                        for (int j = 0; j < mt_coeffs(is).num_rows_loc(); j++) {
                            auto x = mt_coeffs(is).prime(j, i);
                            auto y = phi.mt_coeffs(is).prime(j, i);
                            s[i] += std::conj(x) * y;
                        }
                    }
                }
                break;
            }
            case device_t::GPU: {
#if defined(SIRIUS_GPU)
                wf_dot_gpu(
                    pw_coeffs(is).prime().at(memory_t::device),
                    phi.pw_coeffs(is).prime().at(memory_t::device),
                    pw_coeffs(is).num_rows_loc(),
                    n__,
                    gkvecp_.gvec().reduced(),
                    comm_.rank(),
                    s.at(memory_t::device));
                if (has_mt()) {
                    wf_dot_gpu(
                        mt_coeffs(is).prime().at(memory_t::device),
                        phi.mt_coeffs(is).prime().at(memory_t::device),
                        mt_coeffs(is).num_rows_loc(),
                        n__,
                        0,
                        comm_.rank(),
                        s.at(memory_t::device));
                }
#endif
            }
        }
    }

    if (pu__ == device_t::GPU) {
        s.copy_to(memory_t::host);
    }

    comm_.allreduce(s.at(memory_t::host), n__);

    return s;
}

template <class Ta>
void
Wave_functions::axpby(device_t pu__, spin_range spins__, Ta alpha, Wave_functions const &phi, Ta beta, int n__)
{
    for (int is : spins__) {
        switch (pu__) {
            case device_t::CPU: {
                #pragma omp parallel for
                for (int i = 0; i < n__; i++) {
                    for (int ig = 0; ig < pw_coeffs(is).num_rows_loc(); ig++) {
                        auto x = phi.pw_coeffs(is).prime(ig, i);
                        auto y = pw_coeffs(is).prime(ig, i);
                        
                        pw_coeffs(is).prime(ig, i) = alpha * x + beta * y;
                    }
                    if (has_mt()) {
                        for (int j = 0; j < mt_coeffs(is).num_rows_loc(); j++) {
                            auto x = phi.mt_coeffs(is).prime(j, i);
                            auto y = mt_coeffs(is).prime(j, i);
                            mt_coeffs(is).prime(j, i) = alpha * x + beta * y;
                        }
                    }
                }
                break;
            }
            case device_t::GPU: {
#if defined(SIRIUS_GPU)
                wf_axpby_gpu(
                    static_cast<double_complex>(alpha),
                    phi.pw_coeffs(is).prime().at(memory_t::device),
                    static_cast<double_complex>(beta),
                    pw_coeffs(is).prime().at(memory_t::device),
                    pw_coeffs(is).num_rows_loc(),
                    n__);
                if (has_mt()) {
                    wf_axpby_gpu(
                        static_cast<double_complex>(alpha),
                        phi.mt_coeffs(is).prime().at(memory_t::device),
                        static_cast<double_complex>(beta),
                        mt_coeffs(is).prime().at(memory_t::device),
                        mt_coeffs(is).num_rows_loc(),
                        n__);
                }
#endif
            }
        }
    }
}

template <class Ta>
void
Wave_functions::xpby(device_t pu__, spin_range spins__, Wave_functions const &phi, sddk::mdarray<Ta,1> &betas, int n__)
{
    for (int is : spins__) {
        switch (pu__) {
            case device_t::CPU: {
                #pragma omp parallel for
                for (int i = 0; i < n__; i++) {
                    auto beta = betas[i];

                    for (int ig = 0; ig < pw_coeffs(is).num_rows_loc(); ig++) {
                        auto x = phi.pw_coeffs(is).prime(ig, i);
                        auto y = pw_coeffs(is).prime(ig, i);
                        
                        pw_coeffs(is).prime(ig, i) = x + beta * y;
                    }
                    if (has_mt()) {
                        for (int j = 0; j < mt_coeffs(is).num_rows_loc(); j++) {
                            auto x = phi.mt_coeffs(is).prime(j, i);
                            auto y = mt_coeffs(is).prime(j, i);
                            mt_coeffs(is).prime(j, i) = x + beta * y;
                        }
                    }
                }
                break;
            }
            case device_t::GPU: {
#if defined(SIRIUS_GPU)
                if (!betas.on_device()) {
                    betas.allocate(memory_t::device);
                }

                betas.copy_to(memory_t::device, 0, n__);

                wf_xpby_gpu(
                    phi.pw_coeffs(is).prime().at(memory_t::device),
                    betas.at(memory_t::device),
                    pw_coeffs(is).prime().at(memory_t::device),
                    pw_coeffs(is).num_rows_loc(),
                    n__);

                if (has_mt()) {
                    wf_xpby_gpu(
                        phi.mt_coeffs(is).prime().at(memory_t::device),
                        betas.at(memory_t::device),
                        mt_coeffs(is).prime().at(memory_t::device),
                        mt_coeffs(is).num_rows_loc(),
                        n__);
                }
#endif
            }
        }
    }
}

template<class Ta>
void
Wave_functions::axpy(device_t pu__, spin_range spins__, sddk::mdarray<Ta,1> &alphas, Wave_functions const &phi, int n__)
{
    for (int is : spins__) {
        switch (pu__) {
            case device_t::CPU: {
                #pragma omp parallel for
                for (int i = 0; i < n__; i++) {
                    auto alpha = alphas[i];

                    for (int ig = 0; ig < pw_coeffs(is).num_rows_loc(); ig++) {
                        auto x = phi.pw_coeffs(is).prime(ig, i);
                        auto y = pw_coeffs(is).prime(ig, i);
                        
                        pw_coeffs(is).prime(ig, i) = alpha * x + y;
                    }
                    if (has_mt()) {
                        for (int j = 0; j < mt_coeffs(is).num_rows_loc(); j++) {
                            auto x = phi.mt_coeffs(is).prime(j, i);
                            auto y = mt_coeffs(is).prime(j, i);
                            mt_coeffs(is).prime(j, i) = alpha * x + y;
                        }
                    }
                }
                break;
            }
            case device_t::GPU: {
#if defined(SIRIUS_GPU)
                if (!alphas.on_device()) {
                    alphas.allocate(memory_t::device);
                }

                alphas.copy_to(memory_t::device, 0, n__);

                wf_axpy_gpu(
                    alphas.at(memory_t::device),
                    phi.pw_coeffs(is).prime().at(memory_t::device),
                    pw_coeffs(is).prime().at(memory_t::device),
                    pw_coeffs(is).num_rows_loc(),
                    n__);

                if (has_mt()) {
                    wf_xpby_gpu(
                        alphas.at(memory_t::device),
                        phi.mt_coeffs(is).prime().at(memory_t::device),
                        mt_coeffs(is).prime().at(memory_t::device),
                        mt_coeffs(is).num_rows_loc(),
                        n__);
                }
#endif
            }
        }
    }
}

template<class Ta>
void
Wave_functions::axpy_scatter(device_t pu__, spin_range spins__, sddk::mdarray<Ta,1> &alphas, Wave_functions const &phi, sddk::mdarray<int,1> &ids, int n__)
{
    for (int is : spins__) {
        switch (pu__) {
            case device_t::CPU: {
                #pragma omp parallel for
                for (int i = 0; i < n__; i++) {
                    auto ii = ids[i];
                    auto alpha = alphas[i];

                    for (int ig = 0; ig < pw_coeffs(is).num_rows_loc(); ig++) {
                        pw_coeffs(is).prime(ig, ii) += alpha * phi.pw_coeffs(is).prime(ig, i);
                    }
                    if (has_mt()) {
                        for (int j = 0; j < mt_coeffs(is).num_rows_loc(); j++) {
                            mt_coeffs(is).prime(j, ii) += alpha * phi.mt_coeffs(is).prime(j, i);
                        }
                    }
                }
                break;
            }
            case device_t::GPU: {
#if defined(SIRIUS_GPU)
                if (!alphas.on_device()) {
                    alphas.allocate(memory_t::device);
                }
                if (!ids.on_device()) {
                    ids.allocate(memory_t::device);
                }

                alphas.copy_to(memory_t::device, 0, n__);
                ids.copy_to(memory_t::device, 0, n__);

                wf_axpy_scatter_gpu(
                    alphas.at(memory_t::device),
                    phi.pw_coeffs(is).prime().at(memory_t::device),
                    pw_coeffs(is).prime().at(memory_t::device),
                    ids.at(memory_t::device),
                    pw_coeffs(is).num_rows_loc(),
                    n__);

                if (has_mt()) {
                    wf_axpy_scatter_gpu(
                        alphas.at(memory_t::device),
                        phi.mt_coeffs(is).prime().at(memory_t::device),
                        mt_coeffs(is).prime().at(memory_t::device),
                        ids.at(memory_t::device),
                        mt_coeffs(is).num_rows_loc(),
                        n__);
                }
#endif
            }
        }
    }
}

mdarray<double, 1>
Wave_functions::sumsqr(device_t pu__, spin_range spins__, int n__) const
{
    mdarray<double, 1> s(n__, memory_t::host, "sumsqr");
    s.zero();
    if (pu__ == device_t::GPU) {
        s.allocate(memory_t::device).zero(memory_t::device);
    }

    for (int is : spins__) {
        switch (pu__) {
            case device_t::CPU: {
                #pragma omp parallel for
                for (int i = 0; i < n__; i++) {
                    for (int ig = 0; ig < pw_coeffs(is).num_rows_loc(); ig++) {
                        s[i] += (std::pow(pw_coeffs(is).prime(ig, i).real(), 2) +
                                 std::pow(pw_coeffs(is).prime(ig, i).imag(), 2));
                    }
                    if (gkvecp_.gvec().reduced()) {
                        if (comm_.rank() == 0) {
                            s[i] = 2 * s[i] - std::pow(pw_coeffs(is).prime(0, i).real(), 2);
                        } else {
                            s[i] *= 2;
                        }
                    }
                    if (has_mt()) {
                        for (int j = 0; j < mt_coeffs(is).num_rows_loc(); j++) {
                            s[i] += (std::pow(mt_coeffs(is).prime(j, i).real(), 2) +
                                     std::pow(mt_coeffs(is).prime(j, i).imag(), 2));
                        }
                    }
                }
                break;
            }
            case device_t::GPU: {
#if defined(SIRIUS_GPU)
                add_square_sum_gpu(pw_coeffs(is).prime().at(memory_t::device), pw_coeffs(is).num_rows_loc(), n__,
                                   gkvecp_.gvec().reduced(), comm_.rank(), s.at(memory_t::device));
                if (has_mt()) {
                    add_square_sum_gpu(mt_coeffs(is).prime().at(memory_t::device), mt_coeffs(is).num_rows_loc(), n__, 0,
                                       comm_.rank(), s.at(memory_t::device));
                }
#endif
                break;
            }
        }
    }
    if (pu__ == device_t::GPU) {
        s.copy_to(memory_t::host);
    }
    comm_.allreduce(s.at(memory_t::host), n__);
    return s;
}

// template void Wave_functions::axpby(device_t pu__, spin_range spins__, double alpha, Wave_functions const &phi, double beta, int n__);
// template void Wave_functions::xpby(device_t pu__, spin_range spins__, Wave_functions const &phi, sddk::mdarray<double,1> const &betas, int n__);
// template void Wave_functions::axpy(device_t pu__, spin_range spins__, sddk::mdarray<double,1> const &alphas, Wave_functions const &phi, int n__);
// template void Wave_functions::axpy_scatter(device_t pu__, spin_range spins__, sddk::mdarray<double,1> const &alphas, Wave_functions const &phi, sddk::mdarray<int,1> const &ids, int n__);

template void Wave_functions::axpby(device_t pu__, spin_range spins__, double_complex alpha, Wave_functions const &phi, double_complex beta, int n__);
template void Wave_functions::xpby(device_t pu__, spin_range spins__, Wave_functions const &phi, sddk::mdarray<double_complex,1> &betas, int n__);
template void Wave_functions::axpy(device_t pu__, spin_range spins__, sddk::mdarray<double_complex,1> &alphas, Wave_functions const &phi, int n__);
template void Wave_functions::axpy_scatter(device_t pu__, spin_range spins__, sddk::mdarray<double_complex,1>  &alphas, Wave_functions const &phi, sddk::mdarray<int,1> &ids, int n__);


} // namespace sddk


