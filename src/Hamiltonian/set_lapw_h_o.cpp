//
// Created by mathieut on 7/23/19.
//

// Copyright (c) 2013-2016 Anton Kozhevnikov, Thomas Schulthess
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

/** \file set_lapw_h_o.hpp
 *
 *  \brief Contains functions of LAPW Hamiltonian and overlap setup.
 */


#include "hamiltonian.hpp"

namespace sirius {
//    template<>
//    void
//    Hamiltonian::set_fv_h_o<device_t::CPU, electronic_structure_method_t::full_potential_lapwlo>(K_point *kp__,
//                                                                                                 dmatrix<double_complex> &h__,
//                                                                                                 dmatrix<double_complex> &o__) const {
//        PROFILE("sirius::Hamiltonian::set_fv_h_o");
//
//        h__.zero();
//        o__.zero();
//
//        int num_atoms_in_block = 2 * omp_get_max_threads();
//        int nblk = unit_cell_.num_atoms() / num_atoms_in_block +
//                   std::min(1, unit_cell_.num_atoms() % num_atoms_in_block);
//
//        int max_mt_aw = num_atoms_in_block * unit_cell_.max_mt_aw_basis_size();
//
//        mdarray<double_complex, 2> alm_row(kp__->num_gkvec_row(), max_mt_aw);
//        mdarray<double_complex, 2> alm_col(kp__->num_gkvec_col(), max_mt_aw);
//        mdarray<double_complex, 2> halm_col(kp__->num_gkvec_col(), max_mt_aw);
//        mdarray<double_complex, 2> oalm_col;
//        if (ctx_.valence_relativity() == relativity_t::iora) {
//            oalm_col = mdarray<double_complex, 2>(kp__->num_gkvec_col(), max_mt_aw);
//        } else {
//            oalm_col = mdarray<double_complex, 2>(alm_col.at(memory_t::host), kp__->num_gkvec_col(), max_mt_aw);
//        }
//
//        utils::timer t1("sirius::Hamiltonian::set_fv_h_o|zgemm");
//        /* loop over blocks of atoms */
//        for (int iblk = 0; iblk < nblk; iblk++) {
//            /* number of matching AW coefficients in the block */
//            int num_mt_aw{0};
//            /* offsets for matching coefficients of individual atoms in the AW block */
//            std::vector<int> offsets(num_atoms_in_block);
//            for (int ia = iblk * num_atoms_in_block;
//                 ia < std::min(unit_cell_.num_atoms(), (iblk + 1) * num_atoms_in_block); ia++) {
//                offsets[ia - iblk * num_atoms_in_block] = num_mt_aw;
//                num_mt_aw += unit_cell_.atom(ia).type().mt_aw_basis_size();
//            }
//
//            if (ctx_.control().print_checksum_) {
//                alm_row.zero();
//                alm_col.zero();
//                halm_col.zero();
//            }
//
//            #pragma omp parallel
//            {
//                int tid = omp_get_thread_num();
//                for (int ia = iblk * num_atoms_in_block;
//                     ia < std::min(unit_cell_.num_atoms(), (iblk + 1) * num_atoms_in_block); ia++) {
//                    if (ia % omp_get_num_threads() == tid) {
//                        int ialoc = ia - iblk * num_atoms_in_block;
//                        auto &atom = unit_cell_.atom(ia);
//                        auto &type = atom.type();
//                        int naw = type.mt_aw_basis_size();
//
//                        mdarray<double_complex, 2> alm_row_tmp(alm_row.at(memory_t::host, 0, offsets[ialoc]),
//                                                               kp__->num_gkvec_row(), naw);
//                        mdarray<double_complex, 2> alm_col_tmp(alm_col.at(memory_t::host, 0, offsets[ialoc]),
//                                                               kp__->num_gkvec_col(), naw);
//                        mdarray<double_complex, 2> halm_col_tmp(halm_col.at(memory_t::host, 0, offsets[ialoc]),
//                                                                kp__->num_gkvec_col(), naw);
//                        mdarray<double_complex, 2> oalm_col_tmp;
//                        if (ctx_.valence_relativity() == relativity_t::iora) {
//                            oalm_col_tmp = mdarray<double_complex, 2>(oalm_col.at(memory_t::host, 0, offsets[ialoc]),
//                                                                      kp__->num_gkvec_col(), naw);
//                        } else {
//                            oalm_col_tmp = mdarray<double_complex, 2>(alm_col.at(memory_t::host, 0, offsets[ialoc]),
//                                                                      kp__->num_gkvec_col(), naw);
//                        }
//
//                        kp__->alm_coeffs_row().generate<true>(atom, alm_row_tmp);
//                        //for (int xi = 0; xi < naw; xi++) {
//                        //    for (int igk = 0; igk < kp__->num_gkvec_row(); igk++) {
//                        //        alm_row_tmp(igk, xi) = std::conj(alm_row_tmp(igk, xi));
//                        //    }
//                        //}
//                        kp__->alm_coeffs_col().generate<false>(atom, alm_col_tmp);
//                        apply_hmt_to_apw<spin_block_t::nm>(atom, kp__->num_gkvec_col(), alm_col_tmp, halm_col_tmp);
//
//                        if (ctx_.valence_relativity() == relativity_t::iora) {
//                            alm_col_tmp >> oalm_col_tmp;
//                            apply_o1mt_to_apw(atom, kp__->num_gkvec_col(), alm_col_tmp, oalm_col_tmp);
//                        }
//
//                        /* setup apw-lo and lo-apw blocks */
//                        set_fv_h_o_apw_lo(kp__, type, atom, ia, alm_row_tmp, alm_col_tmp, h__, o__);
//                    }
//                }
//            }
//            if (ctx_.control().print_checksum_) {
//                double_complex z1 = alm_row.checksum();
//                double_complex z2 = alm_col.checksum();
//                double_complex z3 = halm_col.checksum();
//                utils::print_checksum("alm_row", z1);
//                utils::print_checksum("alm_col", z2);
//                utils::print_checksum("halm_col", z3);
//            }
//            linalg<device_t::CPU>::gemm(0, 1, kp__->num_gkvec_row(), kp__->num_gkvec_col(), num_mt_aw,
//                                        linalg_const<double_complex>::one(),
//                                        alm_row.at(memory_t::host), alm_row.ld(),
//                                        oalm_col.at(memory_t::host), oalm_col.ld(),
//                                        linalg_const<double_complex>::one(),
//                                        o__.at(memory_t::host), o__.ld());
//
//            linalg<device_t::CPU>::gemm(0, 1, kp__->num_gkvec_row(), kp__->num_gkvec_col(), num_mt_aw,
//                                        linalg_const<double_complex>::one(),
//                                        alm_row.at(memory_t::host), alm_row.ld(),
//                                        halm_col.at(memory_t::host), halm_col.ld(),
//                                        linalg_const<double_complex>::one(),
//                                        h__.at(memory_t::host), h__.ld());
//        }
//        double tval = t1.stop();
//        if (kp__->comm().rank() == 0 && ctx_.control().print_performance_) {
//            printf("effective zgemm performance: %12.6f GFlops",
//                   2 * 8e-9 * kp__->num_gkvec() * kp__->num_gkvec() * unit_cell_.mt_aw_basis_size() / tval);
//        }
//
//        /* add interstitial contributon */
//        this->set_fv_h_o_it(kp__, h__, o__);
//
//        /* setup lo-lo block */
//        this->set_fv_h_o_lo_lo(kp__, h__, o__);
//    }

#ifdef __GPU
    template<>
     void Hamiltonian::set_fv_h_o<device_t::GPU, electronic_structure_method_t::full_potential_lapwlo>(K_point* kp__,
                                                                                                           dmatrix<double_complex>& h__,
                                                                                                         dmatrix<double_complex>& o__) const
    {
        PROFILE("sirius::Hamiltonian::set_fv_h_o");

        utils::timer t2("sirius::Hamiltonian::set_fv_h_o|alloc");
        h__.allocate(memory_t::device);
        h__.zero(memory_t::host);
        h__.zero(memory_t::device);

        o__.allocate(memory_t::device);
        o__.zero(memory_t::host);
        o__.zero(memory_t::device);

        int num_atoms_in_block = 2 * omp_get_max_threads();
        int nblk = unit_cell_.num_atoms() / num_atoms_in_block +
                   std::min(1, unit_cell_.num_atoms() % num_atoms_in_block);

        int max_mt_aw = num_atoms_in_block * unit_cell_.max_mt_aw_basis_size();

        mdarray<double_complex, 3> alm_row(kp__->num_gkvec_row(), max_mt_aw, 2, memory_t::host_pinned);
        alm_row.allocate(memory_t::device);

        mdarray<double_complex, 3> alm_col(kp__->num_gkvec_col(), max_mt_aw, 2, memory_t::host_pinned);
        alm_col.allocate(memory_t::device);

        mdarray<double_complex, 3> halm_col(kp__->num_gkvec_col(), max_mt_aw, 2, memory_t::host_pinned);
        halm_col.allocate(memory_t::device);
        t2.stop();

        if (ctx_.comm().rank() == 0 && ctx_.control().print_memory_usage_) {
            MEMORY_USAGE_INFO();
        }

        utils::timer t1("sirius::Hamiltonian::set_fv_h_o|zgemm");
        for (int iblk = 0; iblk < nblk; iblk++) {
            int num_mt_aw = 0;
            std::vector<int> offsets(num_atoms_in_block);
            for (int ia = iblk * num_atoms_in_block; ia < std::min(unit_cell_.num_atoms(), (iblk + 1) * num_atoms_in_block); ia++) {
                int ialoc = ia - iblk * num_atoms_in_block;
                auto& atom = unit_cell_.atom(ia);
                auto& type = atom.type();
                offsets[ialoc] = num_mt_aw;
                num_mt_aw += type.mt_aw_basis_size();
            }

            int s = iblk % 2;

#pragma omp parallel
            {
                int tid = omp_get_thread_num();
                for (int ia = iblk * num_atoms_in_block; ia < std::min(unit_cell_.num_atoms(), (iblk + 1) * num_atoms_in_block); ia++) {
                    if (ia % omp_get_num_threads() == tid) {
                        int ialoc = ia - iblk * num_atoms_in_block;
                        auto& atom = unit_cell_.atom(ia);
                        auto& type = atom.type();

                        mdarray<double_complex, 2> alm_row_tmp(alm_row.at(memory_t::host, 0, offsets[ialoc], s),
                                                               alm_row.at(memory_t::device, 0, offsets[ialoc], s),
                                                               kp__->num_gkvec_row(), type.mt_aw_basis_size());

                        mdarray<double_complex, 2> alm_col_tmp(alm_col.at(memory_t::host, 0, offsets[ialoc], s),
                                                               alm_col.at(memory_t::device, 0, offsets[ialoc], s),
                                                               kp__->num_gkvec_col(), type.mt_aw_basis_size());

                        mdarray<double_complex, 2> halm_col_tmp(halm_col.at(memory_t::host, 0, offsets[ialoc], s),
                                                                halm_col.at(memory_t::device, 0, offsets[ialoc], s),
                                                                kp__->num_gkvec_col(), type.mt_aw_basis_size());

                        kp__->alm_coeffs_row().generate(ia, alm_row_tmp);
                        for (int xi = 0; xi < type.mt_aw_basis_size(); xi++) {
                            for (int igk = 0; igk < kp__->num_gkvec_row(); igk++) {
                                alm_row_tmp(igk, xi) = std::conj(alm_row_tmp(igk, xi));
                            }
                        }
                        alm_row_tmp.copy_to(memory_t::device, stream_id(tid));

                        kp__->alm_coeffs_col().generate(ia, alm_col_tmp);
                        alm_col_tmp.copy_to(memory_t::device, stream_id(tid));

                        apply_hmt_to_apw<spin_block_t::nm>(atom, kp__->num_gkvec_col(), alm_col_tmp, halm_col_tmp);
                        halm_col_tmp.copy_to(memory_t::device, stream_id(tid));

                        /* setup apw-lo and lo-apw blocks */
                        set_fv_h_o_apw_lo(kp__, type, atom, ia, alm_row_tmp, alm_col_tmp, h__, o__);
                    }
                }
                acc::sync_stream(stream_id(tid));
            }
            acc::sync_stream(stream_id(omp_get_max_threads()));
            linalg<device_t::GPU>::gemm(0, 1, kp__->num_gkvec_row(), kp__->num_gkvec_col(), num_mt_aw, &linalg_const<double_complex>::one(),
                                        alm_row.at(memory_t::device, 0, 0, s), alm_row.ld(), alm_col.at(memory_t::device, 0, 0, s), alm_col.ld(), &linalg_const<double_complex>::one(),
                                        o__.at(memory_t::device), o__.ld(), omp_get_max_threads());

            linalg<device_t::GPU>::gemm(0, 1, kp__->num_gkvec_row(), kp__->num_gkvec_col(), num_mt_aw, &linalg_const<double_complex>::one(),
                                        alm_row.at(memory_t::device, 0, 0, s), alm_row.ld(), halm_col.at(memory_t::device, 0, 0, s), halm_col.ld(), &linalg_const<double_complex>::one(),
                                        h__.at(memory_t::device), h__.ld(), omp_get_max_threads());
        }

        acc::copyout(h__.at(memory_t::host), h__.ld(), h__.at(memory_t::device), h__.ld(), kp__->num_gkvec_row(), kp__->num_gkvec_col());
        acc::copyout(o__.at(memory_t::host), o__.ld(), o__.at(memory_t::device), o__.ld(), kp__->num_gkvec_row(), kp__->num_gkvec_col());

        // double tval = t1.stop();
        //if (kp__->comm().rank() == 0 && ctx_.control().print_performance_) {
        //    DUMP("effective zgemm performance: %12.6f GFlops",
        //         2 * 8e-9 * kp__->num_gkvec() * kp__->num_gkvec() * unit_cell_.mt_aw_basis_size() / tval);
        //}

        /* add interstitial contributon */
        set_fv_h_o_it(kp__, h__, o__);

        /* setup lo-lo block */
        set_fv_h_o_lo_lo(kp__, h__, o__);

        h__.deallocate(memory_t::device);
        o__.deallocate(memory_t::device);
    }
#endif

    void Hamiltonian::set_fv_h_o_apw_lo(K_point *kp,
                                        Atom_type const &type,
                                        Atom const &atom,
                                        int ia,
                                        mdarray<double_complex, 2> &alm_row, // alm_row comes conjugated
                                        mdarray<double_complex, 2> &alm_col,
                                        mdarray<double_complex, 2> &h,
                                        mdarray<double_complex, 2> &o) const {
        /* apw-lo block */
        for (int i = 0; i < kp->num_atom_lo_cols(ia); i++) {
            int icol = kp->lo_col(ia, i);
            /* local orbital indices */
            int l = kp->lo_basis_descriptor_col(icol).l;
            int lm = kp->lo_basis_descriptor_col(icol).lm;
            int idxrf = kp->lo_basis_descriptor_col(icol).idxrf;
            int order = kp->lo_basis_descriptor_col(icol).order;
            /* loop over apw components */
            for (int j1 = 0; j1 < type.mt_aw_basis_size(); j1++) {
                int lm1 = type.indexb(j1).lm;
                int idxrf1 = type.indexb(j1).idxrf;

                double_complex zsum = atom.radial_integrals_sum_L3<spin_block_t::nm>(idxrf, idxrf1,
                                                                                     gaunt_coefs_->gaunt_vector(lm1,
                                                                                                                lm));

                if (std::abs(zsum) > 1e-14) {
                    for (int igkloc = 0; igkloc < kp->num_gkvec_row(); igkloc++) {
                        h(igkloc, kp->num_gkvec_col() + icol) += zsum * alm_row(igkloc, j1);
                    }
                }
            }

            for (int order1 = 0; order1 < type.aw_order(l); order1++) {
                int xi1 = type.indexb().index_by_lm_order(lm, order1);
                for (int igkloc = 0; igkloc < kp->num_gkvec_row(); igkloc++) {
                    o(igkloc, kp->num_gkvec_col() + icol) += atom.symmetry_class().o_radial_integral(l, order1, order) *
                                                             alm_row(igkloc, xi1);
                }
                if (ctx_.valence_relativity() == relativity_t::iora) {
                    int idxrf1 = type.indexr().index_by_l_order(l, order1);
                    for (int igkloc = 0; igkloc < kp->num_gkvec_row(); igkloc++) {
                        o(igkloc, kp->num_gkvec_col() + icol) +=
                                atom.symmetry_class().o1_radial_integral(idxrf1, idxrf) *
                                alm_row(igkloc, xi1);
                    }
                }
            }
        }

        std::vector<double_complex> ztmp(kp->num_gkvec_col());
        /* lo-apw block */
        for (int i = 0; i < kp->num_atom_lo_rows(ia); i++) {
            int irow = kp->lo_row(ia, i);
            /* local orbital indices */
            int l = kp->lo_basis_descriptor_row(irow).l;
            int lm = kp->lo_basis_descriptor_row(irow).lm;
            int idxrf = kp->lo_basis_descriptor_row(irow).idxrf;
            int order = kp->lo_basis_descriptor_row(irow).order;

            std::fill(ztmp.begin(), ztmp.end(), 0);

            //std::memset(&ztmp[0], 0, kp->num_gkvec_col() * sizeof(double_complex));
            /* loop over apw components */
            for (int j1 = 0; j1 < type.mt_aw_basis_size(); j1++) {
                int lm1 = type.indexb(j1).lm;
                int idxrf1 = type.indexb(j1).idxrf;

                double_complex zsum = atom.radial_integrals_sum_L3<spin_block_t::nm>(idxrf1, idxrf,
                                                                                     gaunt_coefs_->gaunt_vector(lm,
                                                                                                                lm1));

                if (std::abs(zsum) > 1e-14) {
                    for (int igkloc = 0; igkloc < kp->num_gkvec_col(); igkloc++) {
                        ztmp[igkloc] += zsum * alm_col(igkloc, j1);
                    }
                }
            }

            for (int igkloc = 0; igkloc < kp->num_gkvec_col(); igkloc++) {
                h(irow + kp->num_gkvec_row(), igkloc) += ztmp[igkloc];
            }

            for (int order1 = 0; order1 < type.aw_order(l); order1++) {
                int xi1 = type.indexb().index_by_lm_order(lm, order1);
                for (int igkloc = 0; igkloc < kp->num_gkvec_col(); igkloc++) {
                    o(irow + kp->num_gkvec_row(), igkloc) += atom.symmetry_class().o_radial_integral(l, order, order1) *
                                                             alm_col(igkloc, xi1);
                }
                if (ctx_.valence_relativity() == relativity_t::iora) {
                    int idxrf1 = type.indexr().index_by_l_order(l, order1);
                    for (int igkloc = 0; igkloc < kp->num_gkvec_col(); igkloc++) {
                        o(irow + kp->num_gkvec_row(), igkloc) +=
                                atom.symmetry_class().o1_radial_integral(idxrf, idxrf1) *
                                alm_col(igkloc, xi1);
                    }
                }
            }
        }
    }

    void Hamiltonian::set_fv_h_o_it(K_point *kp,
                                    mdarray<double_complex, 2> &h,
                                    mdarray<double_complex, 2> &o) const {
        PROFILE("sirius::Hamiltonian::set_fv_h_o_it");

        //#ifdef __PRINT_OBJECT_CHECKSUM
        //double_complex z1 = mdarray<double_complex, 1>(&effective_potential->f_pw(0), ctx_.gvec().num_gvec()).checksum();
        //DUMP("checksum(veff_pw): %18.10f %18.10f", std::real(z1), std::imag(z1));
        //#endif

        double sq_alpha_half = 0.5 * std::pow(speed_of_light, -2);

#pragma omp parallel for default(shared)
        for (int igk_col = 0; igk_col < kp->num_gkvec_col(); igk_col++) {
            int ig_col = kp->igk_col(igk_col);
            auto gvec_col = kp->gkvec().gvec(ig_col);
            auto gkvec_col_cart = kp->gkvec().gkvec_cart<index_domain_t::global>(ig_col);
            for (int igk_row = 0; igk_row < kp->num_gkvec_row(); igk_row++) {
                int ig_row = kp->igk_row(igk_row);
                auto gvec_row = kp->gkvec().gvec(ig_row);
                auto gkvec_row_cart = kp->gkvec().gkvec_cart<index_domain_t::global>(ig_row);
                int ig12 = ctx_.gvec().index_g12(gvec_row, gvec_col);
                /* pw kinetic energy */
                double t1 = 0.5 * dot(gkvec_row_cart, gkvec_col_cart);

                h(igk_row, igk_col) += this->potential().veff_pw(ig12);
                o(igk_row, igk_col) += ctx_.theta_pw(ig12);

                if (ctx_.valence_relativity() == relativity_t::none) {
                    h(igk_row, igk_col) += t1 * ctx_.theta_pw(ig12);
                }
                if (ctx_.valence_relativity() == relativity_t::zora) {
                    h(igk_row, igk_col) += t1 * this->potential().rm_inv_pw(ig12);
                }
                if (ctx_.valence_relativity() == relativity_t::iora) {
                    h(igk_row, igk_col) += t1 * this->potential().rm_inv_pw(ig12);
                    o(igk_row, igk_col) += t1 * sq_alpha_half * this->potential().rm2_inv_pw(ig12);
                }
            }
        }
    }

    void Hamiltonian::set_fv_h_o_lo_lo(K_point *kp,
                                       mdarray<double_complex, 2> &h,
                                       mdarray<double_complex, 2> &o) const {
        PROFILE("sirius::Hamiltonian::set_fv_h_o_lo_lo");

        /* lo-lo block */
#pragma omp parallel for default(shared)
        for (int icol = 0; icol < kp->num_lo_col(); icol++) {
            int ia = kp->lo_basis_descriptor_col(icol).ia;
            int lm2 = kp->lo_basis_descriptor_col(icol).lm;
            int idxrf2 = kp->lo_basis_descriptor_col(icol).idxrf;

            for (int irow = 0; irow < kp->num_lo_row(); irow++) {
                /* lo-lo block is diagonal in atom index */
                if (ia == kp->lo_basis_descriptor_row(irow).ia) {
                    auto &atom = unit_cell_.atom(ia);
                    int lm1 = kp->lo_basis_descriptor_row(irow).lm;
                    int idxrf1 = kp->lo_basis_descriptor_row(irow).idxrf;

                    h(kp->num_gkvec_row() + irow, kp->num_gkvec_col() + icol) +=
                            atom.radial_integrals_sum_L3<spin_block_t::nm>(idxrf1, idxrf2,
                                                                           gaunt_coefs_->gaunt_vector(lm1, lm2));

                    if (lm1 == lm2) {
                        int l = kp->lo_basis_descriptor_row(irow).l;
                        int order1 = kp->lo_basis_descriptor_row(irow).order;
                        int order2 = kp->lo_basis_descriptor_col(icol).order;
                        o(kp->num_gkvec_row() + irow, kp->num_gkvec_col() + icol) +=
                                atom.symmetry_class().o_radial_integral(l, order1, order2);
                        if (ctx_.valence_relativity() == relativity_t::iora) {
                            int idxrf1 = atom.type().indexr().index_by_l_order(l, order1);
                            int idxrf2 = atom.type().indexr().index_by_l_order(l, order2);
                            o(kp->num_gkvec_row() + irow, kp->num_gkvec_col() + icol) +=
                                    atom.symmetry_class().o1_radial_integral(idxrf1, idxrf2);
                        }
                    }
                }
            }
        }
    }


void Hamiltonian_k::set_fv_h_o(dmatrix<double_complex> &h__, dmatrix<double_complex> &o__) const 
{
    PROFILE("sirius::Hamiltonian_k::set_fv_h_o");

    h__.zero();
    o__.zero();

    auto& uc = H0_.ctx().unit_cell();

    auto& kp = this->kp();

    int num_atoms_in_block = 2 * omp_get_max_threads();
    int nblk = uc.num_atoms() / num_atoms_in_block + std::min(1, uc.num_atoms() % num_atoms_in_block);

    int max_mt_aw = num_atoms_in_block * uc.max_mt_aw_basis_size();

    mdarray<double_complex, 2> alm_row(kp.num_gkvec_row(), max_mt_aw);
    mdarray<double_complex, 2> alm_col(kp.num_gkvec_col(), max_mt_aw);
    mdarray<double_complex, 2> halm_col(kp.num_gkvec_col(), max_mt_aw);

    /* offsets for matching coefficients of individual atoms in the AW block */
    std::vector<int> offsets(uc.num_atoms());

    utils::timer t1("sirius::Hamiltonian::set_fv_h_o|zgemm");
    /* loop over blocks of atoms */
    for (int iblk = 0; iblk < nblk; iblk++) {
        /* number of matching AW coefficients in the block */
        int num_mt_aw{0};
        int ia_begin = iblk * num_atoms_in_block;
        int ia_end = std::min(uc.num_atoms(), (iblk + 1) * num_atoms_in_block);
        for (int ia = ia_begin; ia < ia_end; ia++) {
            offsets[ia] = num_mt_aw;
            num_mt_aw += uc.atom(ia).type().mt_aw_basis_size();
        }

        if (H0_.ctx().control().print_checksum_) {
            alm_row.zero();
            alm_col.zero();
            halm_col.zero();
        }

        #pragma omp parallel
        {
            //int tid = omp_get_thread_num();
            #pragma omp for
            for (int ia = ia_begin; ia < ia_end; ia++) {
                auto& atom = uc.atom(ia);
                auto& type = atom.type();
                int naw = type.mt_aw_basis_size();

                mdarray<double_complex, 2> alm_row_atom;
                alm_row_atom = mdarray<double_complex, 2>(alm_row.at(memory_t::host, 0, offsets[ia]),
                                                          kp.num_gkvec_row(), naw);

                mdarray<double_complex, 2> alm_col_atom;
                alm_col_atom = mdarray<double_complex, 2>(alm_col.at(memory_t::host, 0, offsets[ia]),
                                                          kp.num_gkvec_col(), naw);

                mdarray<double_complex, 2> halm_col_atom;
                halm_col_atom = mdarray<double_complex, 2>(halm_col.at(memory_t::host, 0, offsets[ia]),
                                                           kp.num_gkvec_col(), naw);

                kp.alm_coeffs_row().generate<true>(atom, alm_row_atom);
                kp.alm_coeffs_col().generate<false>(atom, alm_col_atom);

                H0_.apply_hmt_to_apw<spin_block_t::nm>(atom, kp.num_gkvec_col(), alm_col_atom, halm_col_atom);

                /* setup apw-lo and lo-apw blocks */
                set_fv_h_o_apw_lo(atom, ia, alm_row_atom, alm_col_atom, h__, o__);

                /* finally, modify alm coefficients for iora */
                if (H0_.ctx().valence_relativity() == relativity_t::iora) {
                    H0_.add_o1mt_to_apw(atom, kp.num_gkvec_col(), alm_col_atom);
                }
            }
        }
        if (H0_.ctx().control().print_checksum_) {
            double_complex z1 = alm_row.checksum();
            double_complex z2 = alm_col.checksum();
            double_complex z3 = halm_col.checksum();
            utils::print_checksum("alm_row", z1);
            utils::print_checksum("alm_col", z2);
            utils::print_checksum("halm_col", z3);
        }
        linalg<device_t::CPU>::gemm(0, 1, kp.num_gkvec_row(), kp.num_gkvec_col(), num_mt_aw,
                                    linalg_const<double_complex>::one(),
                                    alm_row.at(memory_t::host), alm_row.ld(),
                                    alm_col.at(memory_t::host), alm_col.ld(),
                                    linalg_const<double_complex>::one(),
                                    o__.at(memory_t::host), o__.ld());

        linalg<device_t::CPU>::gemm(0, 1, kp.num_gkvec_row(), kp.num_gkvec_col(), num_mt_aw,
                                    linalg_const<double_complex>::one(),
                                    alm_row.at(memory_t::host), alm_row.ld(),
                                    halm_col.at(memory_t::host), halm_col.ld(),
                                    linalg_const<double_complex>::one(),
                                    h__.at(memory_t::host), h__.ld());
    }
    double tval = t1.stop();
    if (kp.comm().rank() == 0 && H0_.ctx().control().print_performance_) {
        kp.message(1, __func__, "effective zgemm performance: %12.6f GFlops",
               2 * 8e-9 * kp.num_gkvec() * kp.num_gkvec() * uc.mt_aw_basis_size() / tval);
    }

    /* add interstitial contributon */
    set_fv_h_o_it(h__, o__);

    /* setup lo-lo block */
    set_fv_h_o_lo_lo(h__, o__);
}

/* alm_row comes in already conjugated */
void Hamiltonian_k::set_fv_h_o_apw_lo(Atom const& atom__, int ia__, mdarray<double_complex, 2>& alm_row__,
                                      mdarray<double_complex, 2>& alm_col__, mdarray<double_complex, 2>& h__,
                                      mdarray<double_complex, 2>& o__) const
{
    auto& type = atom__.type();
    /* apw-lo block */
    for (int i = 0; i < kp().num_atom_lo_cols(ia__); i++) {
        int icol = kp().lo_col(ia__, i);
        /* local orbital indices */
        int l = kp().lo_basis_descriptor_col(icol).l;
        int lm = kp().lo_basis_descriptor_col(icol).lm;
        int idxrf = kp().lo_basis_descriptor_col(icol).idxrf;
        int order = kp().lo_basis_descriptor_col(icol).order;
        /* loop over apw components and update H */
        for (int j1 = 0; j1 < type.mt_aw_basis_size(); j1++) {
            int lm1 = type.indexb(j1).lm;
            int idxrf1 = type.indexb(j1).idxrf;

            auto zsum = atom__.radial_integrals_sum_L3<spin_block_t::nm>(idxrf, idxrf1,
                H0_.gaunt_coefs().gaunt_vector(lm1, lm));

            if (std::abs(zsum) > 1e-14) {
                for (int igkloc = 0; igkloc < kp().num_gkvec_row(); igkloc++) {
                    h__(igkloc, kp().num_gkvec_col() + icol) += zsum * alm_row__(igkloc, j1);
                }
            }
        }
        /* update O */
        for (int order1 = 0; order1 < type.aw_order(l); order1++) {
            int xi1 = type.indexb().index_by_lm_order(lm, order1);
            double ori = atom__.symmetry_class().o_radial_integral(l, order1, order);
            if (H0_.ctx().valence_relativity() == relativity_t::iora) {
                int idxrf1 = type.indexr().index_by_l_order(l, order1);
                ori += atom__.symmetry_class().o1_radial_integral(idxrf1, idxrf);
            }

            for (int igkloc = 0; igkloc < kp().num_gkvec_row(); igkloc++) {
                o__(igkloc, kp().num_gkvec_col() + icol) += ori * alm_row__(igkloc, xi1);
            }
        }
    }

    std::vector<double_complex> ztmp(kp().num_gkvec_col());
    /* lo-apw block */
    for (int i = 0; i < kp().num_atom_lo_rows(ia__); i++) {
        int irow = kp().lo_row(ia__, i);
        /* local orbital indices */
        int l = kp().lo_basis_descriptor_row(irow).l;
        int lm = kp().lo_basis_descriptor_row(irow).lm;
        int idxrf = kp().lo_basis_descriptor_row(irow).idxrf;
        int order = kp().lo_basis_descriptor_row(irow).order;

        std::fill(ztmp.begin(), ztmp.end(), 0);

        /* loop over apw components */
        for (int j1 = 0; j1 < type.mt_aw_basis_size(); j1++) {
            int lm1 = type.indexb(j1).lm;
            int idxrf1 = type.indexb(j1).idxrf;

            auto zsum = atom__.radial_integrals_sum_L3<spin_block_t::nm>(idxrf1, idxrf,
                H0_.gaunt_coefs().gaunt_vector(lm, lm1));

            if (std::abs(zsum) > 1e-14) {
                for (int igkloc = 0; igkloc < kp().num_gkvec_col(); igkloc++) {
                    ztmp[igkloc] += zsum * alm_col__(igkloc, j1);
                }
            }
        }

        for (int igkloc = 0; igkloc < kp().num_gkvec_col(); igkloc++) {
            h__(irow + kp().num_gkvec_row(), igkloc) += ztmp[igkloc];
        }

        for (int order1 = 0; order1 < type.aw_order(l); order1++) {
            int xi1 = type.indexb().index_by_lm_order(lm, order1);
            double ori = atom__.symmetry_class().o_radial_integral(l, order, order1);
            if (H0_.ctx().valence_relativity() == relativity_t::iora) {
                int idxrf1 = type.indexr().index_by_l_order(l, order1);
                ori += atom__.symmetry_class().o1_radial_integral(idxrf, idxrf1);
            }

            for (int igkloc = 0; igkloc < kp().num_gkvec_col(); igkloc++) {
                o__(irow + kp().num_gkvec_row(), igkloc) += ori * alm_col__(igkloc, xi1);
            }
        }
    }
}

void Hamiltonian_k::set_fv_h_o_lo_lo(dmatrix<double_complex>& h__, dmatrix<double_complex>& o__) const 
{
    PROFILE("sirius::Hamiltonian_k::set_fv_h_o_lo_lo");

    auto& kp = this->kp();

    /* lo-lo block */
    #pragma omp parallel for default(shared)
    for (int icol = 0; icol < kp.num_lo_col(); icol++) {
        int ia = kp.lo_basis_descriptor_col(icol).ia;
        int lm2 = kp.lo_basis_descriptor_col(icol).lm;
        int idxrf2 = kp.lo_basis_descriptor_col(icol).idxrf;

        for (int irow = 0; irow < kp.num_lo_row(); irow++) {
            /* lo-lo block is diagonal in atom index */
            if (ia == kp.lo_basis_descriptor_row(irow).ia) {
                auto& atom = H0_.ctx().unit_cell().atom(ia);
                int lm1 = kp.lo_basis_descriptor_row(irow).lm;
                int idxrf1 = kp.lo_basis_descriptor_row(irow).idxrf;

                h__(kp.num_gkvec_row() + irow, kp.num_gkvec_col() + icol) +=
                    atom.radial_integrals_sum_L3<spin_block_t::nm>(idxrf1, idxrf2,
                        H0_.gaunt_coefs().gaunt_vector(lm1, lm2));

                if (lm1 == lm2) {
                    int l = kp.lo_basis_descriptor_row(irow).l;
                    int order1 = kp.lo_basis_descriptor_row(irow).order;
                    int order2 = kp.lo_basis_descriptor_col(icol).order;
                    o__(kp.num_gkvec_row() + irow, kp.num_gkvec_col() + icol) +=
                            atom.symmetry_class().o_radial_integral(l, order1, order2);
                    if (H0_.ctx().valence_relativity() == relativity_t::iora) {
                        int idxrf1 = atom.type().indexr().index_by_l_order(l, order1);
                        int idxrf2 = atom.type().indexr().index_by_l_order(l, order2);
                        o__(kp.num_gkvec_row() + irow, kp.num_gkvec_col() + icol) +=
                            atom.symmetry_class().o1_radial_integral(idxrf1, idxrf2);
                    }
                }
            }
        }
    }
}

void Hamiltonian_k::set_fv_h_o_it(dmatrix<double_complex>& h__, dmatrix<double_complex>& o__) const 
{
    PROFILE("sirius::Hamiltonian_k::set_fv_h_o_it");

    double sq_alpha_half = 0.5 * std::pow(speed_of_light, -2);

    auto& kp = this->kp();

    #pragma omp parallel for default(shared)
    for (int igk_col = 0; igk_col < kp.num_gkvec_col(); igk_col++) {
        int ig_col = kp.igk_col(igk_col);
        /* fractional coordinates of G vectors */
        auto gvec_col = kp.gkvec().gvec(ig_col);
        /* Cartesian coordinates of G+k vectors */
        auto gkvec_col_cart = kp.gkvec().gkvec_cart<index_domain_t::global>(ig_col);
        for (int igk_row = 0; igk_row < kp.num_gkvec_row(); igk_row++) {
            int ig_row = kp.igk_row(igk_row);
            auto gvec_row = kp.gkvec().gvec(ig_row);
            auto gkvec_row_cart = kp.gkvec().gkvec_cart<index_domain_t::global>(ig_row);
            int ig12 = H0().ctx().gvec().index_g12(gvec_row, gvec_col);
            /* pw kinetic energy */
            double t1 = 0.5 * geometry3d::dot(gkvec_row_cart, gkvec_col_cart);

            h__(igk_row, igk_col) += H0().potential().veff_pw(ig12);
            o__(igk_row, igk_col) += H0().ctx().theta_pw(ig12);

            if (H0().ctx().valence_relativity() == relativity_t::none) {
                h__(igk_row, igk_col) += t1 * H0().ctx().theta_pw(ig12);
            }
            if (H0().ctx().valence_relativity() == relativity_t::zora) {
                h__(igk_row, igk_col) += t1 * H0().potential().rm_inv_pw(ig12);
            }
            if (H0().ctx().valence_relativity() == relativity_t::iora) {
                h__(igk_row, igk_col) += t1 * H0().potential().rm_inv_pw(ig12);
                o__(igk_row, igk_col) += t1 * sq_alpha_half * H0().potential().rm2_inv_pw(ig12);
            }
        }
    }
}

}
