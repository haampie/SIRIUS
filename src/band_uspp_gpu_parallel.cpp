#include <thread>
#include <atomic>
#include "band.h"

namespace sirius {

#ifdef _GPU_
extern "C" void create_beta_pw_gpu_v2(int num_atoms,
                                      int num_gkvec, 
                                      int* beta_pw_desc,
                                      double_complex* beta_pw_type,
                                      double* gkvec,
                                      double* atom_pos,
                                      double_complex* beta_pw);

void Band::apply_h_o_uspp_gpu_parallel_v2(K_point* kp__,
                                          std::vector<double>& effective_potential__,
                                          std::vector<double>& pw_ekin__,
                                          int N__,
                                          int n__,
                                          dmatrix<double_complex>& phi__,
                                          dmatrix<double_complex>& hphi__,
                                          dmatrix<double_complex>& ophi__)
{
    Timer t("sirius::Band::apply_h_o_uspp_gpu_parallel_v2", _global_timer_);

    splindex<block_cyclic> s0(N__,       kp__->num_ranks_col(), kp__->rank_col(), parameters_.cyclic_block_size());
    splindex<block_cyclic> s1(N__ + n__, kp__->num_ranks_col(), kp__->rank_col(), parameters_.cyclic_block_size());

    /* local number of states to which Hamiltonian has to be applied */
    int nloc = static_cast<int>(s1.local_size() - s0.local_size());

    if (!nloc) return;

    auto uc = parameters_.unit_cell();

    /* apply local part of Hamiltonian */
    apply_h_local_parallel(kp__, effective_potential__, pw_ekin__, N__, n__, phi__, hphi__);

    #ifdef _GPU_
    cublas_set_matrix(kp__->num_gkvec_row(), nloc, sizeof(double_complex), &hphi__(0, s0.local_size()), hphi__.ld(), 
                      hphi__.ptr_device(0, s0.local_size()), hphi__.ld());
    #endif
    
    #if !defined(_GPU_)
    /* set intial ophi */
    memcpy(&ophi__(0, s0.local_size()), &phi__(0, s0.local_size()), kp__->num_gkvec_row() * nloc * sizeof(double_complex));
    #else
    cuda_copy_device_to_device(ophi__.ptr_device(0, s0.local_size()), phi__.ptr_device(0, s0.local_size()), 
                               kp__->num_gkvec_row() * nloc * sizeof(double_complex));
    #endif

    /* offset in the packed array of on-site matrices */
    mdarray<int, 1> packed_mtrx_offset(uc->num_atoms());
    int packed_mtrx_size = 0;
    for (int ia = 0; ia < uc->num_atoms(); ia++)
    {   
        int nbf = uc->atom(ia)->mt_basis_size();
        packed_mtrx_offset(ia) = packed_mtrx_size;
        packed_mtrx_size += nbf * nbf;
    }
    
    /* pack Q and D matrices and send them to GPU */
    mdarray<double_complex, 1> d_mtrx_packed(packed_mtrx_size);
    mdarray<double_complex, 1> q_mtrx_packed(packed_mtrx_size);

    for (int ia = 0; ia < uc->num_atoms(); ia++)
    {
        int nbf = uc->atom(ia)->mt_basis_size();
        for (int xi2 = 0; xi2 < nbf; xi2++)
        {
            for (int xi1 = 0; xi1 < nbf; xi1++)
            {
                d_mtrx_packed(packed_mtrx_offset(ia) + xi2 * nbf + xi1) = uc->atom(ia)->d_mtrx(xi1, xi2);
                q_mtrx_packed(packed_mtrx_offset(ia) + xi2 * nbf + xi1) = uc->atom(ia)->type()->uspp().q_mtrx(xi1, xi2);
            }
        }
    }

    d_mtrx_packed.allocate_on_device();
    d_mtrx_packed.copy_to_device();
    q_mtrx_packed.allocate_on_device();
    q_mtrx_packed.copy_to_device();

    int num_atoms_in_block = 256;
    int num_atom_blocks = uc->num_atoms() / num_atoms_in_block + std::min(1, uc->num_atoms() % num_atoms_in_block);

    if (verbosity_level >= 6 && kp__->comm().rank() == 0)
    {
        printf("num_atom_blocks : %i\n", num_atom_blocks);
    }
    
    splindex<block> atom_blocks(uc->num_atoms(), num_atom_blocks, 0);
    
    auto& beta_pw_t = kp__->beta_pw_t();
    beta_pw_t.allocate_on_device();
    beta_pw_t.copy_to_device();

    /* allocate space for <beta|phi> array */
    int nbf_max = uc->max_mt_basis_size() * num_atoms_in_block;
    mdarray<double_complex, 1> beta_phi_tmp(nbf_max * nloc);
    beta_phi_tmp.allocate_on_device();

    /* result of Q or D multiplied by <beta|phi> */
    mdarray<double_complex, 2> tmp(nullptr, nbf_max, nloc);
    tmp.allocate_on_device();

    /* allocate space for beta-projectors */
    mdarray<double_complex, 2> beta_pw(nullptr, kp__->num_gkvec_row(), nbf_max);
    beta_pw.allocate_on_device();
    
    mdarray<int, 2> beta_pw_desc(3, atom_blocks.local_size(0));
    beta_pw_desc.allocate_on_device();

    mdarray<double, 2> atom_pos(3, atom_blocks.local_size(0));
    atom_pos.allocate_on_device();

    /* copy G+k vectors to device */
    mdarray<double, 2> gkvec_row(3, kp__->num_gkvec_row());
    gkvec_row.allocate_on_device();
    for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++)
    {
        for (int x = 0; x < 3; x++) gkvec_row(x, igk_row) = kp__->gklo_basis_descriptor_row(igk_row).gkvec[x];
    }
    gkvec_row.copy_to_device();

    for (int iab = 0; iab < num_atom_blocks; iab++)
    {
        int nbf_in_block = 0;

        for (int i = 0; i < (int)atom_blocks.local_size(iab); i++)
        {
            int ia = (int)atom_blocks.global_index(i, iab);
            auto type = uc->atom(ia)->type();
            /* atom fractional coordinates */
            for (int x = 0; x < 3; x++) atom_pos(x, i) = uc->atom(ia)->position(x);
            /* number of beta functions for atom */
            beta_pw_desc(0, i) = type->mt_basis_size();
            /* offset in beta_pw */
            beta_pw_desc(1, i) = nbf_in_block;
            /* offset in beta_pw_t */
            beta_pw_desc(2, i) = type->offset_lo();

            nbf_in_block += uc->atom(ia)->mt_basis_size();
        }
        beta_pw_desc.copy_to_device();
        atom_pos.copy_to_device();
        
        create_beta_pw_gpu_v2((int)atom_blocks.local_size(iab),
                              kp__->num_gkvec_row(),
                              beta_pw_desc.ptr_device(),
                              beta_pw_t.ptr_device(),
                              gkvec_row.ptr_device(),
                              atom_pos.ptr_device(),
                              beta_pw.ptr_device());

        /* wrapper for <beta|phi> with required dimensions */
        mdarray<double_complex, 2> beta_phi(beta_phi_tmp.ptr(), nbf_in_block, nloc);
        beta_phi.set_ptr_device(beta_phi_tmp.ptr_device());

        Timer t1("sirius::Band::apply_h_o_uspp_cpu_parallel_v2|beta_phi", _global_timer_);
        /* compute <beta|phi> */
        blas<gpu>::gemm(2, 0, nbf_in_block, nloc, kp__->num_gkvec_row(), 
                        beta_pw.ptr_device(), beta_pw.ld(), 
                        phi__.ptr_device(0, s0.local_size()), phi__.ld(), 
                        beta_phi.ptr_device(), beta_phi.ld());

        // TODO: GPU direct MUST(!!!) work here but it doesn't. Standalone tests work, but 
        // here the allreduce fails with a wrong result and a next crash somewhere in ELPA comm.
        beta_phi.copy_to_host();
        kp__->comm_row().allreduce(beta_phi.ptr(), (int)beta_phi.size());
        beta_phi.copy_to_device();

        double tval = t1.stop();

        if (verbosity_level >= 6 && kp__->comm().rank() == 0)
        {
            printf("<beta|phi> effective zgemm with M, N, K: %6i %6i %6i, %12.4f sec, %12.4f GFlops/node\n",
                   nbf_in_block, nloc, kp__->num_gkvec(),
                   tval, 8e-9 * nbf_in_block * nloc * kp__->num_gkvec() / tval / kp__->num_ranks_row());
        }
       
        #pragma omp parallel for
        for (int i = 0; i < (int)atom_blocks.local_size(iab); i++)
        {
            int ia = (int)atom_blocks.global_index(i, iab);
            int ofs = beta_pw_desc(1, i);
            
            /* number of beta functions for a given atom */
            int nbf = beta_pw_desc(0, i);

            /* compute D*<beta|phi> */
            blas<gpu>::gemm(0, 0, nbf, nloc, nbf, d_mtrx_packed.ptr_device(packed_mtrx_offset(ia)), nbf, 
                            beta_phi.ptr_device(ofs, 0), beta_phi.ld(), tmp.ptr_device(ofs, 0), tmp.ld(), 
                            Platform::thread_id());

        }
        cuda_device_synchronize();
        
        double_complex alpha = complex_one;
        /* compute <G+k|beta> * D*<beta|phi> and add to hphi */
        blas<gpu>::gemm(0, 0, kp__->num_gkvec_row(), nloc, nbf_in_block, &alpha,
                        beta_pw.ptr_device(), beta_pw.ld(), tmp.ptr_device(), tmp.ld(), &alpha, 
                        hphi__.ptr_device(0, s0.local_size()), hphi__.ld());

        #pragma omp parallel for
        for (int i = 0; i < (int)atom_blocks.local_size(iab); i++)
        {
            int ia = (int)atom_blocks.global_index(i, iab);
            int ofs = beta_pw_desc(1, i);
            
            /* number of beta functions for a given atom */
            int nbf = beta_pw_desc(0, i);

            /* compute Q*<beta|phi> */
            blas<gpu>::gemm(0, 0, nbf, nloc, nbf, q_mtrx_packed.ptr_device(packed_mtrx_offset(ia)), nbf,
                            beta_phi.ptr_device(ofs, 0), beta_phi.ld(), tmp.ptr_device(ofs, 0), tmp.ld(), 
                            Platform::thread_id());
        }
        cuda_device_synchronize();

        /* compute <G+k|beta> * Q*<beta|phi> and add to ophi */
        blas<gpu>::gemm(0, 0, kp__->num_gkvec_row(), nloc, nbf_in_block, &alpha,
                        beta_pw.ptr_device(), beta_pw.ld(), tmp.ptr_device(), tmp.ld(), &alpha,
                        ophi__.ptr_device(0, s0.local_size()), ophi__.ld());
    }
    
    kp__->comm().barrier();
}

#ifdef _GPU_DIRECT_
void bcast_column_gpu(K_point* kp__,
                      splindex<block_cyclic>& s0_col__,
                      splindex<block_cyclic>& s1_col__,
                      int icol__,
                      dmatrix<double_complex>& m__,
                      mdarray<double_complex, 3>& m_tmp__)
{
    Timer t("sirius::bcast_column_gpu");

    int n = (int)(s1_col__.local_size(icol__) - s0_col__.local_size(icol__));

    if (n > 0 && kp__->rank_col() == icol__)
    {
        cuda_copy_device_to_device(m_tmp__.ptr_device(0, 0, icol__ % 2), m__.ptr_device(0, s0_col__.local_size(icol__)),
                                   kp__->num_gkvec_row() * n * sizeof(double_complex));
    }
    kp__->comm_col().bcast(m_tmp__.ptr_device(0, 0, icol__ % 2), kp__->num_gkvec_row() * n, icol__);
}
#else
void bcast_column_gpu(K_point* kp__,
                      splindex<block_cyclic>& s0_col__,
                      splindex<block_cyclic>& s1_col__,
                      int icol__,
                      dmatrix<double_complex>& m__,
                      mdarray<double_complex, 3>& m_tmp__)
{
    Timer t("sirius::bcast_column_gpu");

    int n = (int)(s1_col__.local_size(icol__) - s0_col__.local_size(icol__));

    if (n > 0 && kp__->rank_col() == icol__)
    {
        for (int i = 0; i < n; i++)
        {
            memcpy(&m_tmp__(0, i, icol__ % 2), &m__(0, s0_col__.local_size(icol__) + i),
                   kp__->num_gkvec_row() * sizeof(double_complex));
        }
    }
    kp__->comm_col().bcast(&m_tmp__(0, 0, icol__ % 2), kp__->num_gkvec_row() * n, icol__);
    #ifdef _GPU_
    cublas_set_matrix(kp__->num_gkvec_row(), n, sizeof(double_complex), &m_tmp__(0, 0, icol__ % 2), m_tmp__.ld(),
                      m_tmp__.ptr_device(0, 0, icol__ % 2), m_tmp__.ld());
    #endif
}
#endif

void Band::set_fv_h_o_uspp_gpu_parallel_v3(int N__,
                                           int n__,
                                           K_point* kp__,
                                           std::vector<double>& veff_it_coarse__,
                                           std::vector<double>& pw_ekin__,
                                           dmatrix<double_complex>& phi__,
                                           dmatrix<double_complex>& hphi__,
                                           dmatrix<double_complex>& ophi__,
                                           dmatrix<double_complex>& h__,
                                           dmatrix<double_complex>& o__,
                                           dmatrix<double_complex>& h_old__,
                                           dmatrix<double_complex>& o_old__)
{
    kp__->comm().barrier();
    Timer t("sirius::Band::set_fv_h_o_uspp_cpu_parallel", _global_timer_);
    
    splindex<block_cyclic> s0_col(N__,       kp__->num_ranks_col(), kp__->rank_col(), parameters_.cyclic_block_size());
    splindex<block_cyclic> s1_col(N__ + n__, kp__->num_ranks_col(), kp__->rank_col(), parameters_.cyclic_block_size());
    splindex<block_cyclic> s0_row(N__,       kp__->num_ranks_row(), kp__->rank_row(), parameters_.cyclic_block_size());
    splindex<block_cyclic> s1_row(N__ + n__, kp__->num_ranks_row(), kp__->rank_row(), parameters_.cyclic_block_size());

    /* copy old Hamiltonian and overlap */
    for (int i = 0; i < (int)s0_col.local_size(); i++)
    {
        memcpy(&h__(0, i), &h_old__(0, i), s0_row.local_size() * sizeof(double_complex));
        memcpy(&o__(0, i), &o_old__(0, i), s0_row.local_size() * sizeof(double_complex));
    }

    #ifdef _GPU_
    cublas_set_matrix(kp__->num_gkvec_row(), (int)(s1_col.local_size() - s0_col.local_size()), sizeof(double_complex),
                      &phi__(0, s0_col.local_size()), phi__.ld(), 
                      phi__.ptr_device(0, s0_col.local_size()), phi__.ld());
    #endif

    /* apply Hamiltonian and overlap operators to the new basis functions */
    apply_h_o_uspp_gpu_parallel_v2(kp__, veff_it_coarse__, pw_ekin__, N__, n__, phi__, hphi__, ophi__);

    #ifdef _GPU_
    cublas_get_matrix(kp__->num_gkvec_row(), (int)(s1_col.local_size() - s0_col.local_size()), sizeof(double_complex), 
                      hphi__.ptr_device(0, s0_col.local_size()), hphi__.ld(),
                      &hphi__(0, s0_col.local_size()), hphi__.ld());
    cublas_get_matrix(kp__->num_gkvec_row(), (int)(s1_col.local_size() - s0_col.local_size()), sizeof(double_complex), 
                      ophi__.ptr_device(0, s0_col.local_size()), ophi__.ld(),
                      &ophi__(0, s0_col.local_size()), ophi__.ld());
    #endif
    int max_num_hphi = 0;
    for (int icol = 0; icol < kp__->num_ranks_col(); icol++)
        max_num_hphi = std::max(max_num_hphi, (int)(s1_col.local_size(icol) - s0_col.local_size(icol)));
    
    int num_phi = (int)s1_col.local_size();
 
    mdarray<double_complex, 3> hphi_tmp(kp__->num_gkvec_row(), max_num_hphi, 2);
    mdarray<double_complex, 3> ophi_tmp(kp__->num_gkvec_row(), max_num_hphi, 2);
    mdarray<double_complex, 3> h_tmp(num_phi, max_num_hphi, 2);
    mdarray<double_complex, 3> o_tmp(num_phi, max_num_hphi, 2);

    #ifdef _GPU_
    hphi_tmp.allocate_on_device();
    ophi_tmp.allocate_on_device();
    h_tmp.allocate_on_device();
    o_tmp.allocate_on_device();
    #endif

    std::array<std::atomic_bool, 2> lock_hphi;
    std::array<std::atomic_bool, 2> lock_ophi;
    std::array<std::atomic_bool, 2> lock_h;
    std::array<std::atomic_bool, 2> lock_o;
    for (int i = 0; i < 2; i++)
    {
        lock_hphi[i].store(false);
        lock_ophi[i].store(false);
        lock_h[i].store(false);
        lock_o[i].store(false);
    }
   
    int icol = 0;
    
    Timer t1("sirius::Band::set_fv_h_o_uspp_cpu_parallel|zgemm_eff", _global_timer_);
    
    bcast_column_gpu(kp__, s0_col, s1_col, icol, hphi__, hphi_tmp);
    bcast_column_gpu(kp__, s0_col, s1_col, icol, ophi__, ophi_tmp);
    lock_hphi[0].store(true);
    lock_ophi[0].store(true);

    int nthread = omp_get_max_threads();
    if (nthread > 1) omp_set_num_threads(nthread - 1);

    /* crate communication thread */
    std::thread comm_thread([kp__, &s0_col, &s1_col, &s0_row, &s1_row, &lock_hphi, &lock_ophi, &lock_h, &lock_o, 
                             &hphi__, &ophi__, &hphi_tmp, &ophi_tmp, &h_tmp, &o_tmp, &h__, &o__]()
    {
        int num_phi = (int)s1_col.local_size();
    
        for (int icol = 0; icol < kp__->num_ranks_col(); icol++)
        {
            /* local number of new basis functions */
            int n = (int)(s1_col.local_size(icol) - s0_col.local_size(icol));
            
            Timer t1("sirius::Band::set_fv_h_o_uspp_gpu_parallel_v3|comm_thread|1");
            /* broadcast next column */
            if (icol + 1 < kp__->num_ranks_col())
            {
                while (lock_hphi[(icol + 1) % 2].load());
                //printf("#1 broadcasting column %i\n", icol);
                bcast_column_gpu(kp__, s0_col, s1_col, icol + 1, hphi__, hphi_tmp);
                lock_hphi[(icol + 1) % 2].store(true);
                
                while (lock_ophi[(icol + 1) % 2].load());
                //printf("#1 broadcasting column %i\n", icol);
                bcast_column_gpu(kp__, s0_col, s1_col, icol + 1, ophi__, ophi_tmp);
                lock_ophi[(icol + 1) % 2].store(true);
            }
            t1.stop();
    
            Timer t2("sirius::Band::set_fv_h_o_uspp_gpu_parallel_v3|comm_thread|2");
            if (n > 0)
            {
                while (!lock_h[icol % 2].load());
                //printf("#2 reducing h for column %i\n", icol);
                kp__->comm_row().allreduce(&h_tmp(0, 0, icol % 2), num_phi * n);
    
                for (int j = 0; j < n; j++)
                {
                    int idx_hphi_glob = (int)s1_col.global_index(s0_col.local_size(icol) + j, icol);
                    auto p = s1_row.location(idx_hphi_glob);
                    if (p.second == kp__->rank_row())
                    {
                        for (int i = 0; i < num_phi; i++)
                        {
                            h__(p.first, i) = conj(h_tmp(i, j, icol % 2));
                        }
                    }
                }
                /* remove lock from h buffer */
                lock_h[icol % 2].store(false);
    
                while (!lock_o[icol % 2].load());
                kp__->comm_row().allreduce(&o_tmp(0, 0, icol % 2), num_phi * n);
    
                for (int j = 0; j < n; j++)
                {
                    int idx_hphi_glob = (int)s1_col.global_index(s0_col.local_size(icol) + j, icol);
                    auto p = s1_row.location(idx_hphi_glob);
                    if (p.second == kp__->rank_row())
                    {
                        for (int i = 0; i < num_phi; i++)
                        {
                            o__(p.first, i) = conj(o_tmp(i, j, icol % 2));
                        }
                    }
                }
                /* remove lock from o buffer */
                lock_o[icol % 2].store(false);
            }
            t2.stop();
        }
    });

    for (int icol = 0; icol < kp__->num_ranks_col(); icol++)
    {
        int n = (int)(s1_col.local_size(icol) - s0_col.local_size(icol));

        /* wait for broadcast of this column */
        while (!lock_hphi[icol % 2].load());
        /* wait for unlock of h buffer */
        while (lock_h[icol % 2].load());

        if (n > 0)
        {
            //printf("#5 zgemm for column %i\n", icol);
            Timer t2("sirius::Band::set_fv_h_o_uspp_cpu_parallel|zgemm_loc", _global_timer_);
            blas<gpu>::gemm(2, 0, num_phi, n, kp__->num_gkvec_row(), phi__.ptr_device(), phi__.ld(),
                            hphi_tmp.ptr_device(0, 0, icol % 2), hphi_tmp.ld(), h_tmp.ptr_device(0, 0, icol % 2), h_tmp.ld());
            h_tmp.copy_to_host();
            lock_h[icol % 2].store(true);
            lock_hphi[icol % 2].store(false);
        }
            
        while (!lock_ophi[icol % 2].load());
        while (lock_o[icol % 2].load());
        if (n > 0)
        {
            Timer t2("sirius::Band::set_fv_h_o_uspp_cpu_parallel|zgemm_loc", _global_timer_);
            //printf("#6 zgemm for column %i\n", icol);
            blas<gpu>::gemm(2, 0, num_phi, n, kp__->num_gkvec_row(), phi__.ptr_device(), phi__.ld(),
                            ophi_tmp.ptr_device(0, 0, icol % 2), ophi_tmp.ld(), o_tmp.ptr_device(0, 0, icol % 2), o_tmp.ld());
            o_tmp.copy_to_host();
            lock_o[icol % 2].store(true);
            lock_ophi[icol % 2].store(false);
        }
    }
    comm_thread.join();
    omp_set_num_threads(nthread);

    double tval = t1.stop();

    if (verbosity_level >= 6 && kp__->comm().rank() == 0)
    {
        printf("effective zgemm #4&5 with M, N, K: %6i %6i %6i,                        %12.4f sec, %12.4f GFlops/node\n",
               N__ + n__, n__, kp__->num_gkvec(),
               tval, 2 * 8e-9 * (N__ + n__) * n__ * kp__->num_gkvec() / tval / kp__->num_ranks());
    }

    /* restore right block of the matrix */
    if (N__ != 0)
    {
        dmatrix<double_complex>::tranc(N__, n__, h__, N__, 0, h__, 0, N__);
        dmatrix<double_complex>::tranc(N__, n__, o__, N__, 0, o__, 0, N__);
    }

    /* save Hamiltonian and overlap */
    for (int i = 0; i < (int)s1_col.local_size(); i++)
    {
        memcpy(&h_old__(0, i), &h__(0, i), s1_row.local_size() * sizeof(double_complex));
        memcpy(&o_old__(0, i), &o__(0, i), s1_row.local_size() * sizeof(double_complex));
    }

    kp__->comm().barrier();
}

void Band::uspp_residuals_gpu_parallel(int N__,
                                       int num_bands__,
                                       K_point* kp__,
                                       std::vector<double>& eval__,
                                       dmatrix<double_complex>& evec__,
                                       dmatrix<double_complex>& hphi__,
                                       dmatrix<double_complex>& ophi__,
                                       dmatrix<double_complex>& hpsi__,
                                       dmatrix<double_complex>& opsi__,
                                       dmatrix<double_complex>& res__,
                                       std::vector<double_complex>& h_diag__,
                                       std::vector<double_complex>& o_diag__,
                                       std::vector<double>& res_norm__)

{
    Timer t("sirius::Band::uspp_residuals_gpu_parallel");

    Timer t1("sirius::Band::uspp_residuals_gpu_parallel|zgemm_eff");

    splindex<block_cyclic> spl_num_bands_col(num_bands__, kp__->num_ranks_col(), kp__->rank_col(), parameters_.cyclic_block_size());

    dmatrix<double_complex> evec_t(num_bands__, N__, kp__->blacs_grid());
    dmatrix<double_complex>::tranu(num_bands__, N__, evec__, 0, 0, evec_t, 0, 0);
    
    /* local number of basis function |phi> */
    int num_phi_loc = evec_t.num_cols_local();

    mdarray<double_complex, 2> evec_t_all(num_bands__, num_phi_loc);
    evec_t_all.zero();
    for (int j = 0; j < num_phi_loc; j++)
    {
        for (int i = 0; i < evec_t.num_rows_local(); i++)
        {
            evec_t_all(evec_t.irow(i), j) = evec_t(i, j);
        }
    }
    kp__->comm_row().allreduce(evec_t_all.ptr(), (int)evec_t_all.size());

    mdarray<double_complex, 2> tmp(kp__->num_gkvec_row(), spl_num_bands_col.local_size(0));

    for (int rank_col = 0; rank_col < kp__->num_ranks_col(); rank_col++)
    {
        mdarray<double_complex, 2> evec_tmp(num_phi_loc, spl_num_bands_col.local_size(rank_col));
        for (int i = 0; i < (int)spl_num_bands_col.local_size(rank_col); i++)
        {
            for (int j = 0; j < num_phi_loc; j++) 
                evec_tmp(j, i) = evec_t_all(spl_num_bands_col.global_index(i, rank_col), j);
        }
        
        int num_bands_of_rank = (int)spl_num_bands_col.local_size(rank_col);

        blas<cpu>::gemm(0, 0, kp__->num_gkvec_row(), num_bands_of_rank, num_phi_loc, 
                        &hphi__(0, 0), hphi__.ld(), &evec_tmp(0, 0), evec_tmp.ld(), &tmp(0, 0), tmp.ld());
        kp__->comm_col().reduce(tmp.ptr(), &hpsi__(0, 0), kp__->num_gkvec_row() * num_bands_of_rank, rank_col); 
        
        blas<cpu>::gemm(0, 0, kp__->num_gkvec_row(), num_bands_of_rank, num_phi_loc, 
                        &ophi__(0, 0), ophi__.ld(), &evec_tmp(0, 0), evec_tmp.ld(), &tmp(0, 0), tmp.ld());
        kp__->comm_col().reduce(tmp.ptr(), &opsi__(0, 0), kp__->num_gkvec_row() * num_bands_of_rank, rank_col); 
    }

    //== splindex<block> sub_spl_gkvec(kp__->num_gkvec_row(), kp__->num_ranks_col(), kp__->rank_col());

    //== mdarray<double_complex, 2> hphi_slice(N__, sub_spl_gkvec.local_size());
    //== hphi__.shuffle_horizontal<_panel_to_slice_>(N__, hphi_slice);

    //== mdarray<double_complex, 2> ophi_slice(N__, sub_spl_gkvec.local_size());
    //== ophi__.shuffle_horizontal<_panel_to_slice_>(N__, ophi_slice);

    //== mdarray<double_complex, 2> hpsi_slice(num_bands__, sub_spl_gkvec.local_size());
    //== mdarray<double_complex, 2> opsi_slice(num_bands__, sub_spl_gkvec.local_size());


    //== splindex<block_cyclic> spl_bands(num_bands__, kp__->num_ranks_row(), kp__->rank_row(), parameters_.cyclic_block_size());

    //== mdarray<double_complex, 2> hpsi_slice_tmp(spl_bands.local_size(0), sub_spl_gkvec.local_size());
    //== mdarray<double_complex, 2> opsi_slice_tmp(spl_bands.local_size(0), sub_spl_gkvec.local_size());
    //== 
    //== Timer t2("sirius::Band::uspp_cpu_residuals_parallel_v2|zgemm_loop");
    //== for (int irow = 0; irow < kp__->num_ranks_row(); irow++)
    //== {
    //==     mdarray<double_complex, 2> evec_tmp(spl_bands.local_size(irow), N__);
    //==     evec_tmp.zero();
    //==     if (irow == kp__->rank_row())
    //==     {
    //==         for (int j = 0; j < evec_t.num_cols_local(); j++)
    //==         {
    //==             for (int i = 0; i < (int)spl_bands.local_size(irow); i++)
    //==             {
    //==                 evec_tmp(i, evec_t.icol(j)) = evec_t(i, j);
    //==             }
    //==         }
    //==         kp__->comm_col().allreduce(evec_tmp.ptr(), (int)evec_tmp.size());
    //==     }
    //==     kp__->comm_row().bcast(evec_tmp.ptr(), (int)evec_tmp.size(), irow);

    //==     Timer t3("sirius::Band::uspp_cpu_residuals_parallel_v2|zgemm_loc");
    //==     blas<cpu>::gemm(0, 0, (int)spl_bands.local_size(irow), (int)sub_spl_gkvec.local_size(), N__, 
    //==                     evec_tmp.ptr(), evec_tmp.ld(), hphi_slice.ptr(), hphi_slice.ld(), 
    //==                     hpsi_slice_tmp.ptr(), hpsi_slice_tmp.ld());
    //==     
    //==     blas<cpu>::gemm(0, 0, (int)spl_bands.local_size(irow), (int)sub_spl_gkvec.local_size(), N__, 
    //==                     evec_tmp.ptr(), evec_tmp.ld(), ophi_slice.ptr(), ophi_slice.ld(), 
    //==                     opsi_slice_tmp.ptr(), opsi_slice_tmp.ld());
    //==     t3.stop();
    //==     for (int j = 0; j < (int)sub_spl_gkvec.local_size(); j++)
    //==     {
    //==         for (int i = 0; i < (int)spl_bands.local_size(irow); i++)
    //==         {
    //==             hpsi_slice(spl_bands.global_index(i, irow), j) = hpsi_slice_tmp(i, j);
    //==             opsi_slice(spl_bands.global_index(i, irow), j) = opsi_slice_tmp(i, j);
    //==         }
    //==     }
    //== }
    //== t2.stop();
    //== 
    //== hpsi__.shuffle_horizontal<_slice_to_panel_>(num_bands__, hpsi_slice);
    //== opsi__.shuffle_horizontal<_slice_to_panel_>(num_bands__, opsi_slice);
        


    /* Compute H\Psi_{i} = H\phi_{mu} * Z_{mu, i} */
    //blas<cpu>::gemm(0, 0, kp__->num_gkvec(), num_bands__, N__, complex_one, hphi__, evec__, complex_zero, hpsi__);
    /* Compute O\Psi_{i} = O\phi_{mu} * Z_{mu, i} */
    //blas<cpu>::gemm(0, 0, kp__->num_gkvec(), num_bands__, N__, complex_one, ophi__, evec__, complex_zero, opsi__);
    double tval = t1.stop();

    if (verbosity_level >= 6 && kp__->comm().rank() == 0)
    {
        printf("effective zgemm #6&7 with M, N, K: %6i %6i %6i,                        %12.4f sec, %12.4f GFlops/node\n",
               kp__->num_gkvec(), num_bands__, N__,
               tval, 2 * 8e-9 * kp__->num_gkvec() * num_bands__ * N__ / tval / kp__->num_ranks());
    }

    memset(&res_norm__[0], 0, num_bands__ * sizeof(double));
    /* compute residuals r_{i} = H\Psi_{i} - E_{i}O\Psi_{i} and norm squared */
    #pragma omp parallel for
    for (int i = 0; i < res__.num_cols_local(); i++)
    {
        int ires = res__.icol(i);
        double norm2 = 0;
        for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) 
        {
            res__(igk_row, i) = hpsi__(igk_row, i) - eval__[ires] * opsi__(igk_row, i);
            norm2 += real(conj(res__(igk_row, i)) * res__(igk_row, i));
        }
        res_norm__[ires] = norm2;
    }
    kp__->comm().allreduce(res_norm__);
    
    /* compute norm */
    for (int i = 0; i < num_bands__; i++) res_norm__[i] = std::sqrt(res_norm__[i]);
    
    /* apply preconditioner */
    #pragma omp parallel for
    for (int i = 0; i < res__.num_cols_local(); i++)
    {
        int ires = res__.icol(i);
    
        for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++)
        {
            double_complex z = h_diag__[igk_row] - eval__[ires] * o_diag__[igk_row];
            if (std::abs(z) < 1e-12) error_local(__FILE__, __LINE__, "problematic division");
            res__(igk_row, i) /= z;
        }
    }
    
    std::vector<double> norm2(num_bands__, 0);
    /* Normalize new basis functions */
    #pragma omp parallel for
    for (int i = 0; i < res__.num_cols_local(); i++)
    {
        int ires = res__.icol(i);
        double d = 0;
        for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) 
            d += real(conj(res__(igk_row, i)) * res__(igk_row, i));
        norm2[ires] = d;
    }
    kp__->comm().allreduce(norm2);
    #pragma omp parallel for
    for (int i = 0; i < res__.num_cols_local(); i++)
    {
        int ires = res__.icol(i);
        double d = 1.0 / std::sqrt(norm2[ires]);
        for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) res__(igk_row, i) *= d;
    }
}

void Band::diag_fv_uspp_gpu_parallel(K_point* kp__,
                                     double v0__,
                                     std::vector<double>& veff_it_coarse__)
{
    log_function_enter(__func__);
    Timer t("sirius::Band::diag_fv_uspp_gpu_parallel", _global_timer_);

    /* cache kinetic energy */
    std::vector<double> pw_ekin = kp__->get_pw_ekin();

    /* get diagonal elements for preconditioning */
    std::vector<double_complex> h_diag;
    std::vector<double_complex> o_diag;
    get_h_o_diag(kp__, v0__, pw_ekin, h_diag, o_diag);

    /* short notation for number of target wave-functions */
    int num_bands = parameters_.num_fv_states();

    auto& itso = parameters_.iterative_solver_input_section_;

    int num_phi = std::min(itso.subspace_size_ * num_bands, kp__->num_gkvec());

    dmatrix<double_complex> phi(kp__->num_gkvec(), num_phi, kp__->blacs_grid());
    dmatrix<double_complex> hphi(kp__->num_gkvec(), num_phi, kp__->blacs_grid());
    dmatrix<double_complex> ophi(kp__->num_gkvec(), num_phi, kp__->blacs_grid());

    #ifdef _GPU_
    phi.allocate_on_device();
    hphi.allocate_on_device();
    ophi.allocate_on_device();
    #endif

    /* current diagonalziation subspace size */
    int N = 0;

    /* number of newly added basis functions */
    int n = num_bands;

    dmatrix<double_complex> hmlt(num_phi, num_phi, kp__->blacs_grid());
    dmatrix<double_complex> ovlp(num_phi, num_phi, kp__->blacs_grid());
    dmatrix<double_complex> hmlt_old(num_phi, num_phi, kp__->blacs_grid());
    dmatrix<double_complex> ovlp_old(num_phi, num_phi, kp__->blacs_grid());
    
    dmatrix<double_complex> evec(num_phi, num_bands, kp__->blacs_grid());
    std::vector<double> eval(num_bands);
    std::vector<double> eval_old(num_bands);

    /* alias for wave-functions */
    dmatrix<double_complex>& psi = kp__->fv_states_panel();
    
    dmatrix<double_complex> hpsi(kp__->num_gkvec(), num_bands, kp__->blacs_grid());
    dmatrix<double_complex> opsi(kp__->num_gkvec(), num_bands, kp__->blacs_grid());
    dmatrix<double_complex> res(kp__->num_gkvec(), num_bands, kp__->blacs_grid());

    /* trial basis functions */
    assert(phi.num_rows_local() == psi.num_rows_local());
    for (int i = 0; i < psi.num_cols_local(); i++) 
        memcpy(&phi(0, i), &psi(0, i), kp__->num_gkvec_row() * sizeof(double_complex));
    //#ifdef _GPU_
    //cublas_set_matrix(kp__->num_gkvec_row(), psi.num_cols_local(), sizeof(double_complex), phi.ptr(), phi.ld(), 
    //                  phi.ptr_device(), phi.ld());
    //#endif

    std::vector<double> res_norm(num_bands);
    std::vector<double> res_rms(num_bands);

    /* start iterative diagonalization */
    for (int k = 0; k < itso.num_steps_; k++)
    {
        /* set H and O for the variational subspace */
        set_fv_h_o_uspp_gpu_parallel_v3(N, n, kp__, veff_it_coarse__, pw_ekin, phi, hphi, ophi, hmlt, ovlp, hmlt_old, ovlp_old);
        
        /* increase size of the variation space */
        N += n;

        if (verbosity_level >= 6 && kp__->comm().rank() == 0)
        {
            printf("iteration : %i, subspace size : %i\n", k, N);
        }

        {
        Timer t2("sirius::Band::diag_fv_uspp_cpu_parallel|solve_gevp");
        eval_old = eval;
        gen_evp_solver()->solve(N, hmlt.num_rows_local(), hmlt.num_cols_local(), num_bands, 
                                hmlt.ptr(), hmlt.ld(), ovlp.ptr(), ovlp.ld(), 
                                &eval[0], evec.ptr(), evec.ld());
        
        if (kp__->comm().rank() == 0)
        {
            printf("subspace size : %i, eigen-values:\n", N);
            for (int i = 0; i < std::min(num_bands, 10); i++) printf("%18.12f ", eval[i]);
            printf("\n");
        }
        }

        /* don't recompute residuals if we are going to exit on the last iteration */
        std::vector<int> res_list;
        if (k != itso.num_steps_ - 1)
        {
            uspp_residuals_gpu_parallel(N, num_bands, kp__, eval, evec, hphi, ophi, hpsi, opsi, res, h_diag, o_diag, res_norm);
            //uspp_cpu_residuals_parallel(N, num_bands, kp__, eval, evec, hphi, ophi, hpsi, opsi, res, h_diag, o_diag, res_norm);

            for (int i = 0; i < num_bands; i++)
            {
                /* take the residual if it's norm is above the threshold */
                if (kp__->band_occupancy(i) > 1e-12 &&
                    (res_norm[i] > itso.tolerance_ || (res_norm[i] > itso.extra_tolerance_ && n != 0)))
                {
                    res_list.push_back(i);
                }
            }

            /* number of additional basis functions */
            n = (int)res_list.size();
        }

        /* check if we run out of variational space or eigen-vectors are converged or it's a last iteration */
        if (N + n > num_phi || n == 0 || k == (itso.num_steps_ - 1))
        {   
            Timer t3("sirius::Band::diag_fv_uspp_cpu_parallel|update_phi", _global_timer_);

            /* recompute wave-functions: \Psi_{i} = \phi_{mu} * Z_{mu, i} */
            blas<cpu>::gemm(0, 0, kp__->num_gkvec(), num_bands, N, complex_one, phi, evec, complex_zero, psi); 
            
            /* exit loop if the eigen-vectors are converged or this is the last iteration */
            if (n == 0 || k == (itso.num_steps_ - 1))
            {
                if (verbosity_level >= 6 && kp__->comm().rank() == 0)
                {
                    double demax = 0;
                    for (int i = 0; i < num_bands; i++)
                    {
                         if (kp__->band_occupancy(i) > 1e-12) demax = std::max(demax, std::abs(eval_old[i] - eval[i]));
                    }
                    if (k == 0) demax = 0.0;
                    printf("converged in %i iterations with maximum eigen-value error %18.12e\n", k, demax);
                }
                break;
            }

            for (int i = 0; i < psi.num_cols_local(); i++) 
            {
                /* update \phi */
                memcpy(&phi(0, i), &psi(0, i), kp__->num_gkvec_row() * sizeof(double_complex));
                /* update H\phi */
                memcpy(&hphi(0, i), &hpsi(0, i), kp__->num_gkvec_row() * sizeof(double_complex));
                /* update O\phi */
                memcpy(&ophi(0, i), &opsi(0, i), kp__->num_gkvec_row() * sizeof(double_complex));
            }
            STOP();

            /* update H and O matrices. */
            hmlt_old.zero();
            ovlp_old.zero();
            for (int i = 0; i < num_bands; i++)
            {
                hmlt_old.set(i, i, eval[i]);
                ovlp_old.set(i, i, complex_one);
            }
            
            /* set new size of the variational space */
            N = num_bands;
        }
        
        /* expand variational space with extra basis functions */
        for (int i = 0; i < n; i++)
        {
            dmatrix<double_complex>::copy_col(res, res_list[i], phi, N + i);
        }
    }

    kp__->set_fv_eigen_values(&eval[0]);
}
#endif

}
