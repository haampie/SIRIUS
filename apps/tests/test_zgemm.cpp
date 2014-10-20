#include <sirius.h>

void test_gemm(int M, int N, int K, int transa)
{
    sirius::Timer t("test_gemm"); 
    
    mdarray<double_complex, 2> a, b, c;
    int imax, jmax;
    if (transa == 0)
    {
        imax = M;
        jmax = K;
    }
    else
    {
        imax = K;
        jmax = M;
    }
    a = matrix<double_complex>(nullptr, imax, jmax);
    b = matrix<double_complex>(nullptr, K, N);
    c = matrix<double_complex>(nullptr, M, N);
    a.allocate(alloc_mode);
    b.allocate(alloc_mode);
    c.allocate(alloc_mode);

    for (int j = 0; j < jmax; j++)
    {
        for (int i = 0; i < imax; i++) a(i, j) = type_wrapper<double_complex>::random();
    }

    for (int j = 0; j < N; j++)
    {
        for (int i = 0; i < K; i++) b(i, j) = type_wrapper<double_complex>::random();
    }

    c.zero();

    printf("testing serial zgemm with M, N, K = %i, %i, %i, opA = %i\n", M, N, K, transa);
    printf("a.ld() = %i\n", a.ld());
    printf("b.ld() = %i\n", b.ld());
    printf("c.ld() = %i\n", c.ld());
    sirius::Timer t1("gemm_only"); 
    blas<cpu>::gemm(transa, 0, M, N, K, a.ptr(), a.ld(), b.ptr(), b.ld(), c.ptr(), c.ld());
    double tval = t1.stop();
    printf("execution time (sec) : %12.6f\n", tval);
    printf("performance (GFlops) : %12.6f\n", 8e-9 * M * N * K / tval);
}

#ifdef _TEST_REAL_
typedef double gemm_type;
#else
typedef double_complex gemm_type;
#endif

#ifdef _SCALAPACK_
double test_pgemm(int M, int N, int K, int nrow, int ncol, int transa, int n)
{
    //== #ifdef _GPU_
    //== sirius::pstdout pout;
    //== pout.printf("rank : %3i, free GPU memory (Mb) : %10.2f\n", Platform::mpi_rank(), cuda_get_free_mem() / double(1 << 20));
    //== pout.flush(0);
    //== #endif
    BLACS_grid blacs_grid(MPI_COMM_WORLD, nrow, ncol);

    dmatrix<gemm_type> a, b, c;
    if (transa == 0)
    {
        a = dmatrix<gemm_type>(nullptr, M, K, blacs_grid);
    }
    else
    {
        a = dmatrix<gemm_type>(nullptr, K, M, blacs_grid);
    }
    b = dmatrix<gemm_type>(nullptr, K, N, blacs_grid);
    c = dmatrix<gemm_type>(nullptr, M, N - n, blacs_grid);
    a.allocate(alloc_mode);
    b.allocate(alloc_mode);
    c.allocate(alloc_mode);

    for (int ic = 0; ic < a.num_cols_local(); ic++)
    {
        for (int ir = 0; ir < a.num_rows_local(); ir++) a(ir, ic) = type_wrapper<gemm_type>::random();
    }

    for (int ic = 0; ic < b.num_cols_local(); ic++)
    {
        for (int ir = 0; ir < b.num_rows_local(); ir++) b(ir, ic) = type_wrapper<gemm_type>::random();
    }

    c.zero();

    if (Platform::comm_world().rank() == 0)
    {
        printf("testing parallel zgemm with M, N, K = %i, %i, %i, opA = %i\n", M, N - n, K, transa);
        printf("nrow, ncol = %i, %i, bs = %i\n", nrow, ncol, linalg<scalapack>::cyclic_block_size());
    }
    Platform::comm_world().barrier();
    sirius::Timer t1("gemm_only"); 
    gemm_type one = 1;
    gemm_type zero = 0;
    blas<cpu>::gemm(transa, 0, M, N - n, K, one, a, 0, 0, b, 0, n, zero, c, 0, 0);
    //== #ifdef _GPU_
    //== cuda_device_synchronize();
    //== #endif
    Platform::comm_world().barrier();
    double tval = t1.stop();
    double perf = 8e-9 * M * (N - n) * K / tval / nrow / ncol;
    if (Platform::comm_world().rank() == 0)
    {
        printf("execution time : %12.6f seconds\n", tval);
        printf("performance    : %12.6f GFlops / rank\n", perf);
    }

    return perf;
}
#endif

int main(int argn, char **argv)
{
    cmd_args args;
    args.register_key("--M=", "{int} M");
    args.register_key("--N=", "{int} N");
    args.register_key("--K=", "{int} K");
    args.register_key("--opA=", "{0|1|2} 0: op(A) = A, 1: op(A) = A', 2: op(A) = conjg(A')");
    args.register_key("--n=", "{int} skip first n elements in N index");
    args.register_key("--nrow=", "{int} number of row MPI ranks");
    args.register_key("--ncol=", "{int} number of column MPI ranks");
    args.register_key("--bs=", "{int} cyclic block size");
    args.register_key("--repeat=", "{int} repeat test number of times");

    args.parse_args(argn, argv);
    if (argn == 1)
    {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        exit(0);
    }

    int nrow = args.value<int>("nrow", 1);
    int ncol = args.value<int>("ncol", 1);

    int M = args.value<int>("M");
    int N = args.value<int>("N");
    int K = args.value<int>("K");

    int transa = args.value<int>("opA", 0);

    int n = args.value<int>("n", 0);

    int repeat = args.value<int>("repeat", 1);

    Platform::initialize(true);

    if (nrow * ncol == 1)
    {
        for (int i = 0; i < repeat; i++) test_gemm(M, N, K, transa);
    }
    else
    {
        #ifdef _SCALAPACK_
        int bs = args.value<int>("bs");
        linalg<scalapack>::set_cyclic_block_size(bs);
        double perf = 0;
        for (int i = 0; i < repeat; i++) perf += test_pgemm(M, N, K, nrow, ncol, transa, n);
        if (Platform::comm_world().rank() == 0)
        {
            printf("\n");
            printf("average performance    : %12.6f GFlops / rank\n", perf / repeat);
        }
        #else
        terminate(__FILE__, __LINE__, "not compiled with ScaLAPACK support");
        #endif
    }

    Platform::finalize();
}
