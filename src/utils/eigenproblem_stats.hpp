#ifndef __EIGENPROBLEM_STATS_HPP__
#define __EIGENPROBLEM_STATS_HPP__

#include <vector>
#include <ostream>

namespace utils {

struct eigenproblem_stat {
    int problem_size;
    int num_eigenpairs_computed;
    int num_unconverged;
    int num_locked;
};

struct orthogonalization_stats {
    std::vector<double> cholesky_diag;
};

class davidson_stats {
public:
    static std::vector<eigenproblem_stat> eig_stats;
    static std::vector<orthogonalization_stats> ortho_stats;
};

std::ostream &operator<<(std::ostream &os, davidson_stats const &stats);

}

#endif