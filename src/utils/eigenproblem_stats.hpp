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

class eigenproblem_stats {
public:
    static std::vector<eigenproblem_stat> stats;
};

std::ostream &operator<<(std::ostream &os, eigenproblem_stats const &stats);

}

#endif