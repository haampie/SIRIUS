#include "eigenproblem_stats.hpp"

namespace utils {

std::vector<eigenproblem_stat> eigenproblem_stats::stats = {};

std::ostream &operator<<(std::ostream &os, eigenproblem_stats const &stats) {
    for (auto const &stat : stats.stats)
        os << stat.problem_size << "\t" << stat.num_eigenpairs_computed << "\t" << stat.num_unconverged << "\t" << stat.num_locked << "\n";
    return os;
}

}
