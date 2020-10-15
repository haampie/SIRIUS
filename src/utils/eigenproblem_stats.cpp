#include "eigenproblem_stats.hpp"

#include "json.hpp"

namespace utils {

std::vector<eigenproblem_stat> davidson_stats::eig_stats = {};
std::vector<orthogonalization_stats> davidson_stats::ortho_stats = {};

std::ostream &operator<<(std::ostream &os, davidson_stats const &stats) {
    nlohmann::json data;
    
    {
        nlohmann::json eig_stats;
        for (auto const &stat : stats.eig_stats) {
            eig_stats.push_back({
                {"problem_size", stat.problem_size},
                {"num_eigenpairs_computed", stat.num_eigenpairs_computed},
                {"num_unconverged", stat.num_unconverged},
                {"num_locked", stat.num_locked}
            });
        }
        data["eig_stats"] = eig_stats;
    }

    {
        nlohmann::json ortho_stats;
        for (auto const &stat : stats.ortho_stats) {
            ortho_stats.push_back(stat.cholesky_diag);
        }
        data["ortho_stats"] = ortho_stats;
    }

    os << data.dump(4) << '\n';

    return os;
}

}
