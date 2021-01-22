#include "utils/profiler.hpp"
#include <sirius.hpp>
#include "../dft_loop/filesystem.hpp"
#include <utils/json.hpp>
#include <cfenv>
#include <fenv.h>

using namespace sirius;
using json = nlohmann::json;

const std::string aiida_output_file = "output_aiida.json";

enum class task_t : int
{
    ground_state_new     = 0,
    ground_state_restart = 1,
};

void rewrite_relative_paths(json& dict__, fs::path const &working_directory = fs::current_path())
{
    // the json.unit_cell.atom_files[] dict might contain relative paths,
    // which should be relative to the json file. So better make them
    // absolute such that the simulation context does not have to be
    // aware of paths.
    if (!dict__.count("unit_cell"))
        return;

    auto &section = dict__["unit_cell"];

    if (!section.count("atom_files"))
        return;

    auto &atom_files = section["atom_files"];

    for (auto& label : atom_files.items()) {
        label.value() = working_directory / std::string(label.value());
    }
}

nlohmann::json preprocess_json_input(std::string fname__)
{
    if (fname__.find("{") == std::string::npos) {
        // If it's a file, set the working directory to that file.
        auto json = utils::read_json_from_file(fname__);
        rewrite_relative_paths(json, fs::path{fname__}.parent_path());
        return json;
    } else {
        // Raw JSON input
        auto json = utils::read_json_from_string(fname__);
        rewrite_relative_paths(json);
        return json;
    }
}

std::unique_ptr<Simulation_context> create_sim_ctx(std::string fname__,
                                                   cmd_args const& args__)
{

    auto json = preprocess_json_input(fname__);

    auto ctx_ptr = std::make_unique<Simulation_context>(json.dump(), Communicator::world());
    Simulation_context& ctx = *ctx_ptr;

    auto& inp = ctx.parameters_input();
    if (inp.gamma_point_ && !(inp.ngridk_[0] * inp.ngridk_[1] * inp.ngridk_[2] == 1)) {
        TERMINATE("this is not a Gamma-point calculation")
    }

    ctx.import(args__);

    return ctx_ptr;
}


double ground_state(Simulation_context& ctx,
                    task_t              task,
                    cmd_args const&     args,
                    int                 write_output)
{
    auto& inp = ctx.parameters_input();

    K_point_set kset(ctx, ctx.parameters_input().ngridk_, ctx.parameters_input().shiftk_, ctx.use_symmetry());
    DFT_ground_state dft(kset);

    auto& potential = dft.potential();
    auto& density = dft.density();
    dft.initial_state();

    double initial_tol = ctx.iterative_solver_tolerance();

    /* launch the calculation */
    auto result = dft.find(inp.density_tol_, inp.energy_tol_, initial_tol, inp.num_dft_iter_, true);

    /* wait for all */
    // ctx.comm().barrier();

    // now do something linear responsy.
    // Hamiltonian0 H0(potential);

    // for (int ikloc = 0; ikloc < kset.spl_num_kpoints().local_size(); ikloc++) {
    //     int ik  = kset.spl_num_kpoints(ikloc);
    //     auto kp = kset[ik];
    //     auto Hk = H0(*kp);

    //     // now do something with cg.
    // }
}

/// Run a task based on a command line input.
void run_tasks(cmd_args const& args)
{
    /* get the task id */
    task_t task = static_cast<task_t>(args.value<int>("task", 0));

    /* get the input file name */
    auto fpath = args.value<fs::path>("input", "sirius.json");

    if (fs::is_directory(fpath)) {
        fpath /= "sirius.json";
    }

    if (!fs::exists(fpath)) {
        if (Communicator::world().rank() == 0) {
            std::printf("input file does not exist\n");
        }
        return;
    }

    auto fname = fpath.string();

    if (task == task_t::ground_state_new || task == task_t::ground_state_restart) {
        auto ctx = create_sim_ctx(fname, args);
        ctx->initialize();
        ground_state(*ctx, task, args, 1);
    }
}

int main(int argn, char** argv)
{
    std::feclearexcept(FE_ALL_EXCEPT);
    cmd_args args;
    args.register_key("--input=", "{string} input file name");
    args.register_key("--output=", "{string} output file name");
    args.register_key("--task=", "{int} task id");
    args.register_key("--aiida_output", "write output for AiiDA");
    args.register_key("--test_against=", "{string} json file with reference values");
    args.register_key("--repeat_update=", "{int} number of times to repeat update()");
    args.register_key("--fpe", "enable check of floating-point exceptions using GNUC library");
    args.register_key("--control.processing_unit=", "");
    args.register_key("--control.verbosity=", "");
    args.register_key("--control.verification=", "");
    args.register_key("--control.mpi_grid_dims=","");
    args.register_key("--control.std_evp_solver_name=", "");
    args.register_key("--control.gen_evp_solver_name=", "");
    args.register_key("--control.fft_mode=", "");
    args.register_key("--control.memory_usage=", "");
    args.register_key("--parameters.ngridk=", "");
    args.register_key("--parameters.gamma_point=", "");
    args.register_key("--parameters.pw_cutoff=", "");
    args.register_key("--iterative_solver.orthogonalize=", "");
    args.register_key("--iterative_solver.early_restart=", "{double} value between 0 and 1 to control the early restart ratio in Davidson");
    args.register_key("--mixer.type=", "{string} mixer name (anderson, anderson_stable, broyden2, linear)");
    args.register_key("--mixer.beta=", "{double} mixing parameter");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        std::printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

#if defined(_GNU_SOURCE)
    if (args.exist("fpe")) {
        feenableexcept(FE_DIVBYZERO|FE_INVALID|FE_OVERFLOW);
    }
#endif

    sirius::initialize(1);

    run_tasks(args);

    int my_rank = Communicator::world().rank();

    sirius::finalize(1);

    if (my_rank == 0)  {
        auto timing_result = ::utils::global_rtgraph_timer.process();
        std::cout << timing_result.print({rt_graph::Stat::Count, rt_graph::Stat::Total, rt_graph::Stat::Percentage,
                                          rt_graph::Stat::SelfPercentage, rt_graph::Stat::Median, rt_graph::Stat::Min,
                                          rt_graph::Stat::Max});
        std::ofstream ofs("timers.json", std::ofstream::out | std::ofstream::trunc);
        ofs << timing_result.json();
    }
    if (std::fetestexcept(FE_DIVBYZERO)) {
        std::cout << "FE_DIVBYZERO exception\n";
    }
    if (std::fetestexcept(FE_INVALID)) {
        std::cout << "FE_INVALID exception\n";
    }
    if (std::fetestexcept(FE_UNDERFLOW)) {
        std::cout << "FE_UNDERFLOW exception\n";
    }
    if (std::fetestexcept(FE_OVERFLOW)) {
        std::cout << "FE_OVERFLOW exception\n";
    }

    return 0;
}
