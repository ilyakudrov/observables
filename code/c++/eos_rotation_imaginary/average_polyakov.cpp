#include "data_processing.h"
#include "make_observables.h"
#include "read_data.h"
#include "visitors.h"

#include "omp.h"
#include <DataFrame/DataFrame.h>
#include <DataFrame/Utils/Threads/ThreadGranularity.h>
#include <filesystem>
#include <fstream>
#include <map>
#include <tuple>

void add_to_map(
    std::map<std::tuple<int, int, double>,
             std::vector<std::tuple<double, double>>> &result,
    std::map<std::string, std::tuple<double, double>> &jackknife_aver,
    int bin_size, int box_size, double radius) {
  std::vector<std::string> observables = {"ReL", "ImL", "modL", "mod<L>"};
  std::vector<std::tuple<double, double>> tmp(4);
  for (int i = 0; i < 4; i++) {
    tmp[i] = std::make_tuple(std::get<0>(jackknife_aver[observables[i]]),
                             std::get<1>(jackknife_aver[observables[i]]));
  }
  result[std::make_tuple(bin_size, box_size, radius)] = tmp;
}

int main(int argc, char *argv[]) {
  std::string base_path;
  std::string beta;
  std::string velocity;
  std::string lattice_size;
  std::string boundary;
  std::string result_path;
  std::string spec_additional_path;
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--base_path") {
      base_path = argv[++i];
    } else if (std::string(argv[i]) == "--beta") {
      beta = argv[++i];
    } else if (std::string(argv[i]) == "--velocity") {
      velocity = argv[++i];
    } else if (std::string(argv[i]) == "--lattice_size") {
      lattice_size = argv[++i];
    } else if (std::string(argv[i]) == "--boundary") {
      boundary = argv[++i];
    } else if (std::string(argv[i]) == "--result_path") {
      result_path = argv[++i];
    } else if (std::string(argv[i]) == "--spec_additional_path") {
      spec_additional_path = argv[++i];
    }
  }
  std::cout << "base_path " << base_path << std::endl;
  std::cout << "beta " << beta << std::endl;
  std::cout << "velocity " << velocity << std::endl;
  std::cout << "lattice_size " << lattice_size << std::endl;
  std::cout << "boundary " << boundary << std::endl;
  std::cout << "result_path " << result_path << std::endl;
  std::cout << "spec_additional_path " << spec_additional_path << std::endl;

  double start_time;
  double end_time;
  double search_time;

  std::stringstream ss;
  ss << base_path << "/" << lattice_size << "/" << boundary << "/" << velocity
     << "/" << beta;
  std::string dir_path = ss.str();

  hmdf::ThreadGranularity::set_optimum_thread_level();

  std::tuple<int, int, int> lattice_sizes = get_lattice_sizes(lattice_size);
  int Ns = std::get<2>(lattice_sizes);
  int coord_max = Ns / 2;

  start_time = omp_get_wtime();
  int block_size = 1;
  hmdf::StdDataFrame<unsigned long> df =
      read_data_polyakov(dir_path, Ns * Ns, block_size);
  end_time = omp_get_wtime();
  search_time = end_time - start_time;
  std::cout << "read time: " << search_time << std::endl;

  if (!df.empty()) {
    int thermalization_length =
        get_thermalization_length(base_path, spec_additional_path, lattice_size,
                                  boundary, velocity, beta);
    std::cout << "thermalization_length: " << thermalization_length
              << std::endl;
    auto functor_less = [thermalization_length](const unsigned long &,
                                                const int &conf_last) -> bool {
      return conf_last > thermalization_length;
    };

    int bin_size =
        get_bin_length_polyakov(base_path, spec_additional_path, lattice_size,
                                boundary, velocity, beta) /
        block_size;
    if (bin_size == 0) {
      bin_size = 1;
    }

    start_time = omp_get_wtime();
    df = df.get_data_by_sel<int, decltype(functor_less), int, double>(
        "conf_end", functor_less);
    end_time = omp_get_wtime();
    search_time = end_time - start_time;
    std::cout << "remove thermalization time: " << search_time << std::endl;

    hmdf::CountVisitor<int> count_visitor;
    int df_len =
        df.single_act_visit<int>("conf_end", count_visitor).get_result();
    if (df_len / Ns / Ns >= 3 * bin_size) {

      df.remove_column<double>("sqL");

      RadiusSquaredVisitor<int> rad_squared_visitor(coord_max);
      df.single_act_visit<int, int>("x", "y", rad_squared_visitor);
      df.load_result_as_column(rad_squared_visitor, "rad_sqared");
      std::vector<int> radii_sq;
      std::vector<std::vector<double>> data_aver;
      hmdf::StdDataFrame<unsigned long> df1;
      hmdf::StdDataFrame<unsigned long> df_aver;
      std::map<std::tuple<int, int, double>,
               std::vector<std::tuple<double, double>>>
          result;
      end_time = omp_get_wtime();
      for (int cut = 0; cut < coord_max - 2; cut++) {
        // for (int cut = 0; cut < 2; cut++) {
        // start_time = omp_get_wtime();
        auto functor_cut = [coord_max, cut](const unsigned long &, const int &x,
                                            const int &y) -> bool {
          return (x <= 2 * coord_max - cut) && (x >= cut) &&
                 (y <= 2 * coord_max - cut) && (y >= cut);
        };
        df = df.get_data_by_sel<int, int, decltype(functor_cut), int, double>(
            "x", "y", functor_cut);
        // end_time = omp_get_wtime();
        // search_time = end_time - start_time;
        // std::cout << "box cut time: " << search_time << std::endl;

        radii_sq = get_radii_sq(coord_max - cut);
        for (int rad_cut : radii_sq) {
          // std::cout << "rad_cut: " << rad_cut << std::endl;
          // start_time = omp_get_wtime();
          auto functor_rad_cut = [rad_cut](const unsigned long &,
                                           const int &rad_sq) -> bool {
            return rad_sq <= rad_cut;
          };
          df1 = df.get_data_by_sel<int, decltype(functor_rad_cut), int, double>(
              "rad_sqared", functor_rad_cut);
          // end_time = omp_get_wtime();
          // search_time = end_time - start_time;
          // std::cout << "radii cut time: " << search_time << std::endl;

          // start_time = omp_get_wtime();
          data_aver = observables_aver_polyakov(df1);
          // end_time = omp_get_wtime();
          // search_time = end_time - start_time;
          // std::cout << "observables_aver time: " << search_time << std::endl;

          // std::cout.precision(10);
          // start_time = omp_get_wtime();
          std::map<std::string, std::tuple<double, double>> jackknife_aver =
              jackknife_polyakov(data_aver, bin_size);
          add_to_map(result, jackknife_aver, bin_size * block_size,
                     coord_max - cut, sqrt(rad_cut));
          // for (auto &res : jackknife_aver) {
          //   std::cout << res.first << " " << std::get<0>(res.second) << " "
          //             << std::get<1>(res.second) << std::endl;
          // }
          // end_time = omp_get_wtime();
          // search_time = end_time - start_time;
          // std::cout << "jackknife time: " << search_time << std::endl;
        }
      }
      end_time = omp_get_wtime();
      search_time = end_time - start_time;
      std::cout << "data processing time: " << search_time << std::endl;
      try {
        std::filesystem::create_directories(result_path);
      } catch (...) {
      }

      std::ofstream stream_result;
      stream_result.open(result_path + "/polyakov_result.csv");
      stream_result.precision(17);
      stream_result << "ReL ReL_err ImL ImL_err <|L|> <|L|>_err |<L>| "
                       "|<L>|_err bin_size box_size radius"
                    << std::endl;

      for (auto &res : result) {
        for (int i = 0; i < 4; i++) {
          stream_result << std::get<0>(res.second[i]) << " "
                        << std::get<1>(res.second[i]) << " ";
        }
        stream_result << get<0>(res.first) << " " << get<1>(res.first) << " "
                      << get<2>(res.first) << std::endl;
      }
      stream_result.close();
    }
  }
}