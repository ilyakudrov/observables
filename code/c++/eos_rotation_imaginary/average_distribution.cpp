#include "data_processing.h"
#include "make_observables.h"
#include "read_data.h"

#include "omp.h"
#include <filesystem>
#include <fstream>
#include <map>
#include <tuple>

void add_to_map(
    std::map<std::tuple<int, int, int, int>,
             std::vector<std::tuple<double, double>>> &result,
    std::map<std::string, std::tuple<double, double>> &jackknife_aver, int x,
    int y, int bin_size, int thermalization_length) {
  std::vector<std::string> observables = {
      "S",   "Jv", "Jv1", "Jv2",    "Blab",    "E",  "Elab", "Bz",
      "Bxy", "Ez", "Exy", "ElabzT", "ElabxyT", "Ae", "Am",   "AlabeT"};
  std::vector<std::tuple<double, double>> tmp(16);
  for (int i = 0; i < 16; i++) {
    tmp[i] = std::make_tuple(std::get<0>(jackknife_aver[observables[i]]),
                             std::get<1>(jackknife_aver[observables[i]]));
  }
  result[std::make_tuple(x, y, bin_size, thermalization_length)] = tmp;
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

  std::tuple<int, int, int> lattice_sizes = get_lattice_sizes(lattice_size);
  int Ns = std::get<2>(lattice_sizes);
  int coord_max = Ns / 2;

  int thermalization_length = get_thermalization_length(
      base_path, spec_additional_path, lattice_size, boundary, velocity, beta);
  std::cout << "thermalization_length: " << thermalization_length << std::endl;

  start_time = omp_get_wtime();
  int block_size = 1;
  std::vector<std::vector<std::vector<std::vector<double>>>> data =
      read_data_to_vector(dir_path, Ns, block_size, thermalization_length);
  end_time = omp_get_wtime();
  search_time = end_time - start_time;
  std::cout << "read time: " << search_time << std::endl;

  int bin_size = get_bin_length(base_path, spec_additional_path, lattice_size,
                                boundary, velocity, beta) /
                 block_size;
  if (bin_size == 0) {
    bin_size = 1;
  }

  if (data[0][0][0].size() > 3 * bin_size) {
    start_time = omp_get_wtime();
    data = make_observables(data, Ns);
    end_time = omp_get_wtime();
    search_time = end_time - start_time;
    std::cout << "make_observables time: " << search_time << std::endl;

    std::map<std::string, std::tuple<double, double>> jackknife_aver;
    std::map<std::tuple<int, int, int, int>,
             std::vector<std::tuple<double, double>>>
        result;
    for (int i = 0; i < Ns; i++) {
      for (int j = 0; j < Ns; j++) {
        jackknife_aver = jackknife(data[i][j], bin_size);
        add_to_map(result, jackknife_aver, i, j, bin_size * block_size,
                   thermalization_length);
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
    stream_result.open(result_path + "/observables_distribution.csv");
    stream_result.precision(17);
    stream_result
        << "x y S S_err Jv Jv_err Jv1 Jv1_err Jv2 Jv2_err Blab Blab_err E "
           "E_err "
           "Elab Elab_err Bz Bz_err Bxy Bxy_err Ez Ez_err Exy Exy_err ElabzT "
           "ElabzT_err ElabxyT ElabxyT_err Ae Ae_err Am Am_err AlabeT "
           "AlabeT_err bin_size thermalization_length"
        << std::endl;

    for (auto &res : result) {
      stream_result << get<0>(res.first) << " " << get<1>(res.first) << " ";
      for (int i = 0; i < 16; i++) {
        stream_result << std::get<0>(res.second[i]) << " "
                      << std::get<1>(res.second[i]) << " ";
      }
      stream_result << get<2>(res.first) << " " << get<3>(res.first)
                    << std::endl;
    }
    stream_result.close();
  }
}