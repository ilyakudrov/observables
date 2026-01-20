#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <string>
#include <sys/stat.h>

#include "binning.h"
#include "jackknife.h"
#include "read_data.h"

using Eigen::MatrixXd;
using namespace std;

int main(int argc, char *argv[]) {
  string dir_path;
  string file_start;
  string file_end;
  string output_path;
  string functional_path;
  int padding_conf;
  int padding_functional;
  int num_max;
  int t0;
  for (int i = 1; i < argc; i++) {
    if (string(argv[i]) == "-dir_path") {
      dir_path = argv[++i];
    } else if (string(argv[i]) == "-file_start") {
      file_start = argv[++i];
    } else if (string(argv[i]) == "-file_end") {
      file_end = argv[++i];
    } else if (string(argv[i]) == "-output_path") {
      output_path = argv[++i];
    } else if (string(argv[i]) == "-functional_path") {
      functional_path = argv[++i];
    } else if (string(argv[i]) == "-padding_conf") {
      padding_conf = stoi(string(argv[++i]));
    } else if (string(argv[i]) == "-padding_functional") {
      padding_functional = stoi(string(argv[++i]));
    } else if (string(argv[i]) == "-num_max") {
      num_max = stoi(string(argv[++i]));
    } else if (string(argv[i]) == "-t0") {
      t0 = stoi(string(argv[++i]));
    }
  }
  cout << "dir_path " << dir_path << endl;
  cout << "file_start " << file_start << endl;
  cout << "file_end " << file_end << endl;
  cout << "output_path " << output_path << endl;
  cout << "functional_path " << functional_path << endl;
  cout << "padding_conf " << padding_conf << endl;
  cout << "padding_functional " << padding_functional << endl;
  cout << "num_max " << num_max << endl;
  cout << "t0 " << t0 << endl;

  std::map<std::tuple<int, int>, std::vector<std::vector<double>>> data;
  std::map<std::tuple<std::string, int, int>, double> functional =
      read_functional(functional_path, padding_functional, num_max);
  std::vector<double> functional_bins = get_bin_edges(functional);
  double functional_average;
  std::map<std::tuple<int, int, double>, std::tuple<double, double>> result;
  for (int i = 0; i < functional_bins.size() - 1; i++) {
    data = read_data_bins(dir_path, file_start, file_end, padding_conf,
                          functional, functional_bins[i],
                          functional_bins[i + 1], functional_average);
    if (!data.empty()) {
      if (data.begin()->second[0].size() >= 10) {
        std::map<std::tuple<int, int>, std::tuple<double, double>> potential =
            calculate_potential(data, 1, t0);
        for (const auto &pair : potential) {
          result[{std::get<0>(pair.first), std::get<1>(pair.first),
                  functional_average}] = pair.second;
        }
      }
    }
  }
  ofstream stream_potential;
  stream_potential.precision(17);
  stream_potential.open(output_path);
  stream_potential << "space_size,time_size,functional,potential,err" << endl;
  for (auto const &[key, value] : result) {
    stream_potential << get<0>(key) << "," << get<1>(key) << "," << get<2>(key)
                     << "," << get<0>(value) << "," << get<1>(value) << endl;
  }
  stream_potential.close();
}
