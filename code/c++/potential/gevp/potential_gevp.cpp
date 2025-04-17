#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <string>
#include <sys/stat.h>

#include "jackknife.h"
#include "read_data.h"

using Eigen::MatrixXd;
using namespace std;

int main(int argc, char *argv[]) {
  string dir_path;
  string file_start;
  string file_end;
  string output_path;
  int padding;
  int num_max;
  int smearing_max;
  int t0;
  bool copies;
  int copy;
  for (int i = 1; i < argc; i++) {
    if (string(argv[i]) == "-dir_path") {
      dir_path = argv[++i];
    } else if (string(argv[i]) == "-file_start") {
      file_start = argv[++i];
    } else if (string(argv[i]) == "-file_end") {
      file_end = argv[++i];
    } else if (string(argv[i]) == "-output_path") {
      output_path = argv[++i];
    } else if (string(argv[i]) == "-padding") {
      padding = stoi(string(argv[++i]));
    } else if (string(argv[i]) == "-num_max") {
      num_max = stoi(string(argv[++i]));
    } else if (string(argv[i]) == "-smearing_max") {
      smearing_max = stoi(string(argv[++i]));
    } else if (string(argv[i]) == "-t0") {
      t0 = stoi(string(argv[++i]));
    } else if (string(argv[i]) == "-copies") {
      istringstream(string(argv[++i])) >> copies;
    } else if (string(argv[i]) == "-copy") {
      istringstream(string(argv[++i])) >> copy;
    }
  }
  cout << "dir_path " << dir_path << endl;
  cout << "file_start " << file_start << endl;
  cout << "file_end " << file_end << endl;
  cout << "output_path " << output_path << endl;
  cout << "padding " << padding << endl;
  cout << "num_max " << num_max << endl;
  cout << "smearing_max " << smearing_max << endl;
  cout << "t0 " << t0 << endl;
  cout << "copies " << copies << endl;
  cout << "copy " << copy << endl;

  Eigen::Matrix3cd A;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      A(i, j) = std::complex<double>(1. * i / 2, 1. * j / 3);
    }
  }
  cout << A << endl;

  std::map<std::tuple<int, int>, std::vector<std::vector<double>>> data;

  if (copies) {
    data = read_data_copies(dir_path, file_start, file_end, padding, num_max,
                            smearing_max, copy);
  } else {
    data = read_data(dir_path, file_start, file_end, padding, num_max,
                     smearing_max);
  }
  // std::cout << data.size() << std::endl;
  std::map<std::tuple<int, int>, std::tuple<double, double>> potential =
      calculate_potential(data, 1, t0);
  for (auto const &[key, value] : potential) {
    cout << get<0>(key) << ", " << get<1>(key) << ": " << get<0>(value)
         << " +- " << get<1>(value) << endl;
  }
  ofstream stream_potential;
  stream_potential.precision(17);
  stream_potential.open(output_path);
  stream_potential << "space_size,time_size,potential,err" << endl;
  for (auto const &[key, value] : potential) {
    stream_potential << get<0>(key) << "," << get<1>(key) << ","
                     << get<0>(value) << "," << get<1>(value) << endl;
  }
  stream_potential.close();
}
