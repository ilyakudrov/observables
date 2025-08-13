#include "jackknife.h"
#include "read_data.h"

#include "fstream"
#include "iostream"
#include "vector"

using namespace std;

int main(int argc, char *argv[]) {
  string dir_path;
  string file_start;
  string file_end;
  string output_path;
  int padding;
  int num_max;
  for (int i = 1; i < argc; i++) {
    if (string(argv[i]) == "--dir_path") {
      dir_path = argv[++i];
    } else if (string(argv[i]) == "--file_start") {
      file_start = argv[++i];
    } else if (string(argv[i]) == "--file_end") {
      file_end = argv[++i];
    } else if (string(argv[i]) == "--output_path") {
      output_path = argv[++i];
    } else if (string(argv[i]) == "--padding") {
      padding = stoi(string(argv[++i]));
    } else if (string(argv[i]) == "--num_max") {
      num_max = stoi(string(argv[++i]));
    }
  }

  cout << "dir_path " << dir_path << endl;
  cout << "file_start " << file_start << endl;
  cout << "file_end " << file_end << endl;
  cout << "output_path " << output_path << endl;
  cout << "padding " << padding << endl;
  cout << "num_max " << num_max << endl;

  std::map<std::tuple<double, double, double, double, int>,
           std::tuple<std::vector<double>, std::vector<double>>>
      data =
          read_data_diagonal(dir_path, file_start, file_end, padding, num_max);
  if (!data.empty()) {
    int data_size = (unsigned long)get<0>(data.begin()->second).size();
    std::vector<unsigned long> bin_borders = get_bin_borders(data_size, 1);
    vector<double> jackknife_data;
    std::map<std::tuple<double, double, double, double, int>,
             std::tuple<std::tuple<double, double>, std::tuple<double, double>>>
        result;
    std::map<std::tuple<double, double, double, double, int>,
             std::vector<std::tuple<std::vector<double>, std::vector<double>>>>
        data_tmp;
    std::map<std::tuple<double, double, double, double, int>,
             std::tuple<std::vector<double>, std::vector<double>>>
        data_space_average;

    for (auto pair : data) {
      if (std::get<4>(pair.first) == 3) {
        data_tmp[{std::get<0>(pair.first), std::get<1>(pair.first),
                  std::get<2>(pair.first), std::get<3>(pair.first), 1}]
            .push_back(data[pair.first]);
      } else {
        data_tmp[{std::get<0>(pair.first), std::get<1>(pair.first),
                  std::get<2>(pair.first), std::get<3>(pair.first), 0}]
            .push_back(data[pair.first]);
      }
    }
    std::vector<double> vec_tmp1, vec_tmp2;
    for (auto pair : data_tmp) {
      vec_tmp1 = std::vector<double>(std::get<0>(pair.second[0]).size());
      vec_tmp2 = std::vector<double>(std::get<1>(pair.second[0]).size());
      for (int i = 0; i < pair.second.size(); i++) {
        for (int j = 0; j < vec_tmp1.size(); j++) {
          vec_tmp1[j] += std::get<0>(pair.second[i])[j] / pair.second.size();
        }
        for (int j = 0; j < vec_tmp2.size(); j++) {
          vec_tmp2[j] += std::get<1>(pair.second[i])[j] / pair.second.size();
        }
      }
      data_space_average[pair.first] = {vec_tmp1, vec_tmp2};
    }
    std::tuple<double, double> aver_real, aver_imag;
    for (auto pair : data_space_average) {
      jackknife_data = do_jackknife(get<0>(pair.second), bin_borders);
      aver_real = get_aver(jackknife_data);
      jackknife_data = do_jackknife(get<1>(pair.second), bin_borders);
      aver_imag = get_aver(jackknife_data);
      result[pair.first] = {aver_real, aver_imag};
    }
    ofstream stream_propagator;
    stream_propagator.precision(17);
    stream_propagator.open(output_path);
    stream_propagator << "p1,p2,p3,p4,mu,Dr,Dr_err,Di,Di_err" << endl;
    for (auto const &[key, value] : result) {
      stream_propagator << get<0>(key) << "," << get<1>(key) << ","
                        << get<2>(key) << "," << get<3>(key) << ","
                        << get<4>(key) << "," << get<0>(get<0>(value)) << ","
                        << get<1>(get<0>(value)) << "," << get<0>(get<1>(value))
                        << "," << get<1>(get<1>(value)) << endl;
    }
    stream_propagator.close();
  } else {
    cout << "data is empty" << endl;
  }
}
