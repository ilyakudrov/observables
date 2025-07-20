#include "jackknife.h"
#include "read_data.h"

#include <fstream>
#include <iostream>
#include <vector>

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

  std::map<int, std::vector<double>> data;
  data = read_data(dir_path, file_start, file_end, padding, num_max);
  if (!data.empty()) {
    int data_size = data.begin()->second.size();
    std::vector<unsigned long> bin_borders = get_bin_borders(data_size, 1);
    std::vector<vector<double>> jackknife_data;
    std::map<int,
             std::tuple<std::tuple<double, double>, std::tuple<double, double>>>
        result;
    std::tuple<double, double> aver_polyakov_loop, aver_susceptibility;
    for (auto pair : data) {
      jackknife_data = do_jackknife(pair.second, bin_borders);
      aver_polyakov_loop = get_aver(jackknife_data[0]);
      aver_susceptibility = get_aver(jackknife_data[1]);
      result[pair.first] = {aver_polyakov_loop, aver_susceptibility};
    }
    ofstream stream_polyakov_loop;
    stream_polyakov_loop.precision(17);
    stream_polyakov_loop.open(output_path);
    stream_polyakov_loop << "HYP_step,polyakov_loop,polyakov_loop_err,"
                            "susceptibility,susceptibility_err"
                         << endl;
    for (auto const &[key, value] : result) {
      stream_polyakov_loop << key << "," << get<0>(get<0>(value)) << ","
                           << get<1>(get<0>(value)) << ","
                           << get<0>(get<1>(value)) << ","
                           << get<1>(get<1>(value)) << endl;
    }
    stream_polyakov_loop.close();
  } else {
    cout << "data is empty" << endl;
  }
}
