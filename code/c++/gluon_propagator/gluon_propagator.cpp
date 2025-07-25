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

  std::map<std::tuple<int, int, int, int, int, int, int, int>,
           std::tuple<std::vector<double>, std::vector<double>>>
      data;
  data = read_data(dir_path, file_start, file_end, padding, num_max);
  if (!data.empty()) {
    int data_size = (unsigned long)get<0>(data.begin()->second).size();
    std::vector<unsigned long> bin_borders = get_bin_borders(data_size, 1);
    vector<double> jackknife_data;
    std::map<std::tuple<int, int, int, int, int, int, int, int>,
             std::tuple<std::tuple<double, double>, std::tuple<double, double>>>
        result;
    std::tuple<double, double> aver_real, aver_imag;
    for (auto pair : data) {
      jackknife_data = do_jackknife(get<0>(pair.second), bin_borders);
      aver_real = get_aver(jackknife_data);
      jackknife_data = do_jackknife(get<1>(pair.second), bin_borders);
      aver_imag = get_aver(jackknife_data);
      result[pair.first] = {aver_real, aver_imag};
    }
    ofstream stream_propagator;
    stream_propagator.precision(17);
    stream_propagator.open(output_path);
    stream_propagator << "p1,p2,p3,p4,mu,nu,a,b,Dr,Dr_err,Di,Di_err" << endl;
    for (auto const &[key, value] : result) {
      stream_propagator << get<0>(key) << "," << get<1>(key) << ","
                        << get<2>(key) << "," << get<3>(key) << ","
                        << get<4>(key) << "," << get<5>(key) << ","
                        << get<6>(key) << "," << get<7>(key) << ","
                        << get<0>(get<0>(value)) << "," << get<1>(get<0>(value))
                        << "," << get<0>(get<1>(value)) << ","
                        << get<1>(get<1>(value)) << endl;
    }
    stream_propagator.close();
  } else {
    cout << "data is empty" << endl;
  }
}
