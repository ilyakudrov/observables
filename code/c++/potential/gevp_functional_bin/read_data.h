#include <map>
#include <string>
#include <vector>

std::map<std::tuple<int, int>, std::vector<std::vector<double>>>
read_data(std::string dir_path,
          const std::vector<std::tuple<std::string, int, int>> &confs,
          std::string file_start, std::string file_end, int padding,
          int smearing_max);

std::map<std::tuple<std::string, int, int>, double>
read_functional(std::string dir_path, int padding, int num_max);