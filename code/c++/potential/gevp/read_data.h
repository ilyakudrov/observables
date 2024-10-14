#include <map>
#include <string>
#include <vector>

std::map<std::tuple<int, int>, std::vector<std::vector<double>>>
read_data(std::string dir_path, std::string file_start, std::string file_end,
          int padding, int num_max, int smearing_max);