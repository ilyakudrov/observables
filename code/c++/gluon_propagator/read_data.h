#pragma once

#include <map>
#include <string>
#include <vector>

std::map<std::tuple<int, int, int, int, int, int, int, int>,
         std::tuple<std::vector<double>, std::vector<double>>>
read_data(std::string dir_path, std::string file_start, std::string file_end,
          int padding, int num_max);