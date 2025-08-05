#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

std::map<std::tuple<int, int, int, int, int, int, int, int>,
         std::tuple<std::vector<double>, std::vector<double>>>
read_data(std::string dir_path, std::string file_start, std::string file_end,
          int padding, int num_max);
std::set<std::tuple<int, int, int, int>> read_momenta(std::string dir_path,
                                                      std::string file_start,
                                                      std::string file_end,
                                                      int padding, int num_max);
std::map<std::tuple<int, int, int, int>,
         std::tuple<std::vector<double>, std::vector<double>>>
read_data_single_momentum(std::tuple<int, int, int, int> &momentum,
                          std::string dir_path, std::string file_start,
                          std::string file_end, int padding, int num_max);