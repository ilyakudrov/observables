#pragma once

#include <map>
#include <string>
#include <vector>

std::map<int, std::vector<double>> read_data(std::string dir_path,
                                             std::string file_start,
                                             std::string file_end, int padding,
                                             int num_max);