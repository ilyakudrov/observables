#include <map>
#include <string>
#include <vector>

std::map<std::tuple<int, int>, std::vector<std::vector<double>>>
read_data(std::string dir_path, std::string file_start, std::string file_end,
          int padding, int num_max);

std::map<std::tuple<int, int>, std::vector<std::vector<double>>>
read_data_copies(std::string dir_path, std::string file_start,
                 std::string file_end, int padding, int num_max, int copy);

std::map<std::tuple<std::string, int, int>, double>
read_functional(std::string dir_path, int padding, int num_max);

std::map<std::tuple<int, int>, std::vector<std::vector<double>>>
read_data_bins(std::string dir_path, std::string file_start,
               std::string file_end, int padding,
               std::map<std::tuple<std::string, int, int>, double> &functional,
               double bin_left, double bin_right, double &functional_average);