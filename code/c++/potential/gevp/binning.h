#include <map>
#include <string>
#include <tuple>
#include <vector>

std::vector<double>
get_bin_edges(std::map<std::tuple<std::string, int, int>, double> &functional);
std::map<double, std::vector<std::tuple<std::string, int, int>>>
bin_functional(std::map<std::tuple<std::string, int, int>, double> &functional);