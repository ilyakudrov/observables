#include <map>
#include <string>
#include <tuple>
#include <vector>

std::map<double, std::vector<std::tuple<std::string, int, int>>>
bin_functional(std::map<std::tuple<std::string, int, int>, double> &functional);