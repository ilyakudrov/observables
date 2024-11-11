#include <map>
#include <vector>

std::map<std::tuple<int, int>, std::tuple<double, double>> calculate_potential(
    std::map<std::tuple<int, int>, std::vector<std::vector<double>>> &data,
    int bin_size, int t0);