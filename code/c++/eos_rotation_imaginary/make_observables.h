#include <DataFrame/DataFrame.h>

void make_observables(hmdf::StdDataFrame<unsigned long> &df);
std::vector<std::vector<std::vector<std::vector<double>>>> make_observables(
    std::vector<std::vector<std::vector<std::vector<double>>>> &data, int Ns);