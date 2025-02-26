#include "make_observables.h"
#include "visitors.h"

#include <DataFrame/DataFrame.h>

void make_Jv(hmdf::StdDataFrame<unsigned long> &df) {
  JvVisitor1 Jv_visitor1;
  df.single_act_visit<double, double, double, double, double>(
      "18", "14", "15", "16", "5", Jv_visitor1);
  JvVisitor2 Jv_visitor2(Jv_visitor1.get_result());
  df.single_act_visit<double, double>("6", "7", Jv_visitor2);
  df.load_result_as_column(Jv_visitor2, "Jv2");
  Jv_visitor1 = JvVisitor1();
  df.single_act_visit<double, double, double, double, double>(
      "18", "14", "15", "16", "5", Jv_visitor1);
  Jv_visitor2 = JvVisitor2(Jv_visitor1.get_result());
  df.single_act_visit<double, double>("6", "7", Jv_visitor2);
  JvVisitor3 Jv_visitor3(Jv_visitor2.get_result());
  df.single_act_visit<double>("17", Jv_visitor3);
  df.load_result_as_column(Jv_visitor3, "Jv");
  NegativeVisitor negative_visitor;
  df.single_act_visit<double>("17", negative_visitor);
  df.load_result_as_column(negative_visitor, "Jv1");
}

void make_Blab(hmdf::StdDataFrame<unsigned long> &df) {
  Sum5Visitor sum5_visitor;
  df.single_act_visit<double, double, double, double, double>(
      "5", "6", "7", "11", "12", sum5_visitor);
  AddVisitor add_visitor(sum5_visitor.get_result());
  df.single_act_visit<double>("13", add_visitor);
  df.load_result_as_column(add_visitor, "Blab");
}

void make_E(hmdf::StdDataFrame<unsigned long> &df) {
  Sum5Visitor sum5_visitor;
  df.single_act_visit<double, double, double, double, double>(
      "2", "3", "4", "8", "9", sum5_visitor);
  AddVisitor add_visitor(sum5_visitor.get_result());
  df.single_act_visit<double>("10", add_visitor);
  df.load_result_as_column(add_visitor, "E");
}

void make_Elab(hmdf::StdDataFrame<unsigned long> &df) {
  Sum5Visitor sum5_visitor;
  df.single_act_visit<double, double, double, double, double>(
      "2", "3", "4", "8", "9", sum5_visitor);
  Add5Visitor add5_visitor(sum5_visitor.get_result());
  df.single_act_visit<double, double, double, double, double>(
      "10", "17", "18", "14", "15", add5_visitor);
  ElabFinalVisitor Elab_final_visitor(add5_visitor.get_result());
  df.single_act_visit<double, double, double, double>("16", "5", "6", "7",
                                                      Elab_final_visitor);
  df.load_result_as_column(Elab_final_visitor, "Elab");
}

void make_Bz(hmdf::StdDataFrame<unsigned long> &df) {
  Sum2Visitor sum2_visitor;
  df.single_act_visit<double, double>("7", "13", sum2_visitor);
  df.load_result_as_column(sum2_visitor, "Bz");
}

void make_Bxy(hmdf::StdDataFrame<unsigned long> &df) {
  Sum4DivideVisitor sum4_divide_visitor(2);
  df.single_act_visit<double, double, double, double>("5", "6", "11", "12",
                                                      sum4_divide_visitor);
  df.load_result_as_column(sum4_divide_visitor, "Bxy");
}

void make_Ez(hmdf::StdDataFrame<unsigned long> &df) {
  Sum2Visitor sum2_visitor;
  df.single_act_visit<double, double>("4", "10", sum2_visitor);
  df.load_result_as_column(sum2_visitor, "Ez");
}

void make_Exy(hmdf::StdDataFrame<unsigned long> &df) {
  Sum4DivideVisitor sum4_divide_visitor(2);
  df.single_act_visit<double, double, double, double>("2", "3", "8", "9",
                                                      sum4_divide_visitor);
  df.load_result_as_column(sum4_divide_visitor, "Exy");
}

void make_ElabzT(hmdf::StdDataFrame<unsigned long> &df) {
  Sum5Visitor sum5_visitor;
  df.single_act_visit<double, double, double, double, double>(
      "4", "10", "14", "15", "18", sum5_visitor);
  ElabzTFinalVisitor ElabzT_final_visitor(sum5_visitor.get_result());
  df.single_act_visit<double, double>("5", "6", ElabzT_final_visitor);
  df.load_result_as_column(ElabzT_final_visitor, "ElabzT");
}

void make_ElabxyT(hmdf::StdDataFrame<unsigned long> &df) {
  Sum5Visitor sum5_visitor;
  df.single_act_visit<double, double, double, double, double>(
      "2", "3", "8", "9", "16", sum5_visitor);
  ExtractDivideVisitor extract_divide_visitor(sum5_visitor.get_result(), 2);
  df.single_act_visit<double>("7", extract_divide_visitor);
  df.load_result_as_column(extract_divide_visitor, "ElabxyT");
}

void make_Ae(hmdf::StdDataFrame<unsigned long> &df) {
  Sum4DivideVisitor sum4_divide_visitor(2);
  df.single_act_visit<double, double, double, double>("2", "3", "8", "9",
                                                      sum4_divide_visitor);
  Extract2Visitor extract2_visitor(sum4_divide_visitor.get_result());
  df.single_act_visit<double, double>("4", "10", extract2_visitor);
  df.load_result_as_column(extract2_visitor, "Ae");
}

void make_Am(hmdf::StdDataFrame<unsigned long> &df) {
  Sum4DivideVisitor sum4_divide_visitor(2);
  df.single_act_visit<double, double, double, double>("5", "6", "11", "12",
                                                      sum4_divide_visitor);
  Extract2Visitor extract2_visitor(sum4_divide_visitor.get_result());
  df.single_act_visit<double, double>("7", "13", extract2_visitor);
  df.load_result_as_column(extract2_visitor, "Am");
}

void make_AlabeT(hmdf::StdDataFrame<unsigned long> &df) {
  Sum5Visitor sum5_visitor;
  df.single_act_visit<double, double, double, double, double>(
      "2", "3", "8", "9", "16", sum5_visitor);
  ExtractDivideVisitor extract_divide_visitor(sum5_visitor.get_result(), 2);
  df.single_act_visit<double>("7", extract_divide_visitor);
  Extract5Visitor extract5_visitor(extract_divide_visitor.get_result());
  df.single_act_visit<double, double, double, double, double>(
      "4", "10", "14", "15", "18", extract5_visitor);
  Add2Visitor add2_visitor(extract5_visitor.get_result());
  df.single_act_visit<double, double>("5", "6", add2_visitor);
  df.load_result_as_column(add2_visitor, "AlabeT");
}

void make_observables(hmdf::StdDataFrame<unsigned long> &df) {
  make_Jv(df);
  make_Blab(df);
  make_E(df);
  make_Elab(df);
  make_Bz(df);
  make_Bxy(df);
  make_Ez(df);
  make_Exy(df);
  make_ElabzT(df);
  make_ElabxyT(df);
  make_Ae(df);
  make_Am(df);
  make_AlabeT(df);
}

void make_S(std::vector<std::vector<double>> &data,
            std::vector<std::vector<double>> &observables_data) {
  for (int i = 0; i < data[0].size(); i++) {
    observables_data[0][i] = data[17][i];
  }
}
void make_Jv(std::vector<std::vector<double>> &data,
             std::vector<std::vector<double>> &observables_data) {
  for (int i = 0; i < data[0].size(); i++) {
    observables_data[1][i] =
        -data[15][i] - 2 * (data[16][i] + data[12][i] + data[13][i] +
                            data[14][i] - data[3][i] - data[4][i] - data[5][i]);
  }
}
void make_Jv1(std::vector<std::vector<double>> &data,
              std::vector<std::vector<double>> &observables_data) {
  for (int i = 0; i < data[0].size(); i++) {
    observables_data[2][i] = -data[15][i];
  }
}
void make_Jv2(std::vector<std::vector<double>> &data,
              std::vector<std::vector<double>> &observables_data) {
  for (int i = 0; i < data[0].size(); i++) {
    observables_data[3][i] =
        -2 * (data[16][i] + data[12][i] + data[13][i] + data[14][i] -
              data[3][i] - data[4][i] - data[5][i]);
  }
}
void make_Blab(std::vector<std::vector<double>> &data,
               std::vector<std::vector<double>> &observables_data) {
  for (int i = 0; i < data[0].size(); i++) {
    observables_data[4][i] = data[3][i] + data[4][i] + data[5][i] + data[9][i] +
                             data[10][i] + data[11][i];
  }
}
void make_E(std::vector<std::vector<double>> &data,
            std::vector<std::vector<double>> &observables_data) {
  for (int i = 0; i < data[0].size(); i++) {
    observables_data[5][i] = data[0][i] + data[1][i] + data[2][i] + data[6][i] +
                             data[7][i] + data[8][i];
  }
}
void make_Elab(std::vector<std::vector<double>> &data,
               std::vector<std::vector<double>> &observables_data) {
  for (int i = 0; i < data[0].size(); i++) {
    observables_data[6][i] = data[0][i] + data[1][i] + data[2][i] + data[6][i] +
                             data[7][i] + data[8][i] + data[15][i] +
                             data[16][i] + data[12][i] + data[13][i] +
                             data[14][i] - data[3][i] - data[4][i] - data[5][i];
  }
}
void make_Bz(std::vector<std::vector<double>> &data,
             std::vector<std::vector<double>> &observables_data) {
  for (int i = 0; i < data[0].size(); i++) {
    observables_data[7][i] = data[5][i] + data[11][i];
  }
}
void make_Bxy(std::vector<std::vector<double>> &data,
              std::vector<std::vector<double>> &observables_data) {
  for (int i = 0; i < data[0].size(); i++) {
    observables_data[8][i] =
        (data[3][i] + data[4][i] + data[9][i] + data[10][i]) / 2;
  }
}
void make_Ez(std::vector<std::vector<double>> &data,
             std::vector<std::vector<double>> &observables_data) {
  for (int i = 0; i < data[0].size(); i++) {
    observables_data[9][i] = data[2][i] + data[8][i];
  }
}
void make_Exy(std::vector<std::vector<double>> &data,
              std::vector<std::vector<double>> &observables_data) {
  for (int i = 0; i < data[0].size(); i++) {
    observables_data[10][i] =
        (data[0][i] + data[1][i] + data[6][i] + data[7][i]) / 2;
  }
}
void make_ElabzT(std::vector<std::vector<double>> &data,
                 std::vector<std::vector<double>> &observables_data) {
  for (int i = 0; i < data[0].size(); i++) {
    observables_data[11][i] = data[2][i] + data[8][i] + data[12][i] +
                              data[13][i] - data[3][i] - data[4][i] +
                              data[16][i];
  }
}
void make_ElabxyT(std::vector<std::vector<double>> &data,
                  std::vector<std::vector<double>> &observables_data) {
  for (int i = 0; i < data[0].size(); i++) {
    observables_data[12][i] = (data[0][i] + data[1][i] + data[6][i] +
                               data[7][i] + data[14][i] - data[5][i]) /
                              2;
  }
}
void make_Ae(std::vector<std::vector<double>> &data,
             std::vector<std::vector<double>> &observables_data) {
  for (int i = 0; i < data[0].size(); i++) {
    observables_data[13][i] =
        (data[0][i] + data[1][i] + data[6][i] + data[7][i]) / 2 - data[2][i] -
        data[8][i];
  }
}
void make_Am(std::vector<std::vector<double>> &data,
             std::vector<std::vector<double>> &observables_data) {
  for (int i = 0; i < data[0].size(); i++) {
    observables_data[14][i] =
        (data[3][i] + data[4][i] + data[9][i] + data[10][i]) / 2 - data[5][i] -
        data[11][i];
  }
}
void make_AlabeT(std::vector<std::vector<double>> &data,
                 std::vector<std::vector<double>> &observables_data) {
  for (int i = 0; i < data[0].size(); i++) {
    observables_data[15][i] =
        (data[0][i] + data[1][i] + data[6][i] + data[7][i] + data[14][i] -
         data[5][i]) /
            2 -
        (data[2][i] + data[8][i] + data[12][i] + data[13][i] - data[3][i] -
         data[4][i] + data[16][i]);
  }
}

std::vector<std::vector<std::vector<std::vector<double>>>> make_observables(
    std::vector<std::vector<std::vector<std::vector<double>>>> &data, int Ns) {
  std::vector<std::vector<std::vector<std::vector<double>>>> observables_data(
      Ns, std::vector<std::vector<std::vector<double>>>(
              Ns, std::vector<std::vector<double>>(
                      16, std::vector<double>(data[0][0][0].size()))));
  for (int i = 0; i < Ns; i++) {
    for (int j = 0; j < Ns; j++) {
      make_S(data[i][j], observables_data[i][j]);
      make_Jv(data[i][j], observables_data[i][j]);
      make_Jv1(data[i][j], observables_data[i][j]);
      make_Jv2(data[i][j], observables_data[i][j]);
      make_Blab(data[i][j], observables_data[i][j]);
      make_E(data[i][j], observables_data[i][j]);
      make_Elab(data[i][j], observables_data[i][j]);
      make_Bz(data[i][j], observables_data[i][j]);
      make_Bxy(data[i][j], observables_data[i][j]);
      make_Ez(data[i][j], observables_data[i][j]);
      make_Exy(data[i][j], observables_data[i][j]);
      make_ElabzT(data[i][j], observables_data[i][j]);
      make_ElabxyT(data[i][j], observables_data[i][j]);
      make_Ae(data[i][j], observables_data[i][j]);
      make_Am(data[i][j], observables_data[i][j]);
      make_AlabeT(data[i][j], observables_data[i][j]);
    }
  }
  return observables_data;
}