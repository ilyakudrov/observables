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