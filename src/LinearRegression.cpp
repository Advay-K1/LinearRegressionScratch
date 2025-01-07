#include <iostream>
#include "/Users/advaykadam/LinearRegressionScratch/includes/LinearRegression.hpp"

LinearRegression::LinearRegression(
    const xt::xtensor<double, 2>& predictor_matrix_train,
    const xt::xtensor<double, 1>& y_vector_train,
    const xt::xtensor<double, 2>& predictor_matrix_valid,
    const xt::xtensor<double, 1>& y_vector_valid
) : design_matrix(predictor_matrix_train), 
    y_vector(y_vector_train), 
    x_valid_matrix(predictor_matrix_valid), 
    y_valid_vector(y_vector_valid),
    response_vector(xt::zeros<double>({predictor_matrix_train.shape(0)})),
    bias_vector(xt::zeros<double>({predictor_matrix_train.shape(1)})),
    error_vector(xt::zeros<double>({predictor_matrix_train.shape(0)})) {}


double LinearRegression::RSS() {    
    auto residual = y_vector - response_vector;

    double RSS = xt::sum(residual * residual)();

    return RSS;

}

double LinearRegression::TSS() {

    auto deviations = y_vector - xt::mean(y_vector);;

    double TSS = xt::sum(deviations * deviations)();

    return TSS;   
}

double LinearRegression::R_squared() {
    return (1 - (RSS()/ TSS()));
}

void LinearRegression::calculate_response_vector() {
    response_vector = xt::linalg::dot(design_matrix, bias_vector);
}

void LinearRegression::compute_bias_vector() {
    auto X_transpose = xt::transpose(design_matrix);
    auto X_transpose_X = xt::linalg::dot(X_transpose, design_matrix);
    auto X_transpose_X_inv = xt::linalg::inv(X_transpose_X);
    auto X_transpose_y = xt::linalg::dot(X_transpose, y_vector);

    bias_vector = xt::linalg::dot(X_transpose_X_inv, X_transpose_y);
}

void LinearRegression::compute_error_vector() {
    error_vector = y_vector - response_vector;
}


void LinearRegression::gradient_descent() {

}

void LinearRegression::update_bias_vector() {

}
