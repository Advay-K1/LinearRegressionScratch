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


LinearRegression::LinearRegression(
    const xt::xtensor<double, 2>& predictor_matrix_train,
    const xt::xtensor<double, 1>& y_vector_train
) : design_matrix(predictor_matrix_train), 
    y_vector(y_vector_train), 
    x_valid_matrix(xt::zeros<double>({predictor_matrix_train.shape(0)})), 
    y_valid_vector(xt::zeros<double>({predictor_matrix_train.shape(0)})),
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


 xt::xtensor<double, 1> LinearRegression::batch_gradient_descent(const double& learning_rate, const int& max_iterations) {

    xt::xtensor<double, 1> rss_history = xt::zeros<double>({max_iterations});

    for (int i =0; i < max_iterations; ++i) {

        calculate_response_vector();
        compute_error_vector();

        auto gradient = -2 * xt::linalg::dot(xt::transpose(design_matrix), error_vector);

        update_bias_vector(gradient, learning_rate);
        
        double curr_loss = RSS();

        std::cout << "Epoch: " << i + 1 << "| Loss (RSS): " << curr_loss << std::endl;

        rss_history(i) = curr_loss;

        if (i > 0 && std::abs(rss_history(i) - rss_history(i - 1)) < 1e-6) {
            rss_history = xt::view(rss_history, xt::range(0, i + 1));
            break;
        }

    }


    return rss_history;

}

void LinearRegression::update_bias_vector(const xt::xtensor<double, 1>& gradient, const double& learning_rate) {
    bias_vector -= learning_rate * gradient;
}


void LinearRegression::fit_model(const double& learning_rate, const int& max_iterations) {
    auto rss_history = batch_gradient_descent(learning_rate, max_iterations);

    double train_loss = RSS();
    std::cout << "Final Training Loss (RSS): " << train_loss << std::endl;

    xt::xtensor<double, 1> test_predictions = xt::linalg::dot(x_valid_matrix, bias_vector);
    double test_loss = xt::sum(xt::pow(y_valid_vector - test_predictions, 2))();
    std::cout << "Validation Loss (RSS): " << test_loss << std::endl;

}

xt::xtensor<double, 1> LinearRegression::get_bias_vector() {
    return bias_vector;
}

xt::xtensor<double, 1> LinearRegression::get_error_vector() {
    return error_vector;
}


xt::xtensor<double, 2> LinearRegression::get_response_vector() {
    return response_vector;
}