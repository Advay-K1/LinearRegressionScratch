#include "/Users/advaykadam/LinearRegressionScratch/includes/LinearRegression.hpp"
#include <xtensor/xrandom.hpp>


std::pair<xt::xtensor<double, 2>, xt::xtensor<double, 1>> generate_linear_data(int num_samples, int num_features, double noise_level = 0.1) {
    xt::xtensor<double, 2> design_matrix = xt::random::rand<double>({num_samples, num_features}, -10.0, 10.0);

    xt::xtensor<double, 1> true_weights = xt::random::rand<double>({num_features}, -10.0, 10.0);

    xt::xtensor<double, 1> noise = xt::random::randn<double>({num_samples}) * noise_level;
    
    xt::xtensor<double, 1> target_values = xt::linalg::dot(design_matrix, true_weights) + noise;

    return {design_matrix, target_values};
}

int main() {

    int train_samples = 100;
    int valid_samples = 30;
    int num_features = 5;

    auto [train_matrix, train_labels] = generate_linear_data(train_samples, num_features);

    auto [valid_matrix, valid_labels] = generate_linear_data(valid_samples, num_features);

    std::cout << "Training Matrix:\n" << train_matrix << std::endl;
    std::cout << "Training Labels:\n" << train_labels << std::endl;
    // std::cout << "Validation Matrix:\n" << valid_matrix << std::endl;
    // std::cout << "Validation Labels:\n" << valid_labels << std::endl;

    LinearRegression model(train_matrix, train_labels, valid_matrix, valid_labels);

    model.fit_model(0.01, 50);

    // LinearRegression lin_mod(train_matrix, train_labels, valid_matrix, valid_labels);

    return 0;
}