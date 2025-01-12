#include "/Users/advaykadam/LinearRegressionScratch/includes/HypoTesting.hpp"

HypothesisTest::HypothesisTest(LinearRegression model, const xt::xtensor<double, 2>& predictor_matrix_train,
            const xt::xtensor<double, 1>& y_vector_train): 
            lin_mod(model), 
            design_matrix(predictor_matrix_train), 
            y_vector(y_vector_train)
            {}


double HypothesisTest::get_f_statistic(std::vector<int> coeffs_remove) {
    xt::xtensor<double, 1> original_bias_vector = lin_mod.get_bias_vector();
    xt::xtensor<double, 1> reduced_bias_vector = original_bias_vector;


    double original_RSS = lin_mod.RSS();
    std::vector<int> cols_to_keep;

    for (size_t col = 0; col < design_matrix.shape()[1]; ++col) {
        if (std::find(coeffs_remove.begin(), coeffs_remove.end(), col) == coeffs_remove.end()) {
            cols_to_keep.push_back(col);
        }
    }

    xt::xtensor<double, 2> reduced_design_matrix = xt::view(design_matrix, xt::all(), xt::keep(cols_to_keep));


    LinearRegression reduced_model(design_matrix, y_vector);

    reduced_model.fit_model(0.01, 50);

    double reduced_RSS = reduced_model.RSS();

    int removed_coeffs_count = coeffs_remove.size();          
    int design_matrix_height = design_matrix.shape(0);           
    int design_matrix_width = design_matrix.shape(1);


    double f_stat = ((reduced_RSS - original_RSS) / removed_coeffs_count) / (original_RSS / (design_matrix_height - design_matrix_width));

    return f_stat;

}

double HypothesisTest::get_p_value_f(double f_statistic, std::vector<int> coeffs_remove) {


    std::vector<int> cols_to_keep;

    int removed_coeffs_count = coeffs_remove.size();          
    int design_matrix_height = design_matrix.shape(0);           
    int design_matrix_width = design_matrix.shape(1);

    xt::xtensor<double, 2> reduced_design_matrix = xt::view(design_matrix, xt::all(), xt::keep(cols_to_keep));
    boost::math::fisher_f_distribution<> dist(removed_coeffs_count,  design_matrix_height- design_matrix_width);
    double p_value = 1.0 - boost::math::cdf(dist, f_statistic);

    return p_value;
}