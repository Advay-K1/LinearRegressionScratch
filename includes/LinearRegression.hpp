#pragma once

#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>

class LinearRegression {
    public:
        LinearRegression(
            xt::xtensor<double,2> predictor_matrix,
            xt::xtensor<double, 1> y_vector
            );

        LinearRegression(
            xt::xtensor<double, 1> predictor_vector,
            xt::xtensor<double, 1> y_value
        );

        double RSS();

        double TSS();

        double R_squared();

        void calculate_response_vector();

        


    private:
        xt::xtensor<double, 2> design_matrix;
        xt::xtensor<double, 1> bias_vector;
        xt::xtensor<double, 1> error_vector;
        xt::xtensor<double, 1> response_vector;



};