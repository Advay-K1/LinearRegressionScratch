#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H


#include <xtensor/xtensor.hpp>
#include <xtensor/xarray.hpp>
#include "xtensor-blas/xlinalg.hpp"
#include "xtensor-blas/xblas.hpp"
#include <xtensor/xio.hpp>
#include <xtensor/xmath.hpp>

class LinearRegression {
    public:
        LinearRegression(
            const xt::xtensor<double, 2>& predictor_matrix_train,
            const xt::xtensor<double, 1>& y_vector_train,
            const xt::xtensor<double, 2>& predictor_matrix_valid,
            const xt::xtensor<double, 1>& y_vector_valid
        ); 


        LinearRegression(
            const xt::xtensor<double, 2>& predictor_matrix_train,
            const xt::xtensor<double, 1>& y_vector_train
        );

        // LinearRegression(
        //     xt::xtensor<double, 1> predictor_vector,
        //     xt::xtensor<double, 1> y_value
        //     xt::xtensor<double, 1> predictor_vector_valid,
        //     xt::xtensor<double, 1> y_vector_valid
        // );

        double RSS();

        double TSS();

        double R_squared();

        void calculate_response_vector();

        void compute_bias_vector();

        void compute_error_vector();

        xt::xtensor<double, 1> batch_gradient_descent(const double& learning_rate, const int& max_iterations);

        void update_bias_vector(const xt::xtensor<double, 1>& gradient, const double& learning_rate);


        void fit_model(const double& learning_rate, const int& max_iterations);

        xt::xtensor<double, 1> get_bias_vector();

        xt::xtensor<double, 1> get_error_vector();


        xt::xtensor<double, 2> get_response_vector(); 



    private:
        xt::xtensor<double, 2> design_matrix;
        xt::xtensor<double, 1> bias_vector;
        xt::xtensor<double, 1> error_vector;
        
        xt::xtensor<double, 1> response_vector; //y_pred vector
        xt::xtensor<double, 1> y_vector;

        xt::xtensor<double, 2> x_valid_matrix;
        xt::xtensor<double, 1> y_valid_vector;



};


#endif