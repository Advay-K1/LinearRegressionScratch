#include <xtensor/xtensor.hpp>
#include "xtensor-blas/xlinalg.hpp"
#include "xtensor-blas/xblas.hpp"
#include <xtensor/xio.hpp>
#include <xtensor/xmath.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/math/distributions/fisher_f.hpp>

#include "includes/LinearRegression.hpp"


class HypothesisTest {
    public:
        HypothesisTest(LinearRegression model, 
            const xt::xtensor<double, 2>& predictor_matrix_train,
            const xt::xtensor<double, 1>& y_vector_train);

        /*
        H0: The reduced model is adequate given the coefficient indices to remove
        H1: The full model is required
        */

        double get_f_statistic(std::vector<int> coeffs_remove);

        double get_p_value_f(double f_statistic, std::vector<int> coeffs_remove);





    private:
        LinearRegression lin_mod;
        xt::xtensor<double, 2> design_matrix;
        xt::xtensor<double, 1> y_vector;




};
