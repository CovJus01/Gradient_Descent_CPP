#ifndef GRADIENT_DESCENT
#define GRADIENT_DESCENT
#include <math.h>
#include <vector>
#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <iomanip>
// ***************************** Declarations *****************************


template <typename T>
T univariateCost(T w, T b,  std::vector<T> &x, std::vector<T> &y);

template <typename T>
struct pairSum;

template <typename T>
std::pair<T, T> univariateGradient(T w, T b,  std::vector<T> &x, std::vector<T> &y );

template <typename T>
std::pair<T,T> univariateGradientDescent(T w, T b, std::vector<T> x, std::vector<T> y);

// ***************************** Definitions *****************************
template <typename T> class squareError {
    private:
        T w;
        T b;
    public:
    squareError(T w_in, T b_in) {
        w = w_in;
        b = b_in;
    }

    T operator () (T xi, T yi) {
        return pow(((xi * w + b ) - yi), 2.0);
    }
};

template <typename T>
T univariateCost(T w, T b,  std::vector<T> &x, std::vector<T> &y) {

      // Initialization
      int data_size = x.size(); // Size of training set
      std::vector<T> squared_error(data_size); 

      // Perform square_error on dataset
      std::transform(x.begin(), x.end(), y.begin(), squared_error.begin(), squareError<T>(w,b));

      // Sum the square_errors
      T sum = std::accumulate(squared_error.begin(), squared_error.end(), 0.0, std::plus<T>());

      // Divide to find average cost
      sum *= (1.0/(2.0*(double)data_size));

 return sum;
}

template <typename T> class UnivariateGradientError {
    private:
        T w;
        T b;
        T univariateError_dj_dw(T w, T b, T xi, T yi) {
            // Derivative of square error w.r.t w (scalar 2 cancelled in larger equation)
            return ((xi * w + b ) - yi)*xi;
        }
        T univariateError_dj_db(T w, T b, T xi, T yi) {
            // Derivative of square error w.r.t b (scalar 2 cancelled in larger equation)
            return ((xi * w + b ) - yi);
            }
    public:
        UnivariateGradientError(T w_in, T b_in) {
            w = w_in;
            b = b_in;
        }

        // Operator Override to return both gradient errors as a pair
        std::pair<T,T> operator () (T xValue, T yValue) {
            return std::make_pair(univariateError_dj_dw(w, b, xValue, yValue), univariateError_dj_db(w, b, xValue, yValue));
        }
};

// Binary Operator for a sum of two pairs
template <typename T> struct pairSum {
    std::pair<T,T> operator () (std::pair<T,T> x, std::pair<T,T> y){
    return std::make_pair(x.first+y.first, x.second + y.second);
    }
};

template <typename T>
std::pair<T, T> univariateGradient(T w, T b,  std::vector<T> &x, std::vector<T> &y ) {

    // Initialize
    int data_size = x.size(); // Size of training set
    std::vector<std::pair<T, T>> dj_dt_error(data_size);

    // Calculate both Gradient Errors
    transform(x.begin(), x.end(), y.begin(), dj_dt_error.begin(), UnivariateGradientError<T>(w, b));

    // Sum all the errors together
    std::pair<T,T> sum = accumulate(dj_dt_error.begin(), dj_dt_error.end(), std::make_pair(0.0, 0.0), pairSum<T>());

    // Divide to get the gradient
    sum = std::make_pair(sum.first/data_size, sum.second/data_size);
    
    return sum;
}

template <typename T>
std::pair<T,T> univariateGradientDescent(T init_w, T init_b, std::vector<T> x, std::vector<T> y, double learning_rate,int iterations) {
    
    double w = init_w;
    double b = init_b;
    std::pair<T,T> gradient;
    T cost;

    for(int i = 0; i < iterations; i++) {
        cost = univariateCost<T>(w,b,x,y);
        if(i % 150 == 0) {
            std::cout << "Iteration "<< i << ": " << w << ", "<< b << "| Cost: " << std::setprecision(6) << cost << std::endl;
        }
        gradient = univariateGradient<T>(w,b,x,y);
        w = w - learning_rate * gradient.first;
        b = b - learning_rate * gradient.second;
    }

    return std::pair<T,T>(w,b);
}



#endif
