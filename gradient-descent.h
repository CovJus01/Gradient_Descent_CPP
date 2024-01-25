#ifndef GRADIENT_DESCENT
#define GRADIENT_DESCENT
#include <math.h>
#include <vector>

template <typename T>
T compute_univariate_cost();

template <typename T>
std::pair<T, T> compute_univariate_gradient();

template <typename T>
T univariate_gd();

template <typename T>
T compute_univariate_cost(T w, T b,  std::vector<std::pair<T, T>> &expected) {

 std::vector<T> squared_error;
 int data_size = expected.size();
 squared_error.resize(data_size);

 transform(expected.begin(), expected.end(), squared_error.begin(), [constW = w, constB = b](std::vector<std::pair<T, T>> input) {return pow(((input.first * constW + constB ) - input.second), 2)});
 T sum = accumulate(squared_error.begin(), squared_error.end(), 0, [](T lastVal, T currentVal) {return lastVal + currentVal});
 sum *= (1/(2*data_size));
 return sum;
}

template <typename T>
T compute_dj_dw_error(T input, T result, T w, T b) {
  return ((input * w + b ) - result)*input;
}

template <typename T>
T compute_dj_db_error(T input, T result, T w, T b) {
  return ((input * w + b ) - result);
}

template <typename T>
std::pair<T, T> compute_univariate_gradient(T w, T b,  std::vector<std::pair<T, T>> &expected) {

 std::vector<std::pair<T, T>> dj_dt_error;
 int data_size = expected.size();
 dj_dt_error.resize(data_size);

 transform(expected.begin(), expected.end(),
           dj_dt_error.begin(), 
           [constW = w, constB = b](std::vector<std::pair<T, T>> input) {
              return std::make_pair(compute_dj_dw_error(input.first, input.second, constW, constB), compute_dj_db_error(input.first, input.second, constW, constB))
              });
 T sum = accumulate(dj_dt_error.begin(), dj_dt_error.end(),
                    std::make_pair(0, 0),
                    [](T lastVal, T currentVal) {
                      return std::make_pair(lastVal.first+currentVal.first, lastVal.second + currentVal.second)
                      });
 sum = std::make_pair(sum.first/data_size, sum.second/data_size);
 return sum;
}

template <typename T>
T univariate_gradient_d() {

}



#endif
