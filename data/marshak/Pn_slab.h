// vim: sw=2:expandtab:tw=70:spell:syntax=cpp.doxygen

#ifndef __Pn_Slab_H__
#define __Pn_Slab_H__

#include "Mesh1DED.h"
#include "Vector.h"
#include <vector>

// sets the order of Pn
typedef FVM::Vector<12, double> Variables;
typedef Variables::value_type value_type;

double fac(unsigned int M) {
  if (M == 1 || M == 0)  
    return 1.0; 
  else {
    double coe = M * fac(M-1); 
    return coe; 
  }
}


double f(int M, int N) {
  double coe1 = std::pow(-1, (M + N + 1)/2) * fac(M) * fac(N); 
  double coe2 = std::pow(2, M + N -1) * ( M + N + 1) * (1.0 * M - N)  
    * std::pow(fac(M/2), 2) * std::pow(fac((N-1)/2), 2); 
  
  return coe1 / coe2; 
}


class Pn_SlabEq {
  public:
    static double MaxEigenvalue(const Variables& u)
    {
      return 1.;
    }

    static void RealFlux(size_t l, const Variables& u, double& fu, const double & c, const double & eps)
    {
      if(l == 0) {
        fu = u[1];
      }
      if(l >= 1 && l < Variables::len-1) {
        fu = u[l-1] * (1.*l/(2.*l+1.)) + 
          u[l+1] * (l+1.) / (2.*l+1.);
      }
      if( l == Variables::len-1) {
        fu = 1.*l / (2.*l + 1.) * u[l-1];
      }
      fu *= (c / eps);
    }



    static void RealFlux(const Variables& u, Variables& fu, const double & c, const double & eps)
    {
      fu[0] = u[1];
      for(size_t j = 1;j < Variables::len-1; j ++) {
        fu[j] = u[j-1] * (1.*j/(2.*j+1.)) + 
          u[j+1] * (j+1.) / (2.*j+1.);
      }
      size_t l;
      l = Variables::len-1;
      fu[l] = l / (2.*l + 1.) * u[l-1]; 
      for(size_t i = 0;i < Variables::len;i ++) {
        fu[i] *= (c / eps);
      }
    }


    static void HalfSpaceIntegration(std::vector<std::vector<double> > & PositiveHalfSpaceMatrix)
    {
      size_t n_size;
      n_size = Variables::len + 1;
      PositiveHalfSpaceMatrix.resize(n_size);
      for(size_t j = 0;j < n_size;j ++) {
        PositiveHalfSpaceMatrix[j].resize(n_size);
        for(size_t i = 0;i < n_size;i ++) {
          PositiveHalfSpaceMatrix[j][i] = 0.;
        }
      }

      for(size_t j = 0;j < n_size;j ++) {
        for(size_t i = 0;i < n_size;i ++) {
          if(i == j) {
            PositiveHalfSpaceMatrix[j][i] = 1. / (2. * i + 1.);
          }
          else if((i+j) % 2 == 0) { PositiveHalfSpaceMatrix[j][i] = 0.;}
          else {
            if(i % 2 == 0 && j % 2 == 1) {
              PositiveHalfSpaceMatrix[j][i] = f(i,j);
            }
            if(i % 2 == 1 && j % 2 == 0) {
              PositiveHalfSpaceMatrix[j][i] = f(j,i);
            }
          }
        }
      }
    }
};
#endif //__Pn_Slab_H__
