// vim: sw=2:expandtab:tw=70:spell:syntax=cpp.doxygen
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <sstream>
#include <string>

#include "Mesh1DED.h"
#include "Pn_slab.h"
#include "Solution.h"
#include "FVM1D.h"
#include "FVM1D.template.h"


template<class EQUATION>
class MarshakWave {
  public:
    static int RightBoundaryType( ) { return INFINITE; }

    static int LeftBoundaryType( ) { return INFLOW;}

    static void SetConst(double &SpeedOfLight, double & RadiationConst,
          double & eps, double & Cv)
    {
      double c = 29.97924580;
      SpeedOfLight = c;
      eps = 1.;
      Cv = 0.3;
      RadiationConst = 0.01372;
    }

   
    // u denotes I_{l,g}.
    static void Init(const double & c, const double & RadiationConst, const double &x, Variables & u, double & Te)
    {
      // Te = 1e-2;
      if (x<0.1){
        Te = 1e-2 +(1-1e-2)*std::exp(-x*x/(0.1*0.1-x*x));
      }
      else{
        Te = 1e-2;
      }
      u[0] = RadiationConst * c *std::pow(Te, 4);

      for(size_t j = 1; j < Variables::len;j ++) {
        u[j] = 0.;
      }
    }


    static void NumericalFlux(size_t l, const double &SpeedOfLight, const double & eps, const Variables& ul, const Variables& ur, 
          double & flux, double para, int FluxType)
    {
      //LF
      if(FluxType == LF) {
        double fl, fr;
        EQUATION::RealFlux(l, ul ,fl, SpeedOfLight, eps);
        EQUATION::RealFlux(l, ur ,fr, SpeedOfLight, eps);
        fl += fr; fl *= 0.5;
        flux = ul[l]; flux -= ur[l]; flux /= (2. * para);  
        flux += fl;
      }

      //FORCE
      if(FluxType == FORCE) {
        Variables fl, fr;
        EQUATION::RealFlux(ul, fl, SpeedOfLight, eps);
        EQUATION::RealFlux(ur, fr, SpeedOfLight, eps);
        fl *= para; fl += ul;
        fr *= -para; fr += ur;

        Variables u_star = fl; u_star += fr; u_star *= 0.5;
        double flux_lw;
        EQUATION::RealFlux(l, u_star, flux_lw, SpeedOfLight, eps); //calculate flux_lw
        flux = fl[l]; flux -= fr[l]; flux /= 2. * para; //calculate flux_lf
        flux += flux_lw; flux *= 0.5;
      }
    }

    static int BoundaryCondition(const double & SpeedOfLight, const double & RadiationConst, 
          const double & eps, const Variables& u, Variables & ghost, 
          value_type pt, int side, const std::vector< std::vector< double > > & matrix) 
    {
      if(side == LEFT) {
        LeftBoundaryCondition(SpeedOfLight, RadiationConst, eps, u, ghost, pt, matrix);
      }
      else {
        RightBoundaryCondition(SpeedOfLight, RadiationConst, eps, u, ghost, pt, matrix);
      }
      return 0;
    }

    static int LeftBoundaryCondition(const double & SpeedOfLight, const double & RadiationConst,
          const double & eps, const Variables& u, Variables & ghost, 
          value_type pt, const std::vector< std::vector< double > > & matrix)
    {
      double Te = 1.;
      for(size_t l = 0;l < Variables::len;l ++) {
        // inflow at the left boundary
        ghost[l] = 0.5 * RadiationConst * SpeedOfLight * std::pow(Te, 4) * matrix[0][l];

        size_t n_size;
        n_size = Variables::len;

        // Get the outflow of boundary flux by numerical integration
        for(size_t n = 0;n < n_size;n ++) {
          if((n + l) % 2 == 1) {
            ghost[l] -=  (2.*n+1.)/2. * u[n] * matrix[n][l];
          }
          else {
            ghost[l] += (2.*n+1.)/2. * u[n] * matrix[n][l];
          }
        }
      }
      return 0;
    }


    static int RightBoundaryCondition(const double & SpeedOfLight, const double & RadiationConst, 
          const double & eps, const Variables& u, Variables& ghost, 
          value_type pt, const std::vector< std::vector< double > > & matrix)
    {
      return 0;
    }



    static void ExternalSource(const Variables & u, Variables &
          add_source, const double & coordinate) 
    {
      add_source[0] = 0.; 
      add_source[1] = 0.;
      add_source[2] = 0.;
    }


    static void UpdateAbsorpCoeff(const double &x, const double & c, double & sigma, const double &Te)
    {
      sigma = (x<0.2?30.:90.) / std::pow(Te, 3);
    }
};


int main(int argc, char* argv[])
{
  if(argc < 6) {
    std::cerr << "Usage: " << argv[0]
      << "  <x0> <x1> <mesh_size> <t_end> <outputfile>"
      << std::endl;
    return 1;
  }
  double t_end = atof(argv[4]);
  double x0 = atof(argv[1]);
  double x1 = atof(argv[2]);
  size_t mesh_size = atoi(argv[3]);

  Mesh1DED<double> mesh(x0, x1, mesh_size);

  FVM1D<Variables, Mesh1DED<double>, Pn_SlabEq, 
    MarshakWave< Pn_SlabEq > > marshakwave(&mesh);
  marshakwave.Initialize();
  // marshakwave.Run(t_end);
  // marshakwave.OutputData(argv[5]);
  for (double t_mid = 1e-15; t_mid <= t_end; t_mid = t_mid+1){
    marshakwave.Run(t_mid<t_end?t_mid:t_end);
    std::stringstream ss;
    ss << argv[5] << "_" << (t_mid<t_end?t_mid:t_end) ;
    marshakwave.OutputData(ss.str());
  }
  return 0;
}
