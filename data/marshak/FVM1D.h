// vim: sw=2:expandtab:tw=70:spell:syntax=cpp.doxygen

#ifndef __FVM1D_H__
#define __FVM1D_H__

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <string>
#include <ctime>
#include <vector>
#include "Solution.h"
#include "Pn_slab.h"

const double CFL = 0.4;

enum {LEFT = 1, RIGHT = 2};
enum {PERIODIC = 0, DIRICHLET = 1, INFLOW = 2, INFINITE = 3, REFLECTIVE = 4};
enum {LF = 1, FORCE = 2};



template<class VAR, class MESH, class EQUATION, class PROBLEM>
class FVM1D {
  public:
    typedef typename VAR::value_type value_type;
    typedef Solution<VAR> SolutionU;

    class VarInterface : public std::vector<VAR>
  {
    public:
      typedef std::vector<VAR> Base;
      VarInterface( ) : Base(2) { }
  };

    typedef Solution<VarInterface> Reconstructor;
  private:

    MESH* p_mesh;
    value_type t, dt;
    Solution<double>  Te, Te2;
    value_type SpeedOfLight, RadiationConst, PlanckConst, BoltzmannConst, ScaleConst, eps, ElectroSC;
    SolutionU U1, U2;
    Reconstructor recon;

    // coefficient matrix for half space integration.
    std::vector< std::vector< double > > matrix;

    value_type ExFlux;


    std::vector< double > sigma;
       
  public:
    FVM1D( ) : t(0) { }
    FVM1D(MESH* p_m) : p_mesh(p_m), t(0){ }

    void SetMesh(MESH* p_m)
    {
      p_mesh = p_m;
    }

    void Initialize();
    void Run(value_type t_end);

    void TimeStep( );

    void UpdateFlux(size_t, size_t, double & );

    void UpdateTe(size_t );
    void UpdateI0(size_t );
    void UpdateIl(size_t, size_t);
    
    void solve_T(double, double, double, double &); 
    void Reconstructe_bd(const VAR & );
    void Reconstructe(size_t , size_t );
    void FirstOrderStepping();
    void ForwardOneStep( );
    double GetTotalEnergy();
    void OutputData(std::string filename);

};

#endif //__FVM1D_H__
