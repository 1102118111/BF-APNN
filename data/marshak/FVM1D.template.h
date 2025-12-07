// vim: sw=2:expandtab:tw=70:spell:syntax=cpp.doxygen
#ifndef __FVM1D_Template_H__
#define __FVM1D_Template_H__

#include "FVM1D.h"
#include <gsl/gsl_poly.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_errno.h>

#define TEMPLATE template<class VAR, class MESH, class EQUATION, class PROBLEM>
#define THIS FVM1D<VAR, MESH, EQUATION, PROBLEM>


double minmod(double a, double b)
{
  if(a > 0){
    if(b > 0) {
      return std::min(a, b);
    }
    return 0;
  } else {
    if(b < 0) {
      return std::max(a,b);
    }
    return 0;
  }
}


TEMPLATE
void THIS::Initialize()
{
  sigma.resize(p_mesh->n_element());

  PROBLEM::SetConst(SpeedOfLight, RadiationConst, eps, ElectroSC);
 
  U1.resize(p_mesh->n_element());
  U2.resize(p_mesh->n_element());
  recon.resize(p_mesh->n_edge());
  Te.resize(p_mesh->n_element());

  for(size_t j = 0; j < p_mesh->n_element();j ++) {
    PROBLEM::Init(SpeedOfLight, RadiationConst, p_mesh->BaryCenter(j), U1[j], Te[j]);
  }

  EQUATION::HalfSpaceIntegration(matrix);
}


TEMPLATE
void THIS::TimeStep( )
{
  dt = 1000000.;
  for (size_t i = 0; i < p_mesh->n_element(); ++i) 
  {
    dt = std::min(dt, eps * p_mesh->dx(i) * CFL / SpeedOfLight); 
  }
}

TEMPLATE
void THIS::Reconstructe_bd(const VAR & ghost)
{
  double slope;
  double dx = p_mesh->dx(0);
  for(size_t l = 0;l < VAR::len;l ++) {
    slope = minmod( 2*(U1[0][l]-ghost[l]) / 
          (p_mesh->dx(0)+dx),
          2*(U1[1][l]-U1[0][l]) / (p_mesh->dx(0)+p_mesh->dx(1)) );

    recon[0][1][l] = U1[0][l] - slope * 0.5 * p_mesh->dx(0);
    recon[1][0][l] = U1[0][l] + slope * 0.5 * p_mesh->dx(0);
  }
}

TEMPLATE
void THIS::Reconstructe(size_t i, size_t l)
{
  double slope;
  if(i == 0){ 
    value_type ghost;
    value_type dx;
    if(PROBLEM::LeftBoundaryType( ) == PERIODIC) {
      ghost = U1[p_mesh->n_element()-1][l];
      dx = p_mesh->dx(p_mesh->n_element()-1);
    }
    else {
      ghost = U1[i][l];
      dx = p_mesh->dx(i);
    }
    slope = minmod( 2*(U1[i][l]-ghost) / 
          (p_mesh->dx(i)+dx),
          2*(U1[i+1][l]-U1[i][l]) / (p_mesh->dx(i)+p_mesh->dx(i+1)) );
   
    recon[i][1][l] = U1[i][l] - slope * 0.5 * p_mesh->dx(i);
    recon[i+1][0][l] = U1[i][l] + slope * 0.5 * p_mesh->dx(i);
    if(PROBLEM::LeftBoundaryType() == PERIODIC) {
      recon[p_mesh->n_element()][1][l] = recon[0][1][l];
    }
    else {
      recon[i][0][l] = ghost;
    }
  }
  else if(i == p_mesh->n_element()-1) {
    if(PROBLEM::RightBoundaryType( ) == PERIODIC) {
      slope = minmod( 2*(U1[i][l]-U1[i-1][l]) / (p_mesh->dx(i)+p_mesh->dx(i-1)),
            2*(U1[0][l]-U1[i][l]) / (p_mesh->dx(i)+p_mesh->dx(0)) );

      recon[i][1][l] = U1[i][l] - slope * 0.5 * p_mesh->dx(i);
      recon[i+1][0][l] = U1[i][l] + slope * 0.5 * p_mesh->dx(i);
      recon[0][0][l] = recon[i+1][0][l];
    }
    else {
      recon[i][1][l] = U1[i][l];
      recon[i+1][0][l] = U1[i][l];
      recon[i+1][1][l] = U1[i][l];
    }
  } 
  else {
    slope = minmod( 2. * (U1[i][l]-U1[i-1][l]) / (p_mesh->dx(i) + p_mesh->dx(i-1)),
          2. * (U1[i+1][l]-U1[i][l]) / ( p_mesh->dx(i+1) + p_mesh->dx(i)));

    recon[i][1][l] = U1[i][l] - slope * 0.5 * p_mesh->dx(i); 
    recon[i+1][0][l] = U1[i][l] + slope * 0.5 * p_mesh->dx(i);
  }
}

TEMPLATE
void THIS::UpdateFlux(size_t j, size_t l, double & flux)
{
  double flux_l, flux_r;
  VAR ghost;

  if(j == 0) {
    if(PROBLEM::LeftBoundaryType() == PERIODIC) {
      PROBLEM::NumericalFlux(l, SpeedOfLight, eps, recon[j][0], recon[j][1], flux_l, dt/p_mesh->dx(j), FORCE);
    }
    else if(PROBLEM::LeftBoundaryType() == INFINITE) {
      PROBLEM::NumericalFlux(l, SpeedOfLight, eps, U2[j], U2[j], flux_l, dt/p_mesh->dx(j), FORCE);
    }
    else {
      PROBLEM::BoundaryCondition(SpeedOfLight, RadiationConst, eps, U2[j], ghost, 
            p_mesh->BaryCenter(j)-0.5*p_mesh->dx(j), LEFT, matrix);

      PROBLEM::NumericalFlux(l, SpeedOfLight, eps, ghost, recon[j][1], flux_l, dt/p_mesh->dx(j), FORCE);
    }
    PROBLEM::NumericalFlux(l, SpeedOfLight, eps, recon[j+1][0], recon[j+1][1], flux_r, dt/p_mesh->dx(j), FORCE);
  }
  else if(j == p_mesh->n_element() - 1) {
    if(PROBLEM::RightBoundaryType() == PERIODIC) {
      PROBLEM::NumericalFlux(l, SpeedOfLight, eps, recon[j+1][0], recon[j+1][1], flux_r, dt/p_mesh->dx(j), FORCE);
    }
    else if(PROBLEM::RightBoundaryType() == INFINITE) {
      PROBLEM::NumericalFlux(l, SpeedOfLight, eps, recon[j+1][0], recon[j+1][1], flux_r, dt/p_mesh->dx(j), FORCE);
    }
    else {
    
    }   
    PROBLEM::NumericalFlux(l, SpeedOfLight, eps, recon[j][0], recon[j][1], flux_l, dt/p_mesh->dx(j), FORCE);
  }
  else {
    PROBLEM::NumericalFlux(l, SpeedOfLight, eps, recon[j][0], recon[j][1], flux_l, dt/p_mesh->dx(j), FORCE);
    PROBLEM::NumericalFlux(l, SpeedOfLight, eps, recon[j+1][0], recon[j+1][1], flux_r, dt/p_mesh->dx(j), FORCE);
  }
  flux_l -= flux_r;
  flux_l /= (p_mesh->dx(j));
  flux = flux_l;
}

TEMPLATE
void THIS::solve_T(double coe_4, double coe_1, double right, double& T) {
    unsigned int order = 5; 
    double a[order] = {-right, coe_1, 0, 0, coe_4}; 
    double z[8]; 

    gsl_poly_complex_workspace * w 
        = gsl_poly_complex_workspace_alloc(order); 
    gsl_poly_complex_solve(a, order, w, z); 
    gsl_poly_complex_workspace_free(w); 

    for (unsigned int iorder = 0; iorder < order - 1; iorder++)  {
        if (z[2 * iorder] > 0 && fabs(z[2*iorder+1]) < 1e-13) {
            T = z[2*iorder];
            break; 
        }
    } 
 if (fabs(T) <1e-13) {
     std::cout<<"wrong T"<<std::endl;
     abort();
 } 
}


TEMPLATE
void THIS::UpdateTe(size_t j)
{
  double rhs = U2[j][0] + dt * ExFlux;
  double a[5] = { -1, 1, 0, 0, 1};
  double z[8];

  if(std::fabs(sigma[j]) < 1e-12) {return;}
  a[0] = -(rhs * sigma[j] * dt + (eps * eps + SpeedOfLight * sigma[j] * dt) * ElectroSC * Te2[j]);
  a[3] = a[2] = 0.;
  a[1] = (eps * eps + SpeedOfLight * sigma[j] * dt) * ElectroSC;
  a[4] = SpeedOfLight * sigma[j] * dt * RadiationConst;


  gsl_poly_complex_workspace * w 
    = gsl_poly_complex_workspace_alloc (5);
  if(std::fabs(a[4]) < 1e-12) {return;}
  gsl_poly_complex_solve (a, 5, w, z);
  gsl_poly_complex_workspace_free (w);
 
  double err(100000);
  for(size_t i = 0;i < 4;i ++)
  {
    if(z[2*i] > 0 && fabs(z[2*i+1]) < 1e-13)
    {
      if(std::fabs(z[2*i]- Te2[j]) < err) {
        err = std::fabs(z[2*i] - Te2[j]);
        Te[j] = z[2*i];
      }
    }
  }
}


TEMPLATE
void THIS::UpdateI0(size_t j)
{
  U1[j][0] = (U2[j][0] + dt * ExFlux) / (1 + SpeedOfLight*sigma[j]*dt/eps/eps)
    + SpeedOfLight*sigma[j]*dt* RadiationConst * SpeedOfLight * std::pow(Te[j],4) /eps/eps/
    (1 + SpeedOfLight*sigma[j]*dt/eps/eps);
}

TEMPLATE
void THIS::UpdateIl(size_t j, size_t l)
{
  U1[j][l] = (U2[j][l] + dt * ExFlux) / (1. + SpeedOfLight * sigma[j] * dt / eps / eps);
}




TEMPLATE
void THIS::FirstOrderStepping( )
{
  U2 = U1;
  Te2 = Te;
  int num_element = p_mesh->n_element();
#pragma omp for
  for(size_t j = 0;j < num_element;j ++) {
    PROBLEM::UpdateAbsorpCoeff(p_mesh->BaryCenter(j), SpeedOfLight, sigma[j], Te[j]);
  }
#pragma omp for
  for(size_t j = 0;j < num_element;j ++) {
    for(size_t l = 0;l < VAR::len;l ++) {
      Reconstructe(j, l);
    }
  }
#pragma omp for
  for(size_t j = 0;j < num_element;j ++) {
    UpdateFlux(j, 0, ExFlux);
    UpdateTe(j);
    UpdateI0(j);
  }
#pragma omp for
  for(size_t l = 1; l < VAR::len;l ++) {
    for(size_t j = 0;j < num_element;j ++) {
      UpdateFlux(j, l, ExFlux);
      UpdateIl(j, l);
    }
  }
}


TEMPLATE
void THIS::ForwardOneStep( ) 
{
  FirstOrderStepping();
}

TEMPLATE
void THIS::Run(value_type t_end)
{
  bool finished = false;
  clock_t begin = clock( );
  clock_t time_tmp = begin;

  size_t count;

  count = 0;
//#pragma omp parallel
  while(!finished) {
//#pragma omp single
    {
    TimeStep( );
    if(t + dt >= t_end)
    {
      dt = t_end - t;
      finished = true;
    }
    }
    
    ForwardOneStep( );
    
//#pragma omp single
    {
    //GetTotalEnergy();
    t += dt;

    clock_t end = clock( );

    std::cout << "t =\t" << t
      << "\tdt =\t" << dt 
      << "\t total run time is \t " << value_type(end - begin) / CLOCKS_PER_SEC
      << "\t run time for this step is \t " << value_type(end - time_tmp) / CLOCKS_PER_SEC 
      << std::endl;

      count ++;

    time_tmp = end;
    }
  }
}


TEMPLATE
double THIS::GetTotalEnergy()
{
  std::string energy_file = "tmp";
  std::ofstream output(energy_file.c_str(), std::ios_base::app);
  output.precision(12); 
  double totE(0);
  for(size_t i = 0;i < p_mesh->n_element();i ++) {
    totE += U1[i][0];
  }
    output << totE << std::endl;
}

TEMPLATE
void THIS::OutputData(std::string filename)
{
  std::ofstream output(filename.c_str());
output.precision(12); 
  for (size_t i = 0; i < p_mesh->n_element(); ++i) 
  {
    output << p_mesh->BaryCenter(i) << "\t\t";
    output << U1[i][0] << "\t" << Te[i] << std::endl;
  }
}


#undef TEMPLATE
#undef THIS
#endif 
