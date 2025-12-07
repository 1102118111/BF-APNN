// vim: sw=2:expandtab:tw=70:spell:syntax=cpp.doxygen

#ifndef __MESH_1D_ED_H__
#define __MESH_1D_ED_H__

#include <iostream>

template<class value_type>
class Mesh1DED {
  public:
    typedef value_type Point;
  private:
    value_type x0, x1, ddx;
    size_t N;

  public:
    Mesh1DED( ) { }
    Mesh1DED(value_type xx0, value_type xx1, size_t n) 
    {
      SetPara(xx0, xx1, n);
    }

    void SetPara(value_type xx0, value_type xx1, size_t n) 
    {
      x0 = xx0;
      x1 = xx1;
      N = n;
      ddx = (x1-x0) / N;
    }

    value_type dx(int i) { return ddx; }
    Point BaryCenter(int i) { return x0 + (value_type(i)+0.5)*ddx; }
    Point BaryCenterSpherical(int i) 
    {
      value_type xl, xr;
      xl = x0 + (value_type(i) - 0.5) * ddx;
      xr = xl + ddx;
      return 3.*(xl*xl+xr*xr)*(xl+xr) / (4.*(xl*xl+xl*xr+xr*xr)); 
    }
    Point IntervalEnd( ) { return x1; }
    size_t n_element( ) { return N; }
    size_t n_edge( ) { return N+1; }
    size_t n_point( ) { return N+1; }
};

#endif //__MESH_1D_ED_H__
