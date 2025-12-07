// vim: sw=2:expandtab:tw=70:spell:syntax=cpp.doxygen

/** 
 * @file    Vector.h
 * @author  Yuwei Fan   <dawanzhi@163.com>
 * @date    2012-12-15
 */
#ifndef __VECTOR_H__
#define __VECTOR_H__

#include <valarray>
#include <fstream>
#include <cmath>
#include <numeric>
#include <boost/static_assert.hpp>
#include <boost/preprocessor/arithmetic/add.hpp>
#include <boost/preprocessor/repetition.hpp>

namespace FVM {
  /** 
   * @brief Used to discribe the solution u in an element. 
   * This class provides some basic operators.
   */
  template<std::size_t n, class _T>
    class Vector : private std::valarray<_T>
  {
    private:
      typedef std::valarray<_T> Base;
    public:
      typedef _T value_type;

      enum { len = n };

    public:
      Vector():Base(n){}
      Vector(const value_type& x):Base(x, n){}
      Vector(const _T* __px) : Base(__px, n) {}
      Vector(_T* __px) : Base(__px, n) {}

      /** 
       * @brief Compute the L2 norm of the Vector
       */
      value_type Length() const 
      {
        value_type norm_l2 = std::inner_product(&(*this)[0],
              &(*this)[n], &(*this)[0], value_type(0));
        return sqrt(norm_l2);
      }

      value_type& operator[](size_t i) { return Base::operator[](i); }
      const value_type& operator[](size_t i) const { return Base::operator[](i); }
      template <class _C> Vector& operator=(_C __c) { Base::operator=(__c); return *this; }
      template <class _C> Vector& operator+=(_C __c) { Base::operator+=(__c); return *this; }
      template <class _C> Vector& operator-=(_C __c) { Base::operator-=(__c); return *this; }
      template <class _C> Vector& operator*=(_C __c) { Base::operator*=(__c); return *this; }
      template <class _C> Vector& operator/=(_C __c) { Base::operator/=(__c); return *this; }
      template <class _C> Vector& operator^=(_C __c) { Base::operator^=(__c); return *this; }
      template <class _C> Vector& operator&=(_C __c) { Base::operator&=(__c); return *this; }
      template <class _C> Vector& operator|=(_C __c) { Base::operator|=(__c); return *this; }
      template <class _C> Vector& operator%=(_C __c) { Base::operator%=(__c); return *this; }
      template <class _C> Vector& operator<<=(_C __c) { Base::operator<<=(__c); return *this; }
      template <class _C> Vector& operator>>=(_C __c) { Base::operator>>=(__c); return *this; }
  };
};

#endif //__VECTOR_H__
