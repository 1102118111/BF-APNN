// vim: sw=2:expandtab:tw=70:spell:syntax=cpp.doxygen

#ifndef __SOLUTION_H__
#define __SOLUTION_H__

#include <vector>

template<class Vector>
class Solution : public std::vector<Vector>
{
  private:
    typedef std::vector<Vector> Base;

  public:
    Solution(){}
    Solution(size_t n) : Base(n) { }
    Solution(size_t n, const Vector& b) : Base(n, b) { }

    Solution<Vector>& operator+=(const Solution<Vector>& s)
    {
      for(size_t i = 0; i < this->size(); ++i)
      (*this)[i] += s[i];
      return *this;
    }
    Solution<Vector>& operator-=(const Solution<Vector>& s)
    {
      for(size_t i = 0; i < this->size(); ++i)
      (*this)[i] -= s[i];
      return *this;
    }
    Solution<Vector>& operator*=(double factor)
    {
      for(size_t i = 0; i < this->size(); ++i)
      (*this)[i] *= factor;
      return *this;
    }
    Solution<Vector>& operator/=(double factor)
    {
      for(size_t i = 0; i < this->size(); ++i)
      (*this)[i] /= factor;
      return *this;
    }
};

//重载运算符，但这些运算符并不快，不建议使用
template<class Vector>
Solution<Vector> operator+(const Solution<Vector>& s1,
      const Solution<Vector>& s2)
{
  Solution<Vector> s3(s1);
  s3 += s2;
  return s3;
}
template<class Vector>
Solution<Vector> operator-(const Solution<Vector>& s1,
      const Solution<Vector>& s2)
{
  Solution<Vector> s3(s1);
  s3 -= s2;
  return s3;
}
template<class Vector>
Solution<Vector> operator*(double val, const Solution<Vector>& s)
{
  Solution<Vector> s2(s);
  s2 *= val;
  return s2;
}
template<class Vector>
Solution<Vector> operator*(const Solution<Vector>& s, double val)
{
  return val * s;
}
template<class Vector>
Solution<Vector> operator/(const Solution<Vector>& s, double val)
{
  Solution<Vector> s2(s);
  s2 /= val;
  return s2;
}

#endif //__SOLUTION_H__
