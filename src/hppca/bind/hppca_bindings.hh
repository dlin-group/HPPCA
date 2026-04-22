#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cblas.h>
#include <assert.h>

using namespace pybind11;
using namespace std;

void set_cblas_threads(int n){
  openblas_set_num_threads(n);
}

template <typename T>
array_t<T> zeros(size_t m, size_t n){
  array_t<T> z({m, n});
  memset(z.request().ptr, T(0), m*n*sizeof(T));
  return z;
}

template <typename T>
void set_all(array_t<T> m, T val){
  memset(m.request().ptr, val, m.size()*sizeof(T));
}

template <typename T>
array_t<T> outer_prod(array_t<T> v0, array_t<T> v1){
  buffer_info v0_info = v0.request();
  buffer_info v1_info = v1.request();
  const size_t n0 = v0_info.size, n1 = v1_info.size;
  array_t<T> v2({n0, n1});
  buffer_info v2_info = v2.request();
  T* p0 = static_cast<T*>(v0_info.ptr);
  T* p1 = static_cast<T*>(v1_info.ptr);
  T* p2 = static_cast<T*>(v2_info.ptr);
  size_t i2 = 0;
  for(size_t i0=0; i0<n0; i0++)
    for(size_t i1=0; i1<n1; (i1++, i2++))
      p2[i2] = p0[i0] * p1[i1];
  return v2;
}

template <typename T>
T trace_prod(array_t<T> m0, array_t<T> m1){
  const size_t m = m0.shape()[0], n = m0.shape()[1];
  T tr = T();
  for(size_t i=0; i<m; i++)
    for(size_t j=0; j<n; j++)
      tr += m0.at(i, j) * m1.at(j, i);
  return tr;
}

template <typename T>
T trace_prod_t(array_t<T> m0, array_t<T> m1){
  buffer_info m0_info = m0.request();
  buffer_info m1_info = m1.request();
  const size_t n = m0_info.size;
  T* p0 = static_cast<T*>(m0_info.ptr);
  T* p1 = static_cast<T*>(m1_info.ptr);
  T tr = T();
  for(size_t i=0; i<n; i++) tr += p0[i] * p1[i];
  return tr;
}

void mat_prod(array_t<double> m0, array_t<double> m1, array_t<double> m2,
	      bool t0=false, bool t1=false, double alpha=1, double beta=0);

array_t<double> mat_inverse(array_t<double> m);

template <typename T>
array_t<T> sub_vector(array_t<T> m, size_t i0, size_t i1){
  assert(i1 >= i0);
  buffer_info m_info = m.request();
  assert(i1 <= m_info.shape[0]);
  const size_t ni = i1 - i0;
  array_t<T> sm(ni);
  buffer_info sm_info = sm.request();
  T* psm = static_cast<T*>(sm_info.ptr);
  size_t k = 0;
  for(size_t i=i0; i<i1; (i++, k++)) psm[k] = m.at(i);
  return sm;
}

template <typename T>
array_t<T> s_vector(array_t<T> m, array_t<int> rows){
  buffer_info m_info =    m.request();
  buffer_info r_info = rows.request();
  int* pr = static_cast<int*>(r_info.ptr);
  for(int i=0; i<r_info.size; i++) assert(pr[i] >= 0 &&
					     pr[i] <  m_info.shape[0]);
  array_t<T> sm({r_info.size});
  buffer_info sm_info = sm.request();
  T* pm = static_cast<T*>(m_info.ptr), *psm = static_cast<T*>(sm_info.ptr);
  for(int i=0; i<r_info.size; i++) psm[i] = pm[pr[i]];
  return sm;
}

template <typename T>
array_t<T> sub_matrix(array_t<T> m, size_t i0, size_t i1, size_t j0, size_t j1){
  assert(i1 >= i0 && j1 >= j0);
  buffer_info m_info = m.request();
  assert(i1 <= m_info.shape[0] && j1 <= m_info.shape[1]);
  const size_t ni = i1 - i0, nj = j1 - j0; 
  array_t<T> sm({ni, nj});
  buffer_info sm_info = sm.request();
  T* psm = static_cast<T*>(sm_info.ptr);
  size_t k = 0, l = 0;
  for(size_t i=i0; i<i1; (i++, k++, l=k*nj))
    for(size_t j=j0; j<j1; (j++, l++))
      psm[l] = m.at(i, j);
  return sm;
}

template <typename T>
array_t<T> s_matrix(array_t<T> m, array_t<int> rows, array_t<int> cols){
  buffer_info m_info =    m.request();
  buffer_info r_info = rows.request(), c_info = cols.request();
  int* pr = static_cast<int*>(r_info.ptr), *pc = static_cast<int*>(c_info.ptr);
  for(int i=0; i<r_info.size; i++) assert(pr[i] >= 0 &&
					     pr[i] <  m_info.shape[0]);
  for(int i=0; i<c_info.size; i++) assert(pc[i] >= 0 &&
					     pc[i] <  m_info.shape[1]);
  array_t<T> sm({r_info.size, c_info.size});
  buffer_info sm_info = sm.request();
  T* pm = static_cast<T*>(m_info.ptr), *psm = static_cast<T*>(sm_info.ptr);
  size_t k = 0;
  for(int i=0; i<r_info.size; i++)
    for(int j=0; j<c_info.size; (j++, k++))
      psm[k] = pm[pr[i]*m_info.shape[1]+pc[j]];
  return sm;
}

template <typename T>
void add_identity(array_t<T> m, T mult){
  const size_t n = m.shape()[0];
  assert(n == m.shape()[1]);
  for(size_t i=0; i<n; i++) m.mutable_at(i, i) += mult;
}

void add_Gij(array_t<double> Yp_ij_g,
	     array_t<double> EZ1i_g,    array_t<double> EZ2ij_g,
	     array_t<double> EYmZ1T_g,  array_t<double> EYmZ2T_g,
	     array_t<bool>   obs_mask,  array_t<int>    Yp_index,
	     array_t<bool>   miss_mask, array_t<int>    EYm_index,
	     array_t<double> sG1,       array_t<double> sG2);
