#include "hppca_bindings.hh"
#include <lapack.h>
#include <lapacke.h>
#include <vector>

using namespace pybind11;
using namespace std;

void mat_prod(array_t<double> m0, array_t<double> m1, array_t<double> m2,
	      bool t0, bool t1, double alpha, double beta){
  buffer_info m0_info = m0.request();
  buffer_info m1_info = m1.request();
  buffer_info m2_info = m2.request();
  const size_t m0_rows = static_cast<size_t>(m0_info.shape[0]);
  const size_t m0_cols = static_cast<size_t>(m0_info.shape[1]);
  const size_t m1_rows = static_cast<size_t>(m1_info.shape[0]);
  const size_t m1_cols = static_cast<size_t>(m1_info.shape[1]);

  const size_t m = t0 ? m0_cols : m0_rows;
  const size_t k0 = t0 ? m0_rows : m0_cols;
  const size_t k1 = t1 ? m1_cols : m1_rows;
  const size_t n = t1 ? m1_rows : m1_cols;

  assert(k0 == k1);
  assert(static_cast<size_t>(m2_info.shape[0]) == m &&
	 static_cast<size_t>(m2_info.shape[1]) == n);
  double* p0 = static_cast<double*>(m0_info.ptr);
  double* p1 = static_cast<double*>(m1_info.ptr);
  double* p2 = static_cast<double*>(m2_info.ptr);
  cblas_dgemm(CblasRowMajor,
  	      t0 ? CblasTrans : CblasNoTrans,
  	      t1 ? CblasTrans : CblasNoTrans,
  	      static_cast<int>(m),
	      static_cast<int>(n),
	      static_cast<int>(k0),
	      alpha,
	      p0,
	      static_cast<int>(m0_cols),
	      p1,
	      static_cast<int>(m1_cols),
	      beta,
	      p2,
	      static_cast<int>(n));
}



array_t<double> mat_inverse(array_t<double> m){
  buffer_info m_info = m.request();
  vector<double> minv(m_info.size);
  for(int i=0; i<m_info.shape[0]; i++)
    for(int j=0; j<m_info.shape[1]; j++)
      minv[i*m_info.shape[1]+j] = m.at(i, j);
  
  vector<int> pivot(m_info.shape[0]);
  LAPACKE_dgetrf(CblasRowMajor, m_info.shape[0], m_info.shape[1],
		 &minv.front(), m_info.shape[1], &pivot.front());
  LAPACKE_dgetri(CblasRowMajor, m_info.shape[0],
		 &minv.front(), m_info.shape[1], &pivot.front());
  array_t<double> ret(m_info.shape);
  for(int i=0; i<m_info.shape[0]; i++)
    for(int j=0; j<m_info.shape[1]; j++)
      ret.mutable_at(i, j) = minv[i*m_info.shape[1]+j];
  return ret;
}

void add_Gij(array_t<double> Yp_ij_g,
	     array_t<double> EZ1i_g,    array_t<double> EZ2ij_g,
	     array_t<double> EYmZ1T_g,  array_t<double> EYmZ2T_g,
	     array_t<bool>   obs_mask,  array_t<int>    Yp_index,
	     array_t<bool>   miss_mask, array_t<int>    EYm_index,
	     array_t<double> sG1,       array_t<double> sG2){  
  buffer_info EY1_info  =  EYmZ1T_g.request(),  EY2_info = EYmZ2T_g.request();
  buffer_info obs_info  =  obs_mask.request(),   Yp_info = Yp_index.request();
  buffer_info miss_info = miss_mask.request(),  EYm_info = EYm_index.request();
  buffer_info sG1_info  =       sG1.request(),  sG2_info = sG2.request();
  double* pEY1  = static_cast<double*>(EY1_info.ptr);
  double* pEY2  = static_cast<double*>(EY2_info.ptr);
  double* psG1  = static_cast<double*>(sG1_info.ptr);
  double* psG2  = static_cast<double*>(sG2_info.ptr);
  int*    pEYm  = static_cast<int*>(EYm_info.ptr);
  bool*   pobs  = static_cast<bool*>(obs_info.ptr);
  bool*   pmiss = static_cast<bool*>(miss_info.ptr);
  // separate column counts for sG1 and sG2
  const size_t ncol1 = (sG1_info.ndim >= 2) ? static_cast<size_t>(sG1_info.shape[1]) : 0; // d1
  const size_t ncol2 = (sG2_info.ndim >= 2) ? static_cast<size_t>(sG2_info.shape[1]) : 0; // d2
  // ---- observed part: Yp * EZ^T ----
  if (Yp_info.size > 0) {
    array_t<double> sYp = s_vector<double>(Yp_ij_g, Yp_index);

    // sG1 (only if d1>0)
    if (ncol1 > 0) {
      array_t<double> Yp_EZ1 = outer_prod<double>(sYp, EZ1i_g);
      double* pYp_EZ1 = static_cast<double*>(Yp_EZ1.request().ptr);
      size_t m1 = 0;
      for (ssize_t i = 0; i < obs_info.size; ++i) {
        if (!pobs[i]) continue;
        size_t k = static_cast<size_t>(i) * ncol1;
        size_t l = m1 * ncol1;
        ++m1;
        for (size_t j = 0; j < ncol1; ++j) psG1[k + j] += pYp_EZ1[l + j];
      }
    }

    // sG2 (independent of d1)
    if (ncol2 > 0) {
      array_t<double> Yp_EZ2 = outer_prod<double>(sYp, EZ2ij_g);
      double* pYp_EZ2 = static_cast<double*>(Yp_EZ2.request().ptr);
      size_t m2 = 0;
      for (ssize_t i = 0; i < obs_info.size; ++i) {
        if (!pobs[i]) continue;
        size_t k = static_cast<size_t>(i) * ncol2;
        size_t l = m2 * ncol2;
        ++m2;
        for (size_t j = 0; j < ncol2; ++j) psG2[k + j] += pYp_EZ2[l + j];
      }
    }
  }

  // ---- missing part: E[Y_m Z^T] ----
  if (EYm_info.size > 0 && (EY1_info.shape[0] > 0 || EY2_info.shape[0] > 0)) {
    // into sG1 from EYmZ1T
    if (ncol1 > 0 && EY1_info.shape[0] > 0) {
      size_t m1 = 0;
      for (ssize_t i = 0; i < miss_info.size; ++i) {
        if (!pmiss[i]) continue;
        size_t k = static_cast<size_t>(i) * ncol1;
        size_t l = static_cast<size_t>(pEYm[m1]) * ncol1;
        ++m1;
        for (size_t j = 0; j < ncol1; ++j) psG1[k + j] += pEY1[l + j];
      }
    }

    // into sG2 from EYmZ2T
    if (ncol2 > 0 && EY2_info.shape[0] > 0) {
      size_t m2 = 0;
      for (ssize_t i = 0; i < miss_info.size; ++i) {
        if (!pmiss[i]) continue;
        size_t k = static_cast<size_t>(i) * ncol2;
        size_t l = static_cast<size_t>(pEYm[m2]) * ncol2;
        ++m2;
        for (size_t j = 0; j < ncol2; ++j) psG2[k + j] += pEY2[l + j];
      }
    }
  }
}

PYBIND11_MODULE(hppca_bindings, m){
  m.def("set_cblas_threads", &set_cblas_threads, "set number of cblas threads");
  m.def("zeros_int",&zeros<int>, "matrix of zeros, type int");
  m.def("zeros_dbl", &zeros<double>, "matrix of zeros, type float64");
  m.def("set_all_int", &set_all<int>, "set all values in matrix, type int");
  m.def("set_all_dbl", &set_all<double>,
	"set all values in matrix, type float64");
  m.def("outer_prod_int", &outer_prod<int>,
	"outer product of two numpy vectors of type int");
  m.def("outer_prod_dbl", &outer_prod<double>,
	"outer product of two numpy vectors of type float64");
  m.def("trace_prod_int", &trace_prod<int>,
	"trace of the product of 2 matrices of type int");
  m.def("trace_prod_dbl", &trace_prod<double>,
	"trace of the product of 2 matrices of type float64");
  m.def("trace_prod_t_int", &trace_prod_t<int>,
	"trace of the product of 2 matrices with the 2nd transposed of type int");
  m.def("trace_prod_t_dbl", &trace_prod_t<double>,
	"trace of the product of 2 matrices with the 2nd transposed of type float64");
  m.def("mat_prod", &mat_prod, "matrix product");
  m.def("mat_inverse", &mat_inverse, "matrix inverse");
  m.def("sub_vector_int", &sub_vector<int>,
	"subvector from range of rows, type int");
  m.def("sub_vector_dbl", &sub_vector<double>,
	"suvector from range of rows, type float64");
  m.def("s_vector_int", &s_vector<int>,
	"subvector from list of rows, type int");
  m.def("s_vector_dbl", &s_vector<double>,
	"subvector from list of rows, type float64");
  m.def("sub_matrix_int", &sub_matrix<int>,
	"submatrix from range of rows and cols, type int");
  m.def("sub_matrix_dbl", &sub_matrix<double>,
	"submatrix from range of rows and cols, type float64");  
  m.def("s_matrix_int", &s_matrix<int>,
	"submatrix from list of rows and cols, type int");
  m.def("s_matrix_dbl", &s_matrix<double>,
	"submatrix from list of rows and cols, type float64");
  m.def("add_identity_int", &add_identity<int>,
	"add a constant times the identity, type int");
  m.def("add_identity_dbl", &add_identity<double>,
	"add a constant times the identity, type float64");
  m.def("add_Gij", &add_Gij,
	"function to replace construct_Gij_vectorized");
}
