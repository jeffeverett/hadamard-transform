/* Hadamard Transform
   mex function to take hadamard transform

   Usage: w = hadamard_pthreads(x)
   x must be a REAL VALUED COLUMN VECTOR or MATRIX
   m = size(x,1) must be a POWER OF TWO

   Notes:
   1) This implementation uses exactly m*log2(m) additions/subtractions.
   2) This is symmetric and orthogonal. To invert, apply again and
      divide by vector length.

   Written by: Peter Stobbe, Caltech
   Email: stobbe@acm.caltech.edu
   Created: August 2008
   Edits by Stephen Becker, 2009--2014
 
   This "pthread" version by Stephen Becker, May 13 2014

      Note: in R2008b, Matlab added "fwht" and "ifwht" (the Fast Walsh-
          Hadamart Transform and the inverse) to its Signal Processing
          Toolbox.  With the default ordering and scaling, it's not
          equivalent to this, but you can change this with the following:
          y = length(x) * fwht( x, [], 'hadamard' );
          Then y should be the same as hadamard(x) up to roundoff.
          However, it appears that this code is faster than fwht.

 Update Stephen Becker, Feb 27 2014, fix compiling issue for Mac OS X
 Update Stephen Becker, Mar  3 2014, issue error if input data is sparse
 
*/

#include <stdlib.h>

/* Using OpenMP and AVX with MEX script requires special compile flags.
  Use `mex hadamard_avx.c CFLAGS="\$CFLAGS -fopenmp -mavx" LDFLAGS="\$LDFLAGS -fopenmp"` for GCC-based mex */
#include <omp.h>
#include <x86intrin.h> 

/* SRB: Feb 27 2014, gcc-4.8 has problems with char16_t not being defined. 
 * This  seems to fix it
 * (and do this BEFORE including mex.h) */
/* See http://gcc.gnu.org/bugzilla/show_bug.cgi?id=56086#c4 */
#ifndef NO_UCHAR
#define UCHAR_OK
#endif
#if defined(__GNUC__) && !(defined(__clang__)) && defined(UCHAR_OK)
#include <uchar.h>
#endif

#include "mex.h"

#ifdef NTHREADS
#define NTHREADS_ NTHREADS
#else
#define NTHREADS_ 4
#endif

/* 
 y - output
 x - input
 m - length of vector
 */
void hadamard_apply_vector(double *y, double *x, unsigned m)
{
  unsigned bit, dbit, j, k, l, r, p;
  double temp;

  // handles length-2 transforms, and places data in output vector
  for (j = 0; j < m; j+=2) {
      k = j+1;
      y[j] = x[j] + x[k];
      y[k] = x[j] - x[k];
  }

  if (m < 4)
    return;

  // handles length-4 transforms
  static __m128d tmp1, tmp2, sum1, sub1;
  for (j = 0; j < m; j+=4) {
      k = j + 2;

      tmp1 = _mm_load_pd(y+j);
      tmp2 = _mm_load_pd(y+k);

      sum1 = _mm_add_pd(tmp1, tmp2);
      sub1 = _mm_sub_pd(tmp1, tmp2);

      _mm_store_pd(y+j, sum1);
      _mm_store_pd(y+k, sub1);
  }

  if (m < 8)
    return;

  // handles length-8 transforms
  static __m256d tmp3, tmp4, sum2, sub2;  
  for (j = 0; j < m; j+=8) {
      k = j + 4;

      tmp3 = _mm256_load_pd(y+j);
      tmp4 = _mm256_load_pd(y+k);
      
      sum2 = _mm256_add_pd(tmp3, tmp4);
      sub2 = _mm256_sub_pd(tmp3, tmp4);
    
      _mm256_store_pd(y+j, sum2);
      _mm256_store_pd(y+k, sub2);
  }

  if (m < 16)
    return;

  // handles length-16 transforms
  static __m256d tmp5, tmp6, sum3, sub3;  
  for (j = 0; j < m; j+=16) {
      k = j + 8;

      tmp3 = _mm256_load_pd(y+j);
      tmp4 = _mm256_load_pd(y+k);
      
      sum2 = _mm256_add_pd(tmp3, tmp4);
      sub2 = _mm256_sub_pd(tmp3, tmp4);
    
      _mm256_store_pd(y+j, sum2);
      _mm256_store_pd(y+k, sub2);

      l = j + 4;
      r = j + 12;

      tmp5 = _mm256_load_pd(y+l);
      tmp6 = _mm256_load_pd(y+r);
      
      sum3 = _mm256_add_pd(tmp5, tmp6);
      sub3 = _mm256_sub_pd(tmp5, tmp6);
    
      _mm256_store_pd(y+l, sum3);
      _mm256_store_pd(y+r, sub3);
  }

  if (m < 32)
    return;

  // handles length-32 transforms
  for (j = 0; j < m; j+=32) {
      k = j + 16;

      tmp3 = _mm256_load_pd(y+j);
      tmp4 = _mm256_load_pd(y+k);
      
      sum2 = _mm256_add_pd(tmp3, tmp4);
      sub2 = _mm256_sub_pd(tmp3, tmp4);
    
      _mm256_store_pd(y+j, sum2);
      _mm256_store_pd(y+k, sub2);

      l = j + 4;
      r = j + 20;

      tmp5 = _mm256_load_pd(y+l);
      tmp6 = _mm256_load_pd(y+r);
      
      sum3 = _mm256_add_pd(tmp5, tmp6);
      sub3 = _mm256_sub_pd(tmp5, tmp6);
    
      _mm256_store_pd(y+l, sum3);
      _mm256_store_pd(y+r, sub3);

      p = j + 8;
      k = j + 24;

      tmp3 = _mm256_load_pd(y+p);
      tmp4 = _mm256_load_pd(y+k);
      
      sum2 = _mm256_add_pd(tmp3, tmp4);
      sub2 = _mm256_sub_pd(tmp3, tmp4);
    
      _mm256_store_pd(y+p, sum2);
      _mm256_store_pd(y+k, sub2);

      l = j + 12;
      r = j + 28;

      tmp5 = _mm256_load_pd(y+l);
      tmp6 = _mm256_load_pd(y+r);
      
      sum3 = _mm256_add_pd(tmp5, tmp6);
      sub3 = _mm256_sub_pd(tmp5, tmp6);
    
      _mm256_store_pd(y+l, sum3);
      _mm256_store_pd(y+r, sub3);
  }

  // handles all remaining length-(bit*2) transforms
  for (bit = 32; bit < m; bit = dbit) {
      dbit = bit << 1;   
      for (j = 0; j < m; j += dbit) {
          for (l = j; l < j+bit; l+=4) {
            k = l | bit;

            tmp3 = _mm256_load_pd(y+l);
            tmp4 = _mm256_load_pd(y+k);
            
            sum2 = _mm256_add_pd(tmp3, tmp4);
            sub2 = _mm256_sub_pd(tmp3, tmp4);
          
            _mm256_store_pd(y+l, sum2);
            _mm256_store_pd(y+k, sub2);
          }
      }
  }
}

void hadamard_apply_matrix_threads(double *y, double *x, unsigned m, unsigned n)
{
    unsigned nThreads;

    if (n <= NTHREADS_)
      nThreads = n;
    else
      nThreads = NTHREADS_;

    #pragma omp parallel shared(y,x,m,n) num_threads(nThreads)
    {
        // Place declaration outside of for loop to support old OpenMP versions
        // that the MSVC supports.
        int j;
        #pragma omp for schedule(static)
        for (j = 0; j < n; j++)
        {
            double *threadX = x + j*m;
            double *threadY = y + j*m;
            hadamard_apply_vector(threadY, threadX, m);
        }
    }  
}


/* check that the vector length is a power of 2,
   just using bitshifting instead of log */
void checkPowerTwo(unsigned m)
{
    /* check that it's not a degenerate 0 by 1 vector or singleton */
    if (m <= 1) {
        mexErrMsgTxt("Vector length must be greater than 1.");
    }
    /* keep dividing by two until result is odd */
    while( (m & 1) == 0 ){
        m >>= 1;
    }
    /* check that m is not a multiple of an odd number greater than 1 */
    if (m > 1) {
        mexErrMsgTxt("Vector length must be power of 2.");
    }
}


/* The gateway routine. */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  double *x, *y;
  unsigned m, n;
    
  /* Check for the proper number of arguments. */
  if (nrhs != 1) {
    mexErrMsgTxt("One and only one input required; must be a column vector or matrix, with # rows a power of 2.");
  }
  if (nlhs > 1) {
    mexErrMsgTxt("Too many output arguments.");
  }

  /* input size */
  m = mxGetM(prhs[0]);
  checkPowerTwo(m);
  n = mxGetN(prhs[0]);
  
  if (mxIsComplex(prhs[0])) {
    mexErrMsgTxt("Input must be real.");   
  } else if (mxIsSparse(prhs[0])) {
    mexErrMsgTxt("Input must be a full matrix, not sparse.");   
  } else if (!mxIsDouble(prhs[0])) {
    mexErrMsgTxt("Input must be of type double.");      
  }
  
  /* Create matrix for the return argument. */
  plhs[0] = mxCreateDoubleMatrix(m, n, mxREAL);
  
  /* Assign pointers to each input and output. */
  x = mxGetPr(prhs[0]);
  y = mxGetPr(plhs[0]);
  
  /* Call the C subroutine. */
  hadamard_apply_matrix_threads(y, x, m, n);
  return;
}