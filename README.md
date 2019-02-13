# Overview
This is a MATLAB extension for the Walsh-Hadamard transform that outperforms the [Fast Walsh-Hadamard Transform](https://www.mathworks.com/help/signal/ref/fwht.html) that MATLAB provides.

The files are as follows:
* **hadamard_pthreads.c** - Multithreaded implementation using pthreads. No SIMD instructions or loop unrolling.
* **hadamard_openmp.c** - Multithreaded implementation using OpenMP.  No SIMD instructions or loop unrolling.
* **hadamard_avx.c** - Multithreaded implementation using OpenMP. AVX instructions but no loop unrolling.
* **hadamard_unrolling.c** - Multithreaded implementation using OpenMP. AVX instructions and loop unrolling.

# Compilation
For GCC-based `mex`, the extension can be compiled as follows:
```
mex hadamard_XYZ.c CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"
```
For MSVC-based `mex`, the extension can be compiled as follows:
```
mex hadamard_XYZ.c COMPFLAGS="/openmp"
```

For LLVM/Clang, e.g., Mac OS, first run `brew install libomp` then try
```
mex hadamard_XYZ.c CFLAGS="\$CFLAGS -Xpreprocessor -fopenmp -lomp -I/usr/local/opt/libomp/include -L/usr/local/opt/libomp/lib"
```
but you may need more hacking

On all systems, if you just want a basic, non-multi-threaded version, do
```
mex hadamard_openmp.c
```

Note that using the extensions resulting from `hadamard_avx.c` and `hadamard_unrolling.c` requires the AVX instruction set.

# Usage
MATLAB allows the compiled extension to be used as a traditional function. For example, the following would compute the WHT of the 8x8 identity matrix using the extension compiled from the `hadamard_unrolling.c` file:
```
hadamard_unrolling(eye(8))
```
Note that the behavior of this WHT differs from the defaults of MATLAB's `fwht` function. In particular, this WHT is equivalent to `length(x) * fwht( x, [], 'hadamard' )`.

# Acknowledgements
Base versions written by Peter Stobbe from Caltech and Stephen Becker from University of Colorado Boulder. More details found in comments at top of files.

# Timing example
Running on a 2014 Macbook Pro, compared to Matlab's implementation, here are some speeds:
```
>>> mex hadamard_openmp.c
>>> x   = randn(2^20,10);
>>> tic;
>>> y1 = hadamard_openmp(x); % no multithreading
>>> toc
>>> tic;
>>> y2 = size(x,1) * fwht( x, [], 'hadamard' );
>>> toc
>>> norm( y1 - y2 )
```
which gives output
```
Elapsed time is 0.288349 seconds.
Elapsed time is 8.677412 seconds.

ans =

   3.3990e-10
```
