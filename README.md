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

Note that using the extensions resulting from `hadamard_avx.c` and `hadamard_unrolling.c` requires the AVX instruction set.

# Usage
MATLAB allows the compiled extension to be used as a traditional function. For example, the following would compute the WHT of the 8x8 identity matrix using the extension compiled from the `hadamrd_unrolling.c` file:
```
hadamard_unrolling(eye(8))
```
Note that the behavior of this WHT differs from the defaults of MATLAB's `fwht` function. In particular, this WHT is equivalent to `length(x) * fwht( x, [], 'hadamard' )`.

# Acknowledgements
Base versions written by Peter Stobbe from Caltech and Stephen Becker from University of Colorado Boulder. More details found in comments at top of files.
