/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.matrix.operation;

import neureka.backend.standard.operations.linear.fast.array.DenseArray;
import neureka.backend.standard.operations.linear.fast.matrix.store.MatrixCore;

/**
 * <p>
 * Contents in this package loosely corresponds to BLAS. The exact selection of operations and their API:s are
 * entirely dictated by the requirements of the various {@linkplain MatrixCore}
 * implementations.
 * </p>
 * <ul>
 * <li>http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms</li>
 * <li>http://www.netlib.org/blas/</li>
 * <li>http://www.netlib.org/blas/faq.html</li>
 * <li>http://www.netlib.org/lapack/lug/node145.html</li>
 * </ul>
 * Basic Linear Algebra Subprograms (BLAS) Level 1 contains vector operations.
 * <ul>
 * <li><a href="http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_1">BLAS Level 1 @
 * WikipediA</a></li>
 * <li><a href="http://www.netlib.org/blas/#_level_1">BLAS Level 1 @ Netlib</a></li>
 * <li><a href="https://software.intel.com/en-us/node/520730">BLAS Level 1 @ Intel</a></li>
 * </ul>
 * For each operation there should be 2 sets of implementations:
 * <ol>
 * <li>Optimised to be the implementations for the {@linkplain DenseArray} instances.</li>
 * <li>Optimised to be building blocks for higher level algorithms</li>
 * </ol>
 * Basic Linear Algebra Subprograms (BLAS) Level 2 contains matrix-vector operations.
 * <ul>
 * <li><a href="http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_2">BLAS Level 2 @
 * WikipediA</a></li>
 * <li><a href="http://www.netlib.org/blas/#_level_2">BLAS Level 2 @ Netlib</a></li>
 * <li><a href="https://software.intel.com/en-us/node/520748">BLAS Level 2 @ Intel</a></li>
 * </ul>
 * Basic Linear Algebra Subprograms (BLAS) Level 3 contains matrix-matrix operations.
 * <ul>
 * <li><a href="http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3">BLAS Level 3 @
 * WikipediA</a></li>
 * <li><a href="http://www.netlib.org/blas/#_level_3">BLAS Level 3 @ Netlib</a></li>
 * <li><a href="https://software.intel.com/en-us/node/520774">BLAS Level 3 @ Intel</a></li>
 * </ul>
 *
 */
public interface MatrixOperation {

}
