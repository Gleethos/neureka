/*<#LICENSE#>*/
package neureka.backend.main.operations.linear.internal.blas;

/**
 * The ?axpy routines perform a vector-vector operation defined as y := a*x + y where: a is a scalar x and y
 * are vectors each with a number of elements that equals n. <code>y[] += a * x[]</code>
 *
 */
public final class AXPY {

    public static void invoke(
            final double[] y,
            final int yOffset,
            final double multiplier,
            final double[] x,
            final int xOffset,
            final int start,
            final int limit
    ) {
        for (int i = start; i < limit; i++) {
            y[yOffset + i] += multiplier * x[xOffset + i];
        }
    }

    public static void invoke(
            final float[] y,
            final int basey,
            final float a,
            final float[] x,
            final int basex,
            final int first,
            final int limit
    ) {
        for (int i = first; i < limit; i++) {
            y[basey + i] += a * x[basex + i];
        }
    }

}
