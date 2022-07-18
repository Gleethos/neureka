package neureka.ndim;

import org.jetbrains.annotations.Contract;

/**
 *  Static utility methods for the NDArray.
 */
public class NDUtil
{
    @Contract( pure = true )
    public static String shapeString( int[] conf ) {
        StringBuilder str = new StringBuilder();
        for ( int i = 0; i < conf.length; i++ )
            str.append(conf[ i ]).append((i != conf.length - 1) ? ", " : "");
        return "[" + str + "]";
    }

    @Contract(pure = true)
    public static int[][] makeFit( int[] sA, int[] sB ) {
        int lastIndexOfA = 0;
        for ( int i = sA.length-1; i >= 0; i-- ) {
            if ( sA[ i ] != 1 ) {
                lastIndexOfA = i;
                break;
            }
        }
        int firstIndexOfB = 0;
        for ( int i = 0; i < sB.length; i++ ) {
            if ( sB[ i ] != 1 ) {
                firstIndexOfB = i;
                break;
            }
        }
        int newSize = lastIndexOfA + sB.length - firstIndexOfB;
        int[] rsA = new int[ newSize ];
        int[] rsB = new int[ newSize ];
        for( int i = 0; i <newSize; i++ ) {
            if ( i <= lastIndexOfA ) rsA[ i ] = i; else rsA[ i ] = -1;
            if ( i >= lastIndexOfA ) rsB[ i ] = i - lastIndexOfA+firstIndexOfB; else rsB[ i ] = -1;
        }
        return new int[][]{ rsA, rsB };
    }
}
