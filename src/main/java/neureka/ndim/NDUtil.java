package neureka.ndim;


import neureka.Neureka;
import neureka.Shape;
import neureka.Tsr;
import neureka.math.Function;

/**
 *  Static utility methods for the NDArray.
 */
public class NDUtil
{
    
    public static String shapeString( int[] conf ) {
        StringBuilder str = new StringBuilder();
        for ( int i = 0; i < conf.length; i++ )
            str.append(conf[ i ]).append((i != conf.length - 1) ? ", " : "");
        return "[" + str + "]";
    }

    
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

    public static boolean canBeBroadcast(Shape a, Shape b) {
        if ( a.size() != b.size() ) return false;
        boolean areEqual = a.equals(b);
        if ( areEqual ) return true;
        for ( int i = 0; i < a.size(); i++ )
            if ( a.get(i) != b.get(i) && a.get(i) != 1 && b.get(i) != 1 )
                return false;

        return true;
    }

    public static <T> Tsr<T> transpose( Tsr<T> t ) {
        if ( t.rank() == 1 ) return t;
        if ( t.rank() == 2 ) {
            boolean wasIntermediate = t.isIntermediate();
            t.getMut().setIsIntermediate(false);
            Tsr<T> result = Neureka.get().backend().getFunction().transpose2D().call(t);
            t.getMut().setIsIntermediate(wasIntermediate);
            return result;
        }
        StringBuilder operation = new StringBuilder();
        for ( int i = t.rank() - 1; i >= 0; i-- ) operation.append( i ).append( i == 0 ? "" : ", " );
        operation = new StringBuilder( "[" + operation + "]:(I[ 0 ])" );
        return Function.of( operation.toString(), false ).call( t );
    }
}
