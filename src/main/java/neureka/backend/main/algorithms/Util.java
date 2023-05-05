package neureka.backend.main.algorithms;

import neureka.Neureka;
import neureka.Tsr;
import neureka.math.Function;

public class Util {

    public static <T> Tsr<T> transpose(Tsr<T> t ) {
        if ( t.rank() == 1 ) return t;
        if ( t.rank() == 2 ) {
            boolean wasIntermediate = t.isIntermediate();
            t.getMut().setIsIntermediate(false);
            Tsr<T> result = Neureka.get().backend().getFunction().transpose2D().call(t);
            t.getMut().setIsIntermediate(wasIntermediate);
            return result;
        }
        StringBuilder operation = new StringBuilder();
        for ( int i = 0; i < t.rank()-2; i++ )
            operation.append( i ).append( ", " );

        // The last 2 dimensions are swapped:
        operation.append( t.rank()-1 ).append( ", " ).append( t.rank()-2 );
        operation = new StringBuilder( "[" + operation + "]:(I[ 0 ])" );
        return Function.of( operation.toString(), false ).call( t );
    }

}
