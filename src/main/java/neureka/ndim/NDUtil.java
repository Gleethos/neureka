package neureka.ndim;


import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 *  Static utility methods for the NDArray.
 */
public class NDUtil
{

    static List<Integer> asList(int[] array ) {
        List<Integer> intList = new ArrayList<>( array.length );
        for ( int i : array ) intList.add( i );
        return Collections.unmodifiableList(intList);
    }

    public static String shapeString( int[] conf ) {
        StringBuilder str = new StringBuilder();
        for ( int i = 0; i < conf.length; i++ )
            str.append(conf[ i ]).append((i != conf.length - 1) ? ", " : "");
        return "[" + str + "]";
    }

}
