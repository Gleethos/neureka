package neureka.devices.opencl.utility;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public final class DispatchUtility {

    private DispatchUtility() {/* This is a utility class! */}

    private static void _revert( int[] array ) {
        for( int i = 0; i < array.length / 2; i++ )
        {
            int temp = array[ i ];
            array[ i ] = array[ array.length - i - 1 ];
            array[ array.length - i - 1 ] = temp;
        }
    }

    private static int _productOf( int[] array ) {
        int product = 1;
        for ( int i : array ) product *= i;
        return product;
    }

    public static int[] parseTile( int size, int[] shape )
    {
        double root = Math.pow(size, 1/(double)shape.length);
        int center = (int) root;
        int[] current = new int[ shape.length ];
        Arrays.fill( current, center );
        for ( int i = 0; i < shape.length; i++ )
            if ( current[ i ] > shape[ i ] ) current[ i ] = shape[ i ];
        int[][] factors = new int[shape.length][];
        for ( int si = 0; si < shape.length; si++ ) factors[si] = _primeFactors(shape[si]);
        int[] indices = new int[ shape.length ];
        for ( int si = 0; si < shape.length; si++ ) {
            int fi = -1;
            int product = 1;
            do {
                fi++;
                int previousDelta = Math.abs( current[ si ] - product );
                product *= factors[ si ][ fi ];
                int newDelta = Math.abs( current[ si ] - product );
                if ( product > shape[ si ] || previousDelta <= newDelta ) {
                    product /= factors[ si ][ fi ];
                    break;
                }
            }
            while( product < current[ si ] );

            int intermediateProduct = product;
            int intermediateIndex = fi;

            // Now the same in reverse :

            fi = factors[ si ].length;
            product = 1;
            do {
                fi--;
                int previousDelta = Math.abs(current[ si ]-product);
                product *= factors[ si ][ fi ];
                int newDelta = Math.abs(current[ si ]-product);
                if ( product > shape[ si ] || previousDelta <= newDelta ) {
                    product /= factors[ si ][ fi ];
                    _revert( factors[ si ] );
                    fi = factors[ si ].length - 1 - fi;
                    break;
                }
            }
            while( product < current[ si ] );

            boolean firstWasBetter = (
                    Math.abs(current[ si ]-intermediateProduct)
                            <
                    Math.abs(Math.abs(current[ si ]-product))
            );
            current[ si ] = (firstWasBetter) ? intermediateProduct : product;
            indices[ si ] = (firstWasBetter) ? intermediateIndex : fi;
        }

        /*
           Setup done, we've got a basic set of tile dimensions
           which is close to a quadratic / cubic / ... dimensionality !
           Now let's try to find a product that fits our desired size better :
        */
        int totalProduct = _productOf( current );
        int productDelta = Math.abs( size - totalProduct );
        do {
            int bestIndex = -1;
            int lowest = Integer.MAX_VALUE;
            double bestRatio = 1.0;
            for ( int i = 0; i < shape.length; i++ )
            {
                int found = ( factors[ i ].length > indices[ i ] + 1 )
                                ? factors[ i ][ indices[ i ]+1 ]
                                : Integer.MAX_VALUE ;
                double ratio = (found == Integer.MAX_VALUE)
                                    ? 1.0
                                    : (double)current[ i ] / (double)shape[ i ];
                if (
                        lowest == -1 ||
                        found < lowest ||
                        found <= lowest && ratio < bestRatio
                ) {
                    lowest = found;
                    bestIndex = i;
                    bestRatio = ratio;
                    assert indices[bestIndex] > -1;
                }
            }
            if ( bestIndex == -1 ) break; // nothing found :/

            int newTotalProduct = _productOfNewDimension(
                    current,
                    bestIndex,
                    lowest,
                    size
            );
            int newProductDelta = Math.abs(size-newTotalProduct);
            if ( productDelta > newProductDelta ) {
                productDelta = newProductDelta;
                current[bestIndex] *= factors[ bestIndex ][ indices[bestIndex]+1 ];
            }
            else break;

        } while ( true );

        // Maybe we went about some dimensions the wrong way...
        // Let's check if maybe could reverse product relations and get better configurations :

        totalProduct = _productOf( current );
        productDelta = Math.abs(size-totalProduct);
        do {
            int bestIndex = -1;
            int bestInversionDelta = Integer.MAX_VALUE;
            for ( int i=0; i<shape.length; i++)
            {
                int inversionDelta = _inversionProductOfNewDimension(
                        shape, current, i, size
                );
                if ( inversionDelta < productDelta ) {
                    bestIndex = i;
                    bestInversionDelta = inversionDelta;
                }
            }
            if (bestIndex == -1 ) break; // nothing found :/

            productDelta = bestInversionDelta;
            current[bestIndex] = shape[bestIndex] / current[bestIndex];

        } while ( true );

        return current;
    }


    private static int _productOfNewDimension(
            int[] current,
            int index,
            int alteration,
            int size
    ) {
        int[] copy = Arrays.copyOf(current, current.length);
        copy[index] *= alteration;
        int product = _productOf( copy );
        return Math.abs(size-product);
    }

    private static int _inversionProductOfNewDimension(
            int[] shape,
            int[] current,
            int index,
            int size
    ) {
        int[] copy = Arrays.copyOf(current, current.length);
        copy[index] = shape[index] / copy[index];
        int product = _productOf( copy );
        return Math.abs(size-product);
    }

    private static int[] _primeFactors(int n )
    {
        List<Integer> factors = new ArrayList<>();

        // Print the number of 2s that divide n
        while ( n % 2 == 0 ) {
            factors.add( 2 );
            n /= 2;
        }

        // n must be odd at this point.  So we can
        // skip one element (Note i = i +2)
        for ( int i = 3; i <= Math.sqrt(n); i += 2 ) {
            // While i divides n, print i and divide n
            while ( n % i == 0 ) {
                factors.add( i );
                n /= i;
            }
        }

        // This condition is to handle the case when
        // n is a prime number greater than 2
        if ( n > 2 ) factors.add( n );
        return factors.stream().mapToInt( p -> p ).toArray();
    }


    public static int[] findBestParams (
            int local_size,
            int reg_size,
            int com, int row, int col

    ) {
        // We know the global size :
        //[] global = new long[]{com, col};
        //=================
        // GOALS :
        int[] row_com_col = bestMatMulMatch(local_size, row, col, com);
        int max_ts_row = row_com_col[ 0 ];//   = 128, // ts := tile size
        int max_ts_col = row_com_col[ 2 ];//   = 128,
        int max_ts_com = row_com_col[ 1 ];//   = 16,

        int[] wpt_row_col = parseTile(reg_size, new int[]{max_ts_row, max_ts_col});
        int max_wpt_row = wpt_row_col[ 0 ];//  = 8,   // wpt := work per thread
        int max_wpt_col = wpt_row_col[ 1 ]; // = 8,
        //---

        return new int[]{max_ts_row, max_ts_col, max_ts_com, max_wpt_row, max_wpt_col};
    }


    public static int[] bestMatMulMatch(int size, int row, int col, int com)
    {
        int[] row_com = DispatchUtility.parseTile(size, new int[]{row, com});
        int[] col_com = DispatchUtility.parseTile(size, new int[]{col, com});

        int delta1 = Math.abs((row_com[ 0 ] * row_com[ 1 ] + row_com[ 0 ] * col_com[ 0 ])-size);
        int delta2 = Math.abs((col_com[ 0 ] * col_com[ 1 ] + col_com[ 0 ] * row_com[ 0 ])-size);

        if ( delta1 > delta2 ) return new int[]{ row_com[ 0 ], col_com[ 1 ], col_com[ 0 ] };
        else return new int[]{ row_com[ 0 ], row_com[ 1 ], col_com[ 0 ] };
    }



}
