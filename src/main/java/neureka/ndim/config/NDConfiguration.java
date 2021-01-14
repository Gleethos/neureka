/*
MIT License

Copyright (c) 2019 Gleethos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   _   _ _____   _____             __ _                       _   _
  | \ | |  __ \ / ____|           / _(_)                     | | (_)
  |  \| | |  | | |     ___  _ __ | |_ _  __ _ _   _ _ __ __ _| |_ _  ___  _ __
  | . ` | |  | | |    / _ \| '_ \|  _| |/ _` | | | | '__/ _` | __| |/ _ \| '_ \
  | |\  | |__| | |___| (_) | | | | | | | (_| | |_| | | | (_| | |_| | (_) | | | |
  |_| \_|_____/ \_____\___/|_| |_|_| |_|\__, |\__,_|_|  \__,_|\__|_|\___/|_| |_|
                                         __/ |
                                        |___/
*/

package neureka.ndim.config;

import neureka.Neureka;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;

public interface NDConfiguration
{
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    int rank();

    int[] shape();

    int shape( int i );

    int[] idxmap();

    int idxmap( int i );

    int[] translation();

    int translation( int i );

    int[] spread();

    int spread( int i );

    int[] offset();

    int offset( int i );

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    int i_of_i( int i );

    int[] idx_of_i( int i );

    int i_of_idx( int[] idx );

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    int[] asInlineArray();

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    long keyCode();

    boolean equals(NDConfiguration ndc);

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    NDConfiguration newReshaped(int[] newForm);

    public class Utility {

        @Contract(pure = true)
        public static int[] rearrange(int[] tln, int[] shp, @NotNull int[] newForm) {
            int[] shpTln = newTlnOf(shp);
            int[] newTln = new int[newForm.length];
            for ( int i = 0; i < newForm.length; i++ ) {
                if (newForm[ i ] < 0) newTln[ i ] = shpTln[ i ];
                else if (newForm[ i ] >= 0) newTln[ i ] = tln[newForm[ i ]];
            }
            return newTln;
        }

        @Contract(pure = true)
        public static int[] rearrange(@NotNull int[] array, @NotNull int[] ptr) {
            int[] newShp = new int[ptr.length];
            for ( int i = 0; i < ptr.length; i++ ) {
                if (ptr[ i ] < 0) newShp[ i ] = Math.abs(ptr[ i ]);
                else if (ptr[ i ] >= 0) newShp[ i ] = array[ptr[ i ]];
            }
            return newShp;
        }

        @Contract(pure = true)
        public static int[] newTlnOf(int[] shape) {
            int[] tln = new int[ shape.length ];
            int prod = 1;
            for ( int i = tln.length-1; i >= 0; i-- ) {
                tln[ i ] = prod;
                prod *= shape[ i ];
            }
            return tln;
        }

        @Contract(pure = true)
        public static void increment( @NotNull int[] shpIdx, @NotNull int[] shape ) {
            int i = shape.length-1;
            while ( i >= 0 && i < shape.length ) i = _incrementAt( i, shpIdx, shape );
        }

        @Contract(pure = true)
        private static int _incrementAt( int i, @NotNull int[] shpIdx, @NotNull int[] shape )
        {
            if ( shpIdx[ i ] < shape[ i ] ) {
                shpIdx[ i ]++;
                if ( shpIdx[ i ] == shape[ i ] ) {
                    shpIdx[ i ] = 0;
                    i--;
                }
                else i = -1;
            }
            else i--;
            return i;
        }

        @Contract(pure = true)
        public static int szeOfShp( int[] shape ) {
            int size = 1;
            for ( int i : shape ) size *= i;
            return size;
        }


    }

}
