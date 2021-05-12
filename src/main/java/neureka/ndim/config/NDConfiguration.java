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

import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;

public interface NDConfiguration
{
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    int rank();

    int[] shape();

    int shape( int i );

    int[] indicesMap();

    int indicesMap( int i );

    int[] translation();

    int translation( int i );

    int[] spread();

    int spread( int i );

    int[] offset();

    int offset( int i );

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    /**
     *  The following method calculates the true index for an element in the data array
     *  based on a provided "virtual index".
     *  This virtual index might be different from the true index, for example because the nd-array is
     *  a slice of another larger nd-array, or maybe because it is in fact a reshaped version of another nd-array.
     *  This virtual index will be turned into an index array which defines the position for every axis.
     *  Then this index array will be converted into the final and true index targeting an underlying item.
     *  The information needed for performing this translation is contained within an implementation of
     *  the {@link NDConfiguration} interface array, which contains everything
     *  needed to treat a given block of data as an nd-array!
     *
     * @param index The virtual index of the tensor having this configuration.
     * @return The true index which targets the actual data within the underlying data array of an nd-array / tensor.
     */
    int indexOfIndex( int index );

    /**
     *  The following method calculates the axis indices for an element in the nd-array array
     *  based on a provided "virtual index".
     *  The resulting index defines the position of the element for every axis.
     *
     * @param index The virtual index of the tensor having this configuration.
     * @return The position of the (virtually) targeted element represented as an array of axis indices.
     */
    int[] indicesOfIndex( int index );

    int indexOfIndices( int[] indices );

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    int[] asInlineArray();

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    long keyCode();

    boolean equals(NDConfiguration ndc);

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    NDConfiguration newReshaped(int[] newForm);

    class Utility {

        @Contract(pure = true)
        public static int[] rearrange(int[] tln, int[] shape, int[] newForm) {
            int[] shpTln = newTlnOf( shape );
            int[] newTln = new int[ newForm.length ];
            for ( int i = 0; i < newForm.length; i++ ) {
                if ( newForm[ i ] < 0 ) newTln[ i ] = shpTln[ i ];
                else if ( newForm[ i ] >= 0 ) newTln[ i ] = tln[ newForm[ i ] ];
            }
            return newTln;
        }

        @Contract(pure = true)
        public static int[] rearrange(int[] array, int[] ptr) {
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
        public static void increment( int[] indices, int[] shape ) {
            int i = shape.length-1;
            while ( i >= 0 && i < shape.length ) i = _incrementAt( i, indices, shape );
        }

        @Contract(pure = true)
        private static int _incrementAt( int i, int[] indices, int[] shape )
        {
            if ( indices[ i ] < shape[ i ] ) {
                indices[ i ]++;
                if ( indices[ i ] == shape[ i ] ) {
                    indices[ i ] = 0;
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
