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

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 *  This interface represents the access pattern configuration for the data array of a tensor.
 */
public interface NDConfiguration
{
    /**
     *  Types of common data layouts.
     */
    enum Layout {

        ROW_MAJOR, COLUMN_MAJOR, UNSPECIFIC;

        @Contract(pure = true)
        public int[] newTranslationFor( int[] shape ) {
            int[] translation = new int[ shape.length ];
            int prod = 1;
            if ( this == COLUMN_MAJOR ) {
                for ( int i = 0; i < translation.length; i++ ) {
                    translation[ i ] = prod;
                    prod *= shape[ i ];
                }
            } else if ( this == ROW_MAJOR || this == UNSPECIFIC ) {
                for ( int i = translation.length - 1; i >= 0; i-- ) {
                    translation[i] = prod;
                    prod *= shape[i];
                }
            }
            else
                throw new IllegalStateException("Unknown data layout!");

            return translation;
        }

        @Contract(pure = true)
        public int[] rearrange(int[] tln, int[] shape, int[] newForm) {
            int[] shpTln = this.newTranslationFor( shape );
            int[] newTln = new int[ newForm.length ];
            for ( int i = 0; i < newForm.length; i++ ) {
                if ( newForm[ i ] < 0 ) newTln[ i ] = shpTln[ i ];
                else if ( newForm[ i ] >= 0 ) newTln[ i ] = tln[ newForm[ i ] ];
            }
            return newTln;
        }
    }

    /**
     *  The layout of a tensor can either be row major or column major.
     *  Row major means that row elements are right next to one another
     *  in the underlying data array of a tensor.
     *  Column major is the exact opposite...
     *
     * @return The layout of the underlying data array of a tensor.
     */
    default Layout getLayout()
    {
        if ( !this.isCompact() )
            return Layout.UNSPECIFIC;
        else {
            int[] translation1 = Layout.ROW_MAJOR.newTranslationFor(this.shape());
            boolean basicIndices = Arrays.equals(translation1, indicesMap());

            if ( Arrays.equals(translation1, translation()) && basicIndices )
                return Layout.ROW_MAJOR;

            int[] translation2 = Layout.COLUMN_MAJOR.newTranslationFor(this.shape());
            if ( Arrays.equals(translation2, translation()) && basicIndices )
                return Layout.COLUMN_MAJOR;

        }
        return Layout.UNSPECIFIC;
    }

    /**
     *  A slice has a unique kind of access pattern which require
     *  a the {@link neureka.ndim.iterator.NDIterator} for iterating over their data.
     *  This is contrary to the {@link #isSimple()} flag which guarantees
     *  a simple access pattern.
     *
     * @return The truth value determining if this ND-Configuration models a slice index pattern.
     */
    default boolean isSlice() {
        return !this.isSimple();
    }

    /**
     *  This method returns the number of axis of
     *  a nd-array / {@link neureka.Tsr} which is equal to the
     *  length of the shape of an nd-array / {@link neureka.Tsr}.
     *
     * @return The number of axis of an nd-array.
     */
    int rank();

    default int size() {
        return Arrays.stream(shape()).reduce(1, (a, b) -> a * b );
    }

    /**
     *  This method returns an array of axis sizes.
     *
     * @return An array of axis sizes.
     */
    int[] shape();

    /**
     *  This method receives an axis index and return the
     *  size of the axis.
     *  It enables readable access to the shape
     *  of this configuration.
     *
     * @param i The index of the axis whose size ought to be returned.
     * @return The axis size targeted by the provided index.
     */
    int shape( int i );

    int[] indicesMap();

    int indicesMap( int i );

    int[] translation();

    /**
     *  This method receives an axis index and return the
     *  translation value for the targeted axis.
     *  It enables readable and fast access to the translation
     *  of this configuration.
     *
     * @param i The index of the axis whose translation ought to be returned.
     * @return The axis translation targeted by the provided index.
     */
    int translation( int i );

    /**
     *  The spread is the access step size of a slice within the n-dimensional
     *  data array of its parent tensor.
     *
     * @return An array of index step sizes for each tensor dimension / axis.
     */
    int[] spread();

    /**
     *  The spread is the access step size of a slice within the n-dimensional
     *  data array of its parent tensor.
     *  Use this to look up the spread in a particular dimension / axis.
     *
     * @param i The dimension / axis index of the dimension / axis whose spread should be returned.
     * @return The spread of the targeted dimension.
     */
    int spread( int i );

    /**
     *  The offset is the position of a slice within the n-dimensional
     *  data array of its parent tensor.
     *  Use this to get the offsets of all slice dimension.
     *
     * @return The offset position of the slice tensor inside the n-dimensional data array of the parent tensor.
     */
    int[] offset();

    /**
     *  The offset is the position of a slice within the n-dimensional
     *  data array of its parent tensor.
     *  Use this to look up the offset in a particular dimension / axis.
     *
     * @param i The dimension / axis index of the dimension / axis whose offset should be returned.
     * @return The offset of the targeted dimension.
     */
    int offset( int i );

    /**
     *  Use this to calculate the true index for an element in the data array (data array index)
     *  based on a provided "virtual index", or "value array index".
     *  This virtual index may be different from the true index depending on the type of nd-array,
     *  like for example if the nd-array is
     *  a slice of another larger nd-array, or if it is in fact a reshaped version of another nd-array.
     *  This virtual index ought to be turned into an index array which defines the position for every axis.
     *  Then this indices array will be converted into the final and true index targeting an underlying item.
     *  The information needed for performing this translation is expressed by individual implementations of
     *  this {@link NDConfiguration} interface, which contain everything
     *  needed to treat a given block of data as a nd-array!
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

    /**
     * The following method calculates the true index for an element in the data array
     * based on a provided index array.
     *
     * @param indices The indices for every axis of a given nd-array.
     * @return The true index targeting the underlying data array of a given nd-array.
     */
    int indexOfIndices( int[] indices );

    /**
     *  This method returns an array of flattened arrays which
     *  define this nd-configuration in a compact manner.
     *  The array consists of the following arrays joined
     *  in the following order:
     *  [ shape | translation | indicesMap | offsets | strides ]
     *
     * @return An array of flattened arrays which define this nd-configuration in a compact manner.
     */
    default int[] asInlineArray()
    {
        int rank = rank();
        int[] inline = new int[ rank * 5 ];
        //config format: [ shape | translation | indicesMap | offsets | strides ]
        System.arraycopy( shape(),       0, inline, rank * 0, rank ); //=> SHAPE
        System.arraycopy( translation(), 0, inline, rank * 1, rank ); //=> TRANSLATION
        System.arraycopy( indicesMap(),  0, inline, rank * 2, rank ); //=> INDICES MAP (translates scalar to n-dimensional index)
        System.arraycopy( offset(),      0, inline, rank * 3, rank ); //=> SPREAD / STRIDES (step size for dimensions in underlying parent tensor)
        System.arraycopy( spread(),      0, inline, rank * 4, rank ); //=> OFFSET (nd-position inside underlying parent tensor)
        return inline;
    }

    int hashCode();

    boolean equals( NDConfiguration ndc );

    /**
     *  This method enables reshaping for {@link NDConfiguration} implementation instances.
     *  Because {@link NDConfiguration}s are in essence things which define
     *  the access relationship from shape indices to the actual underlying data,
     *  the creation of reshaped {@link NDConfiguration} is up to a specific implementation.
     *
     * @param newForm An array of indices which define how the axis ought to be rearranged.
     * @return A new {@link NDConfiguration} which carries the needed information for the reshaped view.
     */
    NDConfiguration newReshaped( int[] newForm );

    /**
     *  The boolean returned by this method simply reports
     *  if this configuration is the most basic form of configuration
     *  possible for the given shape represented by this instance.
     *  This type of configuration is the typical for freshly created
     *  tensors which are neither slices nor reshaped variants of an
     *  original tensor...
     *  Therefore, such "simple tensors" do not need a fancy {@link neureka.ndim.iterator.NDIterator}
     *  in order to perform operations on them.
     *  One can simply iterate over their underlying data array.
     *
     * @return The truth value determining if this configuration is not modeling more complex indices like reshaped views or slices...
     */
    default boolean isSimple() {
        int[] simpleTranslation = this.getLayout().newTranslationFor( this.shape() );
        return Arrays.equals( this.translation(), simpleTranslation )
                &&
               Arrays.equals( this.indicesMap(), simpleTranslation )
                &&
               isCompact();
    }

    /**
     *  {@link NDConfiguration} instance where this flag is true
     *  will most likely not be slices because they have no offset (all 0)
     *  and a compact spread / stride array (all 1).
     *
     * @return The truth value determining if this configuration has no offset and spread/strides larger than 1.
     */
    default boolean isCompact() {
        return
            IntStream.range( 0, this.rank() ).allMatch( i -> this.spread(i) == 1 )
                    &&
            IntStream.range( 0, this.rank() ).allMatch( i -> this.offset(i) == 0 );
    }

    /**
     * @return The truth value determining if this {@link NDConfiguration}
     *         represents virtual tensors (see {@link neureka.Tsr#isVirtual()}).
     */
    default boolean isVirtual() { return false; }

    default IndexToIndexFunction getIndexToIndexAccessPattern() {
        NDConfiguration nda = this;
        return nda::indexOfIndex;
    }

    interface IndexToIndexFunction { int map( int i ); }

    /**
     *  This utility class provides static methods which are helpful
     *  for nd-configuration related operations like reshaping,
     *  incrementing or decrementing index arrays...
     */
    class Utility
    {
        @Contract(pure = true)
        public static int[] rearrange( int[] array, int[] pointers ) {
            int[] newShp = new int[ pointers.length ];
            for ( int i = 0; i < pointers.length; i++ ) {
                if ( pointers[ i ] < 0 ) newShp[ i ] = Math.abs( pointers[ i ] );
                else if ( pointers[ i ] >= 0 ) newShp[ i ] = array[ pointers[ i ] ];
            }
            return newShp;
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
        public static void decrement( int[] indices, int[] shape ) {
            int i = shape.length-1;
            while ( i >= 0 && i < shape.length ) i = _decrementAt( i, indices, shape );
        }

        @Contract(pure = true)
        private static int _decrementAt( int i, int[] indices, int[] shape )
        {
            if ( indices[ i ] >= 0 ) {
                indices[ i ]--;
                if ( indices[ i ] == -1 ) {
                    indices[ i ] = shape[ i ] - 1;
                    i--;
                }
                else i = -1;
            }
            else i--;
            return i;
        }


        @Contract(pure = true)
        public static int sizeOfShape( int[] shape ) {
            int size = 1;
            for ( int i : shape ) size *= i;
            return size;
        }

    }

}
