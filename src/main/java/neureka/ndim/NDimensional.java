package neureka.ndim;

import neureka.Tsr;
import neureka.ndim.config.NDConfiguration;

import java.util.ArrayList;
import java.util.List;

public interface NDimensional {

    /**
     * @return The number of dimensions of this tensor / nd-array.
     */
    default int rank() { return getNDConf().rank(); }

    /**
     * @return The number of dimensions of this tensor / nd-array.
     */
    default int getRank() { return rank(); }

    /**
     * @return A list of the dimensions of this tensor / array.
     */
    default List<Integer> shape() { return Util.asList(getNDConf().shape()); }
    /**
     * @return A list of the dimensions of this tensor / array.
     */
    default List<Integer> getShape() { return shape(); }

    default List<Integer> indicesMap() { return Util.asList(getNDConf().indicesMap()); }

    default List<Integer> translation() { return Util.asList(getNDConf().translation()); }

    default List<Integer> spread() { return Util.asList(getNDConf().spread()); }

    default List<Integer> offset() { return Util.asList(getNDConf().offset()); }

    /**
     * @return The {@link NDConfiguration} implementation instance of this {@link Tsr} storing dimensionality information.
     */
    NDConfiguration getNDConf();

    default int shape( int i ) { return getNDConf().shape()[ i ]; }

    /**
     * @return The number of elements stored inside the tensor.
     */
    default int size() { return NDConfiguration.Utility.sizeOfShape(getNDConf().shape()); }

    /**
     * @return The number of elements stored inside the tensor.
     */
    default int getSize() { return size(); }

    /**
     *  This is a convenience method identical to {@code tensor.getNDConf().indexOfIndex(i)}.
     *  Use this to calculate the true index for an element in the data array (data array index)
     *  based on a provided "virtual index", or "value array index".
     *  This virtual index may be different from the true index depending on the type of nd-array,
     *  like for example if the nd-array is
     *  a slice of another larger nd-array, or if it is in fact a reshaped version of another nd-array.
     *  The basis for performing this translation is expressed by individual implementations of
     *  this {@link NDConfiguration} interface, which contain everything
     *  needed to treat a given block of data as a nd-array!
     *
     * @param index The virtual index of the tensor having this configuration.
     * @return The true index which targets the actual data within the underlying data array of an nd-array / tensor.
     */
    default int indexOfIndex( int index ) { return getNDConf().indexOfIndex( index ); }

    /**
     *  This is a convenience method identical to {@code tensor.getNDConf().IndicesOfIndex(i)}.
     *  Use this to calculates the axis indices for an element in the nd-array array
     *  based on a provided "virtual index".
     *  The resulting index defines the position of the element for every axis.
     *
     * @param index The virtual index of the tensor having this configuration.
     * @return The position of the (virtually) targeted element represented as an array of axis indices.
     */
    default int[] indicesOfIndex( int index ) { return getNDConf().indicesOfIndex( index ); }

    /**
     *  This is a convenience method identical to {@code tensor.getNDConf().indexOfIndices(indices)}.
     *  Use this to calculates the true index for an element in the data array
     *  based on a provided index array.
     *
     * @param indices The indices for every axis of a given nd-array.
     * @return The true index targeting the underlying data array of a given nd-array.
     */
    default int indexOfIndices( int[] indices ) { return getNDConf().indexOfIndices(indices); }


    class Util {

        public static List<Integer> asList( int[] array ) {
            List<Integer> intList = new ArrayList<>( array.length );
            for ( int i : array ) intList.add( i );
            return intList;
        }

    }

}
