package neureka.ndim;

import neureka.Shape;
import neureka.Tensor;
import neureka.ndim.config.NDConfiguration;

import java.util.List;

/**
 *  This interface defines the most essential methods
 *  of the nd-array/tensor API, which describe them
 *  with respect to their dimensionality. <br>
 *  <i>How many dimensions does a tensor/nd-array have?</i> ({@link #rank()}) <br>
 *  <i>How many elements does a tensor/nd-array have?</i> ({@link #size()}) <br>
 *  <i>What are the sizes of the individual dimensions?</i> ({@link #shape(int i)}) <br>
 *  ...
 *
 */
public interface NDimensional
{
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
    default Shape shape() { return Shape.of(getNDConf().shape()); }

    /**
     * @return A list of the dimensions of this tensor / array.
     */
    default Shape getShape() { return shape(); }

    default List<Integer> indicesMap() { return NDUtil.asList(getNDConf().indicesMap()); }

    default List<Integer> strides() { return NDUtil.asList(getNDConf().strides()); }

    default List<Integer> spread() { return NDUtil.asList(getNDConf().spread()); }

    /**
     * The offset is the position of a slice within the n-dimensional
     * data array of its parent array.
     * Use this to get the offsets of all slice dimension.
     *
     * @return The offset position of the slice tensor inside the n-dimensional data array of the parent array.
     */
    default List<Integer> offset() { return NDUtil.asList(getNDConf().offset()); }

    /**
     * @return The {@link NDConfiguration} implementation instance of this {@link Tensor} storing dimensionality information.
     */
    NDConfiguration getNDConf();

    /**
     *  This method receives an axis index and return the
     *  size of the targeted axis / dimension.
     *  It enables readable access to the shape
     *  of this tensor.
     *
     * @param i The index of the shape dimension size which should be returned.
     * @return The dimension size targeted by the provided dimension index.
     */
    default int shape( int i ) { return getNDConf().shape( i ); }

    /**
     * @return The number of elements stored inside the nd-array.
     */
    default int size() { return NDConfiguration.Utility.sizeOfShape(getNDConf().shape()); }

    /**
     * @return The number of elements stored inside the nd-array.
     */
    default int getSize() { return size(); }

    /**
     *  This is a convenience method identical to {@code ndArray.getNDConf().indexOfIndex(i)}.
     *  Use this to calculate the true index for an element in the data array (data array index)
     *  based on a provided "user index", or "user array index".
     *  This user index may be different from the true data array index depending on the type of nd-array,
     *  like for example if the nd-array is
     *  a slice of another larger nd-array, or if it is in fact a permuted version of another nd-array.
     *  The basis for performing this translation is expressed by individual implementations of
     *  this {@link NDConfiguration} interface, which contain everything
     *  needed to treat a given block of data as a nd-array.
     *
     * @param index The virtual index of the tensor having this configuration.
     * @return The true index which targets the actual data within the underlying data array of an nd-array / tensor.
     */
    default int indexOfIndex( int index ) { return getNDConf().indexOfIndex( index ); }

    /**
     *  This is a convenience method identical to {@code ndArray.getNDConf().IndicesOfIndex(i)}.
     *  Use this to calculates the axis indices for an element in the nd-array array
     *  based on a provided "virtual index".
     *  The resulting index defines the position of the element for every axis.
     *
     * @param index The virtual index of the tensor having this configuration.
     * @return The position of the (virtually) targeted element represented as an array of axis indices.
     */
    default int[] indicesOfIndex( int index ) { return getNDConf().indicesOfIndex( index ); }

    /**
     *  This is a convenience method identical to {@code ndArray.getNDConf().indexOfIndices(indices)}.
     *  Use this to calculates the true index for an element in the data array
     *  based on a provided index array.
     *
     * @param indices The indices for every axis of a given nd-array.
     * @return The true index targeting the underlying data array of a given nd-array.
     */
    default int indexOfIndices( int[] indices ) { return getNDConf().indexOfIndices( indices ); }

}
