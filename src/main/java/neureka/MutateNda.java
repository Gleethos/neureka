package neureka;

import java.util.List;
import java.util.Map;

/**
 * Nd-arrays should be used as immutable data structures mostly, however sometimes it
 * is important to mutate their state for performance reasons.
 * This interface exposes several methods for mutating the state of this nd-array.
 * The usage of methods exposed by this API is generally discouraged
 * because the exposed state can easily lead to broken nd-arrays and exceptions...<br>
 * <br>
 */
public interface MutateNda<T>
{
    /**
     *  This method enables assigning a provided nd-array to be a subset/slice of this nd-array!
     *  It takes a key which is used to configure a slice
     *  sharing the same underlying data as the original nd-array.
     *  This slice is then used to assign the second argument {@code value} to it.
     *  The usage of this method is especially powerful when used in Groovy. <br>
     *  The following code illustrates this very well:
     *  <pre>{@code
     *      a[[[0..0]:1, [0..0]:1, [0..3]:2]] = b
     *  }</pre>
     *  Here a single argument with the format '[i..j]:k' is equivalent
     *  to pythons 'i:j:k' syntax for indexing! (numpy)                            <br>
     *  i... start indexAlias.                                                      <br>
     *  j... end indexAlias. (inclusive!)                                           <br>
     *  k... step size.                                                             <br>
     *
     * @param key This object is a map defining a stride and a targeted index or range of indices...
     * @param value The nd-array which ought to be assigned into a slice of this nd-array.
     * @return A slice nd-array or scalar value.
     */
    Nda<T> putAt( Map<?,Integer> key, Nda<T> value );

    /**
     *  Use this to put a single item at a particular
     *  position within this nd-array.
     *
     * @param indices The indices of the nd-position where the provided item should be placed.
     * @param value The item which should be placed at the position defined by the provided indices.
     * @return This nd-array itself.
     */
    Nda<T> putAt( int[] indices, T value );


    /**
     *  Use this to place a single item at a particular position within this nd-array!
     *
     * @param indices An array of indices targeting a particular position in this nd-array...
     * @param value the value which ought to be placed at the targeted position.
     * @return This very nd-array in order to enable method chaining...
     */
    Nda<T> set( int[] indices, T value );

    default Nda<T> set( int i0, int i1, T value ) { return putAt( new int[]{i0, i1}, value ); }


    default Nda<T> set( int i0, int i1, int i2, T value ) { return putAt( new int[]{i0, i1, i2}, value ); }

    /**
     *  Individual entries for value items in this nd-array can be set
     *  via this method.
     *
     * @param index The scalar index targeting a specific value position within this nd-array
     *          which ought to be replaced by the one provided by the second parameter
     *          of this method.
     *
     * @param value The item which ought to be placed at the targeted position.
     * @return This very nd-array in order to enable method chaining...
     */
    Nda<T> putAt( int index, T value );

    /**
     *  Individual entries for value items in this nd-array can be set
     *  via this method.
     *
     * @param index The scalar index targeting a specific value position within this nd-array
     *          which ought to be replaced by the one provided by the second parameter
     *          of this method.
     *
     * @param value The item which ought to be placed at the targeted position.
     * @return This very nd-array in order to enable method chaining...
     */
    Nda<T> set( int index, T value );

    /**
     *  This method enables injecting slices of nd-array to be assigned into this nd-array!
     *  It takes a key of various types which is used to configure a slice
     *  nd-array sharing the same underlying data as the original nd-array.
     *  This slice is then used to assign the second argument to it, namely
     *  the "value" argument.
     *
     * @param key This object is a list defining a targeted index or range of indices...
     * @param value the nd-array which ought to be assigned to a slice of this nd-array.
     * @return This very nd-array in order to enable method chaining...
     */
    Nda<T> putAt( List<?> key, Nda<T> value );

    /**
     *  Use this to place a single item at a particular position within this nd-array!
     *
     * @param indices A list of indices targeting a particular position in this nd-array...
     * @param value the value which ought to be placed at the targeted position.
     * @return This very nd-array in order to enable method chaining...
     */
    Nda<T> putAt( List<?> indices, T value );

    /**
     *  An NDArray implementation ought to have some way to selectively modify its underlying value.
     *  This method simply overrides an element within this data array sitting at position "i".
     * @param i The index of the value array entry which ought to be addressed.
     * @param o The object which ought to be placed at the requested position.
     * @return This very nd-array in order to enable method chaining.
     */
    Nda<T> setItemAt( int i, T o );

    /**
     *  This method will receive an object an try to interpret
     *  it or its contents to be set as value for this nd-array.
     *  It will not necessarily replace the underlying data array object of this
     *  nd-array itself, but also try to convert and copy the provided value
     *  into the data array of this nd-array.
     *
     * @param value The value which may be a scalar or array and will be used to populate this nd-array.
     * @return This very nd-array to enable method chaining.
     */
    Nda<T> setItems( Object value );



    /**
     * This method is an inline operation which changes the underlying data of this tensor.
     * It converts the data types of the elements of this tensor to the specified type!<br>
     * <br>
     * <b>WARNING : The usage of this method is discouraged for the following reasons: </b><br>
     * <br>
     * 1. Inline operations are inherently error-prone for most use cases. <br>
     * 2. This inline operation in particular has no safety net,
     * meaning that there is no implementation of version mismatch detection
     * like there is for those operations present in the standard operation backend...
     * No exceptions will be thrown during backpropagation! <br>
     * 3. This method has not yet been implemented to also handle instances which
     * are slices of parent tensors!
     * Therefore, there might be unexpected performance penalties or side effects
     * associated with this method.<br>
     * <br>
     *
     * @param typeClass The target type class for elements of this tensor.
     * @param <V>       The type parameter for the returned tensor.
     * @return The same tensor instance whose data has been converted to hold a different type.
     */
    <V> Nda<V> toType(Class<V> typeClass);

    Data<T> getData();

    <A> A getDataAs(Class<A> arrayTypeClass);

    Nda<T> assign( T other );

    Nda<T> assign( Nda<T> other );

    /**
     * This method receives a nested {@link String} array which
     * ought to contain a label for the index of this tensor.
     * The index for a single element of this tensor would be an array
     * of numbers as long as the rank where every number is
     * in the range of the corresponding shape dimension...
     * Labeling an index means that for every dimension there
     * must be a label for elements in this range array! <br>
     * For example the shape (2,3) could be labeled as follows:    <br>
     * <br>
     * dim 0 : ["A", "B"]                                      <br>
     * dim 1 : ["1", "2", "3"]                                 <br>
     * <br>
     *
     * @param labels A nested String array containing labels for indexes of the tensor dimensions.
     * @return This tensor (method chaining).
     */
    default Nda<T> label(String[]... labels) {
        return label(null, labels);
    }

    /**
     * This method receives a label for this tensor and a
     * nested {@link String} array which ought to contain a
     * label for the index of this tensor.
     * The index for a single element of this tensor would be an array
     * of numbers as long as the rank where every number is
     * in the range of the corresponding shape dimension...
     * Labeling an index means that for every dimension there
     * must be a label for elements in this range array! <br>
     * For example the shape (2,3) could be labeled as follows:    <br>
     * <br>
     * dim 0 : ["A", "B"]                                      <br>
     * dim 1 : ["1", "2", "3"]                                 <br>
     * <br>
     *
     * @param tensorName A label for this tensor itself.
     * @param labels     A nested String array containing labels for indexes of the tensor dimensions.
     * @return This tensor (method chaining).
     */
    Nda<T> label(String tensorName, String[]... labels);

    /**
     * This method receives a nested {@link String} list which
     * ought to contain a label for the index of this tensor.
     * The index for a single element of this tensor would be an array
     * of numbers as long as the rank where every number is
     * in the range of the corresponding shape dimension...
     * Labeling an index means that for every dimension there
     * must be a label for elements in this range array! <br>
     * For example the shape (2,3) could be labeled as follows: <br>
     * <br>
     * dim 0 : ["A", "B"]                                   <br>
     * dim 1 : ["1", "2", "3"]                              <br>
     * <br>
     *
     * @param labels A nested String list containing labels for indexes of the tensor dimensions.
     * @return This tensor (method chaining).
     */
    Nda<T> label(List<List<Object>> labels);

    /**
     * This method receives a label for this tensor and a nested
     * {@link String} list which ought to contain a label for the index of
     * this tensor The index for a single element of this tensor would
     * be an array of numbers as long as the rank where every number is
     * in the range of the corresponding shape dimension...
     * Labeling an index means that for every dimension there
     * must be a label for elements in this range array! <br>
     * For example the shape (2,3) could be labeled as follows: <br>
     * <br>
     * dim 0 : ["A", "B"]                                   <br>
     * dim 1 : ["1", "2", "3"]                              <br>
     * <br>
     *
     * @param tensorName A label for this tensor itself.
     * @param labels     A nested String list containing labels for indexes of the tensor dimensions.
     * @return This tensor (method chaining).
     */
    Nda<T> label(String tensorName, List<List<Object>> labels);

    /**
     * This method provides the ability to
     * label not only the indices of the shape of this tensor, but also
     * the dimension of the shape.
     * The first and only argument of the method expects a map instance
     * where keys are the objects which ought to act as dimension labels
     * and the values are lists of labels for the indices of said dimensions.
     * For example the shape (2,3) could be labeled as follows:            <br>
     * [                                                                   <br>
     * "dim 0" : ["A", "B"],                                           <br>
     * "dim 1" : ["1", "2", "3"]                                       <br>
     * ]                                                                   <br>
     * <br>
     *
     * @param labels A map in which the keys are dimension labels and the values are lists of index labels for the dimension.
     * @return This tensor (method chaining).
     */
    Nda<T> label(Map<Object, List<Object>> labels);

    /**
     * This method provides the ability to
     * label not only the indices of the shape of this tensor, but also
     * the dimension of the shape.
     * The first and only argument of the method expects a map instance
     * where keys are the objects which ought to act as dimension labels
     * and the values are lists of labels for the indices of said dimensions.
     * For example the shape (2,3) could be labeled as follows:            <br>
     * [                                                                   <br>
     * "dim 0" : ["A", "B"],                                            <br>
     * "dim 1" : ["1", "2", "3"]                                        <br>
     * ]                                                                   <br>
     * <br>
     *
     * @param tensorName A label for this tensor itself.
     * @param labels     A map in which the keys are dimension labels and the values are lists of index labels for the dimension.
     * @return This tensor (method chaining).
     */
    Nda<T> label(String tensorName, Map<Object, List<Object>> labels);
}
