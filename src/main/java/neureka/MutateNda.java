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
public interface MutateNda<T> {
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
