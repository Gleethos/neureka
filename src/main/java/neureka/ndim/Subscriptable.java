package neureka.ndim;

import neureka.Tsr;
import neureka.fluent.slicing.SmartSlicer;

import java.math.BigDecimal;
import java.util.Collections;
import java.util.Map;

public interface Subscriptable<V> {

    /**
     *  The following method enables access to specific scalar elements within the tensor.
     *  The method name also translates to the subscript operator in Groovy.
     *
     * @param indices The index array of the element which should be returned.
     * @return An element located at the provided index.
     */
    Tsr<V> getAt( int... indices );

    /**
     *  The following method enables access to specific scalar elements within the tensor.
     *  The method name also translates to the subscript operator in Groovy.
     *
     * @param indices The index array of the element which should be returned.
     * @return An element located at the provided index.
     */
    default Tsr<V> get( int... indices ) { return getAt( indices ); }

    /**
     *  The following method enables the creation of tensor slices which access
     *  the same underlying data (possibly from a different view).
     *  The method name also translates to the subscript operator in Groovy.
     *
     * @param args An arbitrary number of arguments which can be used for slicing.
     * @return A slice tensor created based on the passed keys.
     */
    Tsr<V> getAt( Object... args );

    /**
     *  The following method enables the creation of tensor slices which access
     *  the same underlying data (possibly from a different view).
     *  The method name also translates to the subscript operator in Groovy.
     *
     * @param args An arbitrary number of arguments which can be used for slicing.
     * @return A slice tensor created based on the passed keys.
     */
    default Tsr<V> get( Object... args ) {
        return getAt( args );
    }

    /**
     *  This getter method creates and returns a slice of the original tensor.
     *  The returned slice is a scalar tensor wrapping a single value element which
     *  is being targeted by the provided integer index.
     *
     * @param i The index of the value item which should be returned as a tensor instance.
     * @return A tensor holding a single value element which is internally still residing in the original tensor.
     */
    Tsr<V> getAt( int i );

    /**
     *  This getter method creates and returns a slice of the original tensor.
     *  The returned slice is a scalar tensor wrapping a single value element which
     *  is being targeted by the provided integer index.
     *
     * @param i The index of the value item which should be returned as a tensor instance.
     * @return A tensor holding a single value element which is internally still residing in the original tensor.
     */
    default Tsr<V> get( int i ) { return getAt( i ); }

    /**
     *  This getter method creates and returns a slice of the original tensor.
     *  The returned slice is a scalar tensor wrapping a single value element which
     *  is being targeted by the provided integer index.
     *
     * @param i The index of the value item which should be returned as a tensor instance.
     * @return A tensor holding a single value element which is internally still residing in the original tensor.
     */
    Tsr<V> getAt( Number i );

    /**
     *  This getter method creates and returns a slice of the original tensor.
     *  The returned slice is a scalar tensor wrapping a single value element which
     *  is being targeted by the provided integer index.
     *
     * @param i The index of the value item which should be returned as a tensor instance.
     * @return A tensor holding a single value element which is internally still residing in the original tensor.
     */
    default Tsr<V> get( Number i ) { return getAt( i ); }

    Tsr<V> getAt( Map<?,Integer> rangToStrides );

    /**
     *  This method enables tensor slicing!
     *  It takes a key of various types and configures a slice
     *  tensor which shares the same underlying data as the original tensor.
     *
     * @param key This object might be a wide range of objects including maps, lists or arrays...
     * @return A slice tensor or scalar value.
     */
    Tsr<V> getAt( Object key );

    /**
     *  This method enables tensor slicing!
     *  It takes a key of various types and configures a slice
     *  tensor which shares the same underlying data as the original tensor.
     *
     * @param key This object might be a wide range of objects including maps, lists or arrays...
     * @return A slice tensor or scalar value.
     */
    default Tsr<V> get( Object key ) { return getAt( key ); }

    /**
     *  This method enables assigning a provided tensor to be a subset of this tensor!
     *  It takes a key which is used to configure a slice
     *  sharing the same underlying data as the original tensor.
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
     * @return A slice tensor or scalar value.
     */
    Tsr<V> putAt( Map<?,Integer> key, Tsr<V> value );

}
