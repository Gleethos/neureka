package neureka;

import neureka.common.utility.DataConverter;
import neureka.common.utility.LogUtil;
import neureka.ndim.NDimensional;

import java.util.List;
import java.util.Map;
import java.util.function.Predicate;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public interface Nda<V> extends NDimensional, Iterable<V>
{

    boolean isSlice();

    int sliceCount();

    boolean isSliceParent();

    Class<V> getItemClass();

    /*==================================================================================================================
    |
    |       §(6) : ND-ITERATOR LOGIC :
    |   ---------------------------------------
    */

    default Stream<V> stream() {
        boolean executeInParallel = ( this.size() > 1_000 );
        IntStream indices = IntStream.range(0,size());
        return ( executeInParallel ? indices.parallel() : indices ).mapToObj(this::getItemAt);
    }

    default boolean every( Predicate<V> predicate ) {
        return stream().allMatch(predicate);
    }

    default boolean any( Predicate<V> predicate ) {
        return stream().anyMatch(predicate);
    }

    default int count( Predicate<V> predicate ) {
        return (int) stream().filter(predicate).count();
    }


    /**
     *  This returns an unprocessed version of the underlying data of this tensor.
     *  If this tensor is outsourced (stored on a device), then the data will be loaded
     *  into an array and returned by this method.
     *  Do not expect the returned array to be actually stored within the tensor itself!
     *  Contrary to the {@link #getItems()} method, this one will
     *  return the data in an unbiased form, where for example a virtual (see {@link Tsr#isVirtual()})
     *  tensor will have this method return an array of length 1.
     *
     * @return An unbiased copy of the underlying data of this tensor.
     */
    Object getData();

    /**
     *  An NDArray implementation ought to have some way to access its underlying data array.
     *  This method simple returns an element within this data array sitting at position "i".
     * @param i The position of the targeted item within the raw data array of an NDArray implementation.
     * @return The found object sitting at the specified index position.
     */
    V getDataAt( int i );

    Object getItems();

    /**
     *  An NDArray implementation ought to have some way to selectively modify its underlying value.
     *  This method simply overrides an element within this data array sitting at position "i".
     * @param i The index of the value array entry which ought to be addressed.
     * @param o The object which ought to be placed at the requested position.
     * @return This very tensor in order to enable method chaining.
     */
    Nda<V> setItemAt( int i, V o );

    /**
     *  This method will receive an object an try to interpret
     *  it or its contents to be set as value for this tensor.
     *  It will not necessarily replace the underlying data array object of this
     *  tensor itself, but also try to convert and copy the provided value
     *  into the data array of this tensor.
     *
     * @param value The value which may be a scalar or array and will be used to populate this tensor.
     * @return This very tensor to enable method chaining.
     */
    Nda<V> setItems( Object value );

    /**
     *  The following method returns a raw value item within this tensor
     *  targeted by a scalar index.
     *
     * @param i The scalar index of the value item which should be returned by the method.
     * @return The value item found at the targeted index.
     */
    default V getItemAt( int i ) { return getDataAt( indexOfIndex( i ) ); }

    /**
     *  This method returns a raw value item within this tensor
     *  targeted by an index array which is expect to hold an index for
     *  every dimension of the shape of this tensor.
     *  So the provided array must have the same length as the
     *  rank of this tensor!
     *
     * @param indices The index array which targets a single value item within this tensor.
     * @return The found raw value item targeted by the provided index array.
     */
    default V getItemAt( int... indices ) {
        LogUtil.nullArgCheck( indices, "indices", int[].class, "Cannot find tensor value without indices!" );
        if ( indices.length == 0 ) throw new IllegalArgumentException("Index array may not be empty!");
        if ( indices.length < this.rank() ) {
            if ( indices.length == 1 ) return getDataAt( getNDConf().indexOfIndex( indices[0] ) );
            else {
                int[] allIndices = new int[this.rank()]; // We do some 0 padding to make sure we have the correct number of indices.
                System.arraycopy( indices, 0, allIndices, 0, indices.length );
                return getDataAt( getNDConf().indexOfIndices( allIndices ) );
            }
        }
        return getDataAt( getNDConf().indexOfIndices( indices ) );
    }

    default <A> A getItemsAs(Class<A> arrayTypeClass ) {
        return DataConverter.get().convert( getItems(), arrayTypeClass );
    }

    default  <A> A getDataAs( Class<A> arrayTypeClass ) {
        return DataConverter.get().convert( getData(), arrayTypeClass );
    }


    // Slicing:


    /**
     *  The following method enables access to specific scalar elements within the tensor.
     *  The method name also translates to the subscript operator in Groovy.
     *
     * @param indices The index array of the element which should be returned.
     * @return An element located at the provided index.
     */
    Nda<V> getAt( int... indices );

    /**
     *  This getter method creates and returns a slice of the original tensor.
     *  The returned slice is a scalar tensor wrapping a single value element which
     *  is being targeted by the provided integer index.
     *
     * @param i The index of the value item which should be returned as a tensor instance.
     * @return A tensor holding a single value element which is internally still residing in the original tensor.
     */
    Nda<V> getAt( Number i );

    /**
     *  The following method enables access to specific scalar elements within the tensor.
     *  The method name also translates to the subscript operator in Groovy.
     *
     * @param indices The index array of the element which should be returned.
     * @return An element located at the provided index.
     */
    Nda<V> get(int... indices);

    /**
     *  The following method enables the creation of tensor slices which access
     *  the same underlying data (possibly from a different view).
     *  The method name also translates to the subscript operator in Groovy.
     *
     * @param args An arbitrary number of arguments which can be used for slicing.
     * @return A slice tensor created based on the passed keys.
     */
    Nda<V> getAt( Object... args );

    /**
     *  The following method enables the creation of tensor slices which access
     *  the same underlying data (possibly from a different view).
     *  The method name also translates to the subscript operator in Groovy.
     *
     * @param args An arbitrary number of arguments which can be used for slicing.
     * @return A slice tensor created based on the passed keys.
     */
    Nda<V> get( Object... args );

    /**
     *  This getter method creates and returns a slice of the original tensor.
     *  The returned slice is a scalar tensor wrapping a single value element which
     *  is being targeted by the provided integer index.
     *
     * @param i The index of the value item which should be returned as a tensor instance.
     * @return A tensor holding a single value element which is internally still residing in the original tensor.
     */
    Nda<V> getAt( int i );

    /**
     *  This getter method creates and returns a slice of the original tensor.
     *  The returned slice is a scalar tensor wrapping a single value element which
     *  is being targeted by the provided integer index.
     *
     * @param i The index of the value item which should be returned as a tensor instance.
     * @return A tensor holding a single value element which is internally still residing in the original tensor.
     */
    Nda<V> get( int i );

    /**
     *  This getter method creates and returns a slice of the original tensor.
     *  The returned slice is a scalar tensor wrapping a single value element which
     *  is being targeted by the provided integer index.
     *
     * @param i The index of the value item which should be returned as a tensor instance.
     * @return A tensor holding a single value element which is internally still residing in the original tensor.
     */
    Nda<V> get( Number i );

    /**
     *  This method enables tensor slicing!
     *  It takes a key of various types and configures a slice
     *  tensor which shares the same underlying data as the original tensor.
     *
     * @param key This object might be a wide range of objects including maps, lists or arrays...
     * @return A slice tensor or scalar value.
     */
    Nda<V> get( Object key );

    /**
     *  This method is most useful when used in Groovy
     *  where defining maps is done through square brackets,
     *  making it possible to slice tensors like so: <br>
     *  <pre>{@code
     *      var b = a[[[0..0]:1, [0..0]:1, [0..3]:2]]
     *  }</pre>
     *  Here a single argument with the format '[i..j]:k' is equivalent
     *  to Pythons 'i:j:k' syntax for indexing! (numpy)                            <br>
     *  i... start indexAlias.                                                      <br>
     *  j... end indexAlias. (inclusive!)                                           <br>
     *  k... step size.
     *
     * @param rangToStrides A map where the keys define where axes should be sliced and values which define the strides for the specific axis.
     * @return A tensor slice with an offset based on the provided map keys and
     *         strides based on the provided map values.
     */
    Nda<V> getAt( Map<?,Integer> rangToStrides );

    /**
     *  This method enables tensor slicing!
     *  It takes a key of various types and configures a slice
     *  tensor which shares the same underlying data as the original tensor.
     *
     * @param key This object might be a wide range of objects including maps, lists or arrays...
     * @return A slice tensor or scalar value.
     */
    Nda<V> getAt( List<?> key );

    /**
     *  This method enables assigning a provided tensor to be a subset/slice of this tensor!
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
     * @param value The tensor which ought to be assigned into a slice of this tensor.
     * @return A slice tensor or scalar value.
     */
    Nda<V> putAt( Map<?,Integer> key, Nda<V> value );


    Nda<V> putAt( int[] indices, V value );

    /**
     *  Use this to place a single item at a particular position within this tensor!
     *
     * @param indices An array of indices targeting a particular position in this tensor...
     * @param value the value which ought to be placed at the targeted position.
     * @return This very tensor in order to enable method chaining...
     */
    Nda<V> set( int[] indices, V value );

    /**
     *  Individual entries for value items in this tensor can be set
     *  via this method.
     *
     * @param index The scalar index targeting a specific value position within this tensor
     *          which ought to be replaced by the one provided by the second parameter
     *          of this method.
     *
     * @param value The item which ought to be placed at the targeted position.
     * @return This very tensor in order to enable method chaining...
     */
    Nda<V> putAt( int index, V value );

    /**
     *  Individual entries for value items in this tensor can be set
     *  via this method.
     *
     * @param index The scalar index targeting a specific value position within this tensor
     *          which ought to be replaced by the one provided by the second parameter
     *          of this method.
     *
     * @param value The item which ought to be placed at the targeted position.
     * @return This very tensor in order to enable method chaining...
     */
    Nda<V> set( int index, V value );

    /**
     *  This method enables injecting slices of tensor to be assigned into this tensor!
     *  It takes a key of various types which is used to configure a slice
     *  tensor sharing the same underlying data as the original tensor.
     *  This slice is then used to assign the second argument to it, namely
     *  the "value" argument.
     *
     * @param key This object is a list defining a targeted index or range of indices...
     * @param value the tensor which ought to be assigned to a slice of this tensor.
     * @return This very tensor in order to enable method chaining...
     */
    Nda<V> putAt( List<?> key, Nda<V> value );

    /**
     *  Use this to place a single item at a particular position within this tensor!
     *
     * @param indices A list of indices targeting a particular position in this tensor...
     * @param value the value which ought to be placed at the targeted position.
     * @return This very tensor in order to enable method chaining...
     */
    Nda<V> putAt( List<?> indices, V value );

    <T> Nda<T> mapTo(
            Class<T> typeClass,
            java.util.function.Function<V,T> mapper
    );

    default Access<V> at( int... indices ) {
        return new Access<V>() {
            @Override public V    get()          { return getItemAt( indices ); }
            @Override public void set( V value ) { putAt( indices, value ); }

            @Override
            public boolean equals( Object o ) {
                if ( o == null ) return false;
                if ( o == this ) return true;
                if ( o.getClass() != this.getClass() ) return false;
                Access<V> other = (Access<V>) o;
                return this.get().equals( other.get() );
            }
        };
    }

    interface Access<V>
    {
        V get();

        void set( V value );
    }


}
