package neureka;

import neureka.common.utility.DataConverter;
import neureka.common.utility.LogUtil;
import neureka.fluent.building.TensorBuilder;
import neureka.fluent.building.states.WithShapeOrScalarOrVector;
import neureka.fluent.slicing.SliceBuilder;
import neureka.fluent.slicing.states.AxisOrGet;
import neureka.ndim.NDimensional;

import java.util.List;
import java.util.Map;
import java.util.function.Predicate;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 *  {@link Nda}, which is an abbreviation of <b>'N-Dimensional-Array'</b>, represents
 *  a multidimensional, homogeneously filled fixed-size array of items.
 *
 *  {@link Nda}s should be constructed using the fluent builder API exposed by {@link #of(Class)}.
 *
 * @param <V> The type of the items stored in the {@link Nda}.
 */
public interface Nda<V> extends NDimensional, Iterable<V>
{
    static <V> WithShapeOrScalarOrVector<V> of(Class<V> type) { return new TensorBuilder<>( type ); }

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
     *  This returns an unprocessed version of the underlying data of this nd-array.
     *  If this nd-array is outsourced (stored on a device), then the data will be loaded
     *  into an array and returned by this method.
     *  Do not expect the returned array to be actually stored within the nd-array itself!
     *  Contrary to the {@link #getItems()} method, this one will
     *  return the data in an unbiased form, where for example a virtual (see {@link Tsr#isVirtual()})
     *  nd-array will have this method return an array of length 1.
     *
     * @return An unbiased copy of the underlying data of this nd-array.
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
     * @return This very nd-array in order to enable method chaining.
     */
    Nda<V> setItemAt( int i, V o );

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
    Nda<V> setItems( Object value );

    /**
     *  The following method returns a raw value item within this nd-array
     *  targeted by a scalar index.
     *
     * @param i The scalar index of the value item which should be returned by the method.
     * @return The value item found at the targeted index.
     */
    default V getItemAt( int i ) { return getDataAt( indexOfIndex( i ) ); }

    /**
     *  This method returns a raw value item within this nd-array
     *  targeted by an index array which is expected to hold an index for
     *  every dimension of the shape of this nd-array.
     *  So the provided array must have the same length as the
     *  rank of this nd-array!
     *
     * @param indices The index array which targets a single value item within this nd-array.
     * @return The found raw value item targeted by the provided index array.
     */
    default V getItemAt( int... indices ) {
        LogUtil.nullArgCheck( indices, "indices", int[].class, "Cannot find nd-array value without indices!" );
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
     *  This method returns a {@link SliceBuilder} instance exposing a simple builder API
     *  which enables the configuration of a slice of the current nd-array via method chaining.    <br>
     *  The following code snippet slices a 3-dimensional nd-array into a nd-array of shape (2x1x3)  <br>
     * <pre>{@code
     *  myTensor.slice()
     *          .axis(0).from(0).to(1)
     *          .then()
     *          .axis(1).at(5) // equivalent to '.from(5).to(5)'
     *          .then()
     *          .axis().from(0).to(2)
     *          .get();
     * }</pre>
     *
     * @return An instance of the {@link SliceBuilder} class exposing a readable builder API for creating slices.
     */
    AxisOrGet<V> slice();

    /**
     *  The following method enables access to specific scalar elements within the nd-array.
     *  The method name also translates to the subscript operator in Groovy.
     *
     * @param indices The index array of the element which should be returned.
     * @return An element located at the provided index.
     */
    Nda<V> getAt( int... indices );

    /**
     *  This getter method creates and returns a slice of the original nd-array.
     *  The returned slice is a scalar nd-array wrapping a single value element which
     *  is being targeted by the provided integer index.
     *
     * @param i The index of the value item which should be returned as a nd-array instance.
     * @return A nd-array holding a single value element which is internally still residing in the original nd-array.
     */
    Nda<V> getAt( Number i );

    /**
     *  The following method enables access to specific scalar elements within the nd-array.
     *  The method name also translates to the subscript operator in Groovy.
     *
     * @param indices The index array of the element which should be returned.
     * @return An element located at the provided index.
     */
    Nda<V> get(int... indices);

    /**
     *  The following method enables the creation of nd-array slices which access
     *  the same underlying data (possibly from a different view).
     *  The method name also translates to the subscript operator in Groovy.
     *
     * @param args An arbitrary number of arguments which can be used for slicing.
     * @return A slice nd-array created based on the passed keys.
     */
    Nda<V> getAt( Object... args );

    /**
     *  The following method enables the creation of nd-array slices which access
     *  the same underlying data (possibly from a different view).
     *  The method name also translates to the subscript operator in Groovy.
     *
     * @param args An arbitrary number of arguments which can be used for slicing.
     * @return A slice nd-array created based on the passed keys.
     */
    Nda<V> get( Object... args );

    /**
     *  This getter method creates and returns a slice of the original nd-array.
     *  The returned slice is a scalar nd-array wrapping a single value element which
     *  is being targeted by the provided integer index.
     *
     * @param i The index of the value item which should be returned as a nd-array instance.
     * @return A nd-array holding a single value element which is internally still residing in the original nd-array.
     */
    Nda<V> getAt( int i );

    /**
     *  This getter method creates and returns a slice of the original nd-array.
     *  The returned slice is a scalar nd-array wrapping a single value element which
     *  is being targeted by the provided integer index.
     *
     * @param i The index of the value item which should be returned as a nd-array instance.
     * @return A nd-array holding a single value element which is internally still residing in the original nd-array.
     */
    Nda<V> get( int i );

    /**
     *  This getter method creates and returns a slice of the original nd-array.
     *  The returned slice is a scalar nd-array wrapping a single value element which
     *  is being targeted by the provided integer index.
     *
     * @param i The index of the value item which should be returned as a nd-array instance.
     * @return A nd-array holding a single value element which is internally still residing in the original nd-array.
     */
    Nda<V> get( Number i );

    /**
     *  This method enables nd-array slicing!
     *  It takes a key of various types and configures a slice
     *  nd-array which shares the same underlying data as the original nd-array.
     *
     * @param key This object might be a wide range of objects including maps, lists or arrays...
     * @return A slice nd-array or scalar value.
     */
    Nda<V> get( Object key );

    /**
     *  This method is most useful when used in Groovy
     *  where defining maps is done through square brackets,
     *  making it possible to slice nd-arrays like so: <br>
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
     * @return A nd-array slice with an offset based on the provided map keys and
     *         strides based on the provided map values.
     */
    Nda<V> getAt( Map<?,Integer> rangToStrides );

    /**
     *  This method enables nd-array slicing!
     *  It takes a key of various types and configures a slice
     *  nd-array which shares the same underlying data as the original nd-array.
     *
     * @param key This object might be a wide range of objects including maps, lists or arrays...
     * @return A slice nd-array or scalar value.
     */
    Nda<V> getAt( List<?> key );

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
    Nda<V> putAt( Map<?,Integer> key, Nda<V> value );


    Nda<V> putAt( int[] indices, V value );

    /**
     *  Use this to place a single item at a particular position within this nd-array!
     *
     * @param indices An array of indices targeting a particular position in this nd-array...
     * @param value the value which ought to be placed at the targeted position.
     * @return This very nd-array in order to enable method chaining...
     */
    Nda<V> set( int[] indices, V value );

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
    Nda<V> putAt( int index, V value );

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
    Nda<V> set( int index, V value );

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
    Nda<V> putAt( List<?> key, Nda<V> value );

    /**
     *  Use this to place a single item at a particular position within this nd-array!
     *
     * @param indices A list of indices targeting a particular position in this nd-array...
     * @param value the value which ought to be placed at the targeted position.
     * @return This very nd-array in order to enable method chaining...
     */
    Nda<V> putAt( List<?> indices, V value );

    /**
     * <p>
     *     This method is a convenience method for mapping a nd-array to a new type
     *     based on a provided lambda expression.
     *     Here a simple example:
     * </p>
     * <pre>{@code
     *     Nda<String>  a = Nda.of(String.class).vector("1", "2", "3");
     *     Nda<Integer> b = a.mapTo(Integer.class, s -> Integer.parseInt(s));
     * }</pre>
     * <p>
     *     Note: <br>
     *     The provided lambda cannot be executed anywhere else but the CPU.
     *     This is a problem if this nd-array here lives somewhere other than the JVM.
     *     So, therefore, this method will temporally transfer this nd-array from
     *     where ever it may reside back to the JVM!
     * </p>
     * @param typeClass The class of the item type to which the items of this nd-array should be mapped.
     * @param mapper The lambda which maps the items of this nd-array to a new one.
     * @param <T> The type parameter of the items of the returned nd-array.
     * @return A new nd-array of type {@code T}.
     */
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
