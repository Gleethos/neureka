package neureka;

import neureka.common.utility.DataConverter;
import neureka.common.utility.LogUtil;
import neureka.fluent.building.NdaBuilder;
import neureka.fluent.building.states.WithShapeOrScalarOrVector;
import neureka.fluent.slicing.SliceBuilder;
import neureka.fluent.slicing.states.AxisOrGet;
import neureka.framing.Relation;
import neureka.ndim.NDimensional;
import neureka.view.NDPrintSettings;
import neureka.view.NdaAsString;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import java.util.function.Predicate;
import java.util.stream.*;

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
    /**
     *  This is the entry point to the fluent nd-array builder API for building
     *  {@link Nda} instances in a readable and type safe fashion.
     *  The returned {@link WithShapeOrScalarOrVector} is the next step in the
     *  fluent {@link Nda} builder API which will lead to the creation
     *  of an nd-array storing values defined by the provided type class.
     *  A simple usage example would be:
     *   <pre>{@code
     *      Nda.of(Double.class)
     *            .withShape( 2, 3, 4 )
     *            .andFill( 5, 3, 5 )
     *   }</pre>
     *
     *   It is also possible to define a range using the API to populate the tensor with values:
     *   <pre>{@code
     *      Nda.of(Double.class)
     *            .withShape( 2, 3, 4 )
     *            .andFillFrom( 2 ).to( 9 ).step( 2 )
     *   }</pre>
     *
     *   If one needs a simple scalar then the following shortcut is possible:
     *   <pre>{@code
     *      Nda.of(Float.class).scalar( 3f )
     *   }</pre>
     *
     *   This principle works for vectors as well:
     *   <pre>{@code
     *       Nda.of(Byte.class).vector( 2, 5, 6, 7, 8 )
     *   }</pre>
     *   For more fine-grained control over the initialization one can
     *   pass an initialization lambda to the API:
     *   <pre>{@code
     *       Nda.of(Byte.class).withShape(2, 3).andWhere( (i, indices) -> i * 5 - 30 )
     *   }</pre>
     *
     *
     * @param type The type class of the items stored by the tensor built by the exposed builder API.
     * @return The next step of the {@link Nda} builder API which exposes methods for defining shapes.
     */
    static <V> WithShapeOrScalarOrVector<V> of( Class<V> type ) { return new NdaBuilder<>( type ); }

    /**
     * @param value The scalar value which ought to be represented as tensor.
     * @return A scalar double tensor.
     */
    static Nda<Double> of( double value ) { return Tsr.of( Double.class, new int[]{ 1 }, value ); }

    /**
     *  Constructs a vector of floats based on the provided array.
     *
     * @param value The array of floats from which a 1D nd-array ought to be constructed.
     * @return A vector / 1D nd-array of floats.
     */
    static Nda<Float> of( float... value ) { return Tsr.of( Float.class, new int[]{ value.length }, value ); }

    /**
     *  Constructs a vector of doubles based on the provided array.
     *
     * @param value The array of doubles from which a 1D nd-array ought to be constructed.
     * @return A vector / 1D nd-array of doubles.
     */
    static Nda<Double> of( double... value ) { return Tsr.of( Double.class, new int[]{ value.length }, value ); }

    /**
     *  Constructs a vector of bytes based on the provided array.
     *
     * @param value The array of bytes from which a 1D nd-array ought to be constructed.
     * @return A vector / 1D nd-array of bytes.
     */
    static Nda<Byte> of( byte... value ) { return Tsr.of( Byte.class, new int[]{ value.length }, value ); }

    /**
     *  Constructs a vector of ints based on the provided array.
     *
     * @param value The array of ints from which a 1D nd-array ought to be constructed.
     * @return A vector / 1D nd-array of ints.
     */
    static Nda<Integer> of( int... value ) { return Tsr.of( Integer.class, new int[]{ value.length }, value ); }

    /**
     *  Constructs a vector of longs based on the provided array.
     *
     * @param value The array of longs from which a 1D nd-array ought to be constructed.
     * @return A vector / 1D nd-array of longs.
     */
    static Nda<Long> of( long... value ) { return Tsr.of( Long.class, new int[]{ value.length }, value ); }

    /**
     *  Constructs a vector of shorts based on the provided array.
     *
     * @param value The array of shorts from which a 1D nd-array ought to be constructed.
     * @return A vector / 1D nd-array of shorts.
     */
    static Nda<Short> of( short... value ) { return Tsr.of( Short.class, new int[]{ value.length }, value ); }

    /**
     *  Constructs a vector of booleans based on the provided array.
     *
     * @param value The array of booleans from which a 1D nd-array ought to be constructed.
     * @return A vector / 1D nd-array of shorts.
     */
    static Nda<Boolean> of( boolean... value ) { return Tsr.of( Boolean.class, new int[]{ value.length }, value ); }

    /**
     * Constructs a vector of objects based on the provided array.
     *
     * @param values The array of objects from which a 1D nd-array ought to be constructed.
     * @return A vector / 1D nd-array of objects.
     */
    @SafeVarargs
    static <T> Nda<T> of( T... values ) { return Tsr.of(values); }

    /**
     *  If this nd-array is a slice of a parent nd-array then this method will yield true.
     *  Slices can be created by calling the variations of the "{@link Nda#getAt}" method.
     *
     * @return The truth value determining if this nd-array is a slice of another nd-array.
     * @see Nda#getAt(int...)
     * @see Nda#slice()
     */
    boolean isSlice();

    /**
     * If this nd-array is a shallow copy of a parent nd-array then this method will yield true.
     * Shallow copies can be created by calling the "{@link Nda#shallowCopy()}" method.
     * @return The truth value determining if this nd-array is a shallow copy of another nd-array.
     * @see Nda#shallowCopy()
     */
    boolean isShallowCopy();

    /**
     *  If this nd-array is a partial slice of a parent nd-array then this method will yield true.
     *  A partial slice is a slice which does not view all the parents items.
     *  Partial slices can be created by calling the variations of the "{@link Nda#getAt}" method.
     *  This is the inverse of {@link Nda#isFullSlice()}.
     * @return The truth value determining if this nd-array is a partial slice of another nd-array.
     */
    boolean isPartialSlice();

    /**
     *  If this nd-array is a full slice of a parent nd-array then this method will yield true.
     *  A full slice is a slice which views all the parents items.
     *  Full slices can be created by calling the variations of the "{@link Nda#getAt}" method.
     *  This is the inverse of {@link Nda#isPartialSlice()}.
     * @return The truth value determining if this nd-array is a full slice of another nd-array.
     */
    default boolean isFullSlice() { return isSlice() && !isPartialSlice(); }

    /**
     *  This method returns the number of slices which have been
     *  created from this nd-array.
     *  It does so by accessing the {@link Relation} component if present
     *  which internally keeps track of slices via weak references.
     *
     * @return The number of slices derived from this nd-array.
     */
    int sliceCount();

    /**
     *  If slices have been derived from this nd-array then it is a "slice parent".
     *  This is what this method will determine, in which case, it will return true.
     *
     * @return The truth value determining if slices have been derived from this nd-array.
     */
    boolean isSliceParent();

    /**
     * @return The type class of individual value items within this nd-array.
     */
    Class<V> getItemType();

    /**
     * @return The type class of individual value items within this nd-array.
     */
    default Class<V> itemType() { return getItemType(); }

    /*==================================================================================================================
    |
    |       §(6) : ND-ITERATOR LOGIC :
    |   ---------------------------------------
    */

    /**
     *  @return A {@link Stream} of the items in this {@link Nda}.
     */
    default Stream<V> stream() {
        Object rawItems = getRawItems();
        Stream<V> stream;
        if ( rawItems instanceof double[] )
            stream = (Stream<V>) DoubleStream.of( (double[]) rawItems ).boxed();
        else if ( rawItems instanceof int[] )
            stream = (Stream<V>) IntStream.of( (int[]) rawItems ).boxed();
        else if ( rawItems instanceof long[] )
            stream = (Stream<V>) LongStream.of( (long[]) rawItems ).boxed();
        else if ( rawItems instanceof float[] )
            stream = IntStream.range(0,size()).mapToObj(i -> (V) Float.valueOf( ((float[]) rawItems)[i] ) );
        else if ( rawItems instanceof byte[] )
            stream = IntStream.range(0,size()).mapToObj(i -> (V) Byte.valueOf( ((byte[]) rawItems)[i] ) );
        else if ( rawItems instanceof short[] )
            stream = IntStream.range(0,size()).mapToObj(i -> (V) Short.valueOf( ((short[]) rawItems)[i] ) );
        else if ( rawItems instanceof boolean[] )
            stream = IntStream.range(0,size()).mapToObj(i -> (V) Boolean.valueOf( ((boolean[]) rawItems)[i] ) );
        else if ( rawItems instanceof char[] )
            stream = IntStream.range(0,size()).mapToObj(i -> (V) Character.valueOf( ((char[]) rawItems)[i] ) );
        else
            stream = (Stream<V>) Arrays.stream( (Object[]) rawItems );

        boolean executeInParallel = ( this.size() > 1_000 );
        return executeInParallel ? stream.parallel() : stream;
    }

    /**
     * Iterates over every element of this nd-array, and checks whether all
     * elements are <code>true</code> according to the provided lambda.
     * @param predicate The lambda to check each element against.
     * @return true if every item in the nd-array matches the predicate, false otherwise.
     */
    default boolean every( Predicate<V> predicate ) { return stream().allMatch(predicate); }

    /**
     * Iterates over every element of this nd-array, and checks whether any
     * element matches the provided lambda.
     * @param predicate The lambda to check each element against.
     * @return true if any item in the nd-array matches the predicate, false otherwise.
     */
    default boolean any( Predicate<V> predicate ) { return stream().anyMatch(predicate); }

    /**
     * Iterates over every element of this nd-array, and checks whether none
     * of the elements match the provided lambda.
     * @param predicate The lambda to check each element against.
     * @return true if none of the items in the nd-array match the predicate, false otherwise.
     */
    default boolean none( Predicate<V> predicate ) { return stream().noneMatch(predicate);}

    /**
     *  Iterates over every element of this nd-array, and counts the number of
     *  times the provided lambda matches the items of this array.
     * @param predicate The lambda to check each element against.
     * @return The number of items in the nd-array that match the predicate.
     */
    default int count( Predicate<V> predicate ) { return (int) stream().filter(predicate).count(); }

    /**
     * Returns the minimum item of this nd-array according to the provided
     * {@link Comparator}.  This is a special case of a reduction.

     * @param comparator The {@link Comparator} to use to determine the order of the items in the nd-array.
     * @return The minimum value in the nd-array.
     */
    default V minItem( Comparator<V> comparator ) { return stream().min( comparator ).orElse(null); }

    /**
     * Returns the maximum item of this nd-array according to the provided
     * {@link Comparator}.  This is a special case of a reduction.
     *
     * @param comparator The {@link Comparator} to use to determine the order of the items in the nd-array.
     * @return The maximum value in the nd-array.
     */
    default V maxItem( Comparator<V> comparator ) { return stream().max( comparator ).orElse(null); }

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
     *  Use this to access elements of the underlying data array without any index
     *  transformation applied to it. This is usually similar to the {@link #item} method,
     *  however for nd-arrays which are sliced or reshaped views of the data of another nd-array,
     *  this method will always be unbiased access of the raw data...
     *
     * @param i The position of the targeted item within the raw data array of an NDArray implementation.
     * @return The found object sitting at the specified index position.
     */
    V getDataAt( int i );

    /**
     *  A more verbose version of the {@link #items()} method (best used by JVM languages with property support).
     * @return A list of the items in this nd-array.
     */
    default List<V> getItems() { return stream().collect( Collectors.toList() ); }

    /**
     *  A more concise version of the {@link #getItems()} method.
     * @return A list of the items in this nd-array.
     */
    default List<V> items() { return getItems(); }

    Object getRawItems();

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
     *  The following method returns a single item within this nd-array
     *  targeted by the provided integer index.
     *
     * @param i The scalar index of the item which should be returned by the method.
     * @return The item found at the targeted index.
     */
    default V item( int i ) { return getDataAt( indexOfIndex( i ) ); }

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
    default V item( int... indices ) {
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

    /**
     *  Equivalent to the {@code #item(0)}.
     *
     * @return The first item of this nd-array.
     */
    default V item() { return item( 0 ); }

    default <A> A getItemsAs( Class<A> arrayTypeClass ) {
        return DataConverter.get().convert( getRawItems(), arrayTypeClass );
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
     *  myArray.slice()
     *          .axis(0).from(0).to(1)
     *          .axis(1).at(5) // equivalent to '.from(5).to(5)'
     *          .axis().from(0).to(2)
     *          .get();
     * }</pre>
     *
     * @return An instance of the {@link SliceBuilder} class exposing a readable builder API for creating slices.
     */
    AxisOrGet<V> slice();

    /**
     *  The following method enables access to specific scalar elements within the nd-array.
     *  The method name also translates to the subscription operator in Groovy.
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
     *  The method name also translates to the subscription operator in Groovy.
     *
     * @param indices The index array of the element which should be returned.
     * @return An element located at the provided index.
     */
    Nda<V> get( int... indices );

    /**
     *  The following method enables the creation of nd-array slices which access
     *  the same underlying data (possibly from a different view).
     *  The method name also translates to the subscription operator in Groovy.
     *
     * @param args An arbitrary number of arguments which can be used for slicing.
     * @return A slice nd-array created based on the passed keys.
     */
    Nda<V> getAt( Object... args );

    /**
     *  The following method enables the creation of nd-array slices which access
     *  the same underlying data (possibly from a different view).
     *  The method name also translates to the subscription operator in Groovy.
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

    /**
     *  Use this to put a single item at a particular
     *  position within this nd-array.
     *
     * @param indices The indices of the nd-position where the provided item should be placed.
     * @param value The item which should be placed at the position defined by the provided indices.
     * @return This nd-array itself.
     */
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

    /**
     * <p>
     *     This method is a convenience method for mapping the items of this nd-array to another
     *     nd-array of the same type based on a provided lambda expression which will be applies
     *     to all items individually.
     * </p>
     *  Here a simple example:
     *  <pre>{@code
     *  Nda<String> a = Nda.of(String.class).vector("1", "2", "3");
     *  Nda<String> b = a.map( s -> s + "!" );
     *  }</pre>
     *  Note: <br>
     *  The provided lambda cannot be executed anywhere else but the CPU.
     *
     * @param mapper The lambda which maps the items of this nd-array to a new one.
     * @return A new nd-array of type {@code V}.
     */
    Nda<V> map( java.util.function.Function<V,V> mapper );

    /**
     *  This method creates and returns a new nd-array instance
     *  which is not only a copy of the configuration of this nd-array but also a copy of
     *  the underlying data array. <br>
     *  (Note: the underlying nd-array will not be attached to any kind of computation graph)
     *
     * @return A new nd-array instance which is a deep copy of this nd-array.
     */
    Nda<V> deepCopy();

    /**
     *  This creates a copy where the underlying data is still the same. <br>
     *  (Note: the underlying nd-array will not be attached to any kind of computation graph)
     *
     * @return A shallow copy where the underlying data is shared with this nd-array.
     */
    Nda<V> shallowCopy();

    /**
     *  This method exposes the {@link Access} API which allows you to get or set
     *  individual items within this nd-array targeted by an array of provided indices.
     * @param indices An array of indices targeting a particular position in this nd-array...
     * @return An object which allows you to get or set individual items within this nd-array.
     */
    default Access<V> at( int... indices ) {
        return new Access<V>() {
            @Override public V    get()          { return item( indices ); }
            @Override public void set( V value ) { putAt( indices, value ); }
            @Override public boolean equals( Object o ) {
                if ( o == null ) return false;
                if ( o == this ) return true;
                if ( o.getClass() != this.getClass() ) return false;
                Access<V> other = (Access<V>) o;
                return this.get().equals( other.get() );
            }
        };
    }

    /**
     *  Instances of this are being returned by the {@link #at(int...)} method,
     *  and they allow you to get or set individual nd-array items
     * @param <V> The type of the items of this nd-array.
     */
    interface Access<V>
    {
        /**
         *  Get the value at the targeted position.
         * @return The value at the targeted position.
         */
        V get();
        /**
         *  Set the value at the targeted position.
         * @param value The value to be set at the targeted position.
         */
        void set( V value );
    }

    default String toString( NDPrintSettings config ) {
        return NdaAsString.representing( this ).withConfig( config ).toString();
    }

    /**
     *  This allows you to provide a lambda to configure how this nd-array should be
     *  converted to {@link String} instances.
     *  The provided {@link Consumer} will receive a {@link NDPrintSettings} instance
     *  which allows you to change various settings with the help of method chaining.
     *
     * @param config A consumer of the {@link NDPrintSettings} ready to be configured.
     * @return The {@link String} representation of this nd-array.
     */
    default String toString( Consumer<NDPrintSettings> config ) {
        NDPrintSettings defaults = Neureka.get().settings().view().getNDPrintSettings().clone();
        config.accept(defaults);
        return NdaAsString.representing( this ).withConfig( defaults ).toString();
    }

    String toString();


}
