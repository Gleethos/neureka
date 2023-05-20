package neureka;

import neureka.common.utility.DataConverter;
import neureka.common.utility.LogUtil;
import neureka.fluent.building.NdaBuilder;
import neureka.fluent.building.states.WithShapeOrScalarOrVector;
import neureka.fluent.slicing.SliceBuilder;
import neureka.fluent.slicing.states.AxisOrGet;
import neureka.framing.NDFrame;
import neureka.framing.Relation;
import neureka.ndim.NDimensional;
import neureka.view.NDPrintSettings;
import neureka.view.NdaAsString;

import java.math.BigDecimal;
import java.util.*;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.function.Supplier;
import java.util.stream.*;

/**
 *  {@link Nda}, which is an abbreviation of <b>'N-Dimensional-Array'</b>, represents
 *  a multidimensional, homogeneously filled fixed-size array of items.
 *  <p>
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
     *   It is also possible to define a range using the API to populate the nd-array with values:
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
     * @param type The type class of the items stored by the nd-array built by the exposed builder API.
     * @return The next step of the {@link Nda} builder API which exposes methods for defining shapes.
     */
    static <V> WithShapeOrScalarOrVector<V> of( Class<V> type ) { return new NdaBuilder<>( type ); }

    /**
     * This is a shortcut method for {@code Nda.of(String.class)}
     * used to build {@link Nda}s storing {@link String}s.
     * @return The next step of the {@link Nda} builder API which exposes methods for defining shapes.
     */
    static WithShapeOrScalarOrVector<String> ofStrings() { return of( String.class ); }

    /**
     * This is a shortcut method for {@code Nda.of(Integer.class)}
     * used to build {@link Nda}s storing {@link Integer}s.
     * @return The next step of the {@link Nda} builder API which exposes methods for defining shapes.
     */
    static WithShapeOrScalarOrVector<Integer> ofInts() { return of( Integer.class ); }

    /**
     * This is a shortcut method for {@code Nda.of(Double.class)}
     * used to build {@link Nda}s storing {@link Double}s.
     * @return The next step of the {@link Nda} builder API which exposes methods for defining shapes.
     */
    static WithShapeOrScalarOrVector<Double> ofDoubles() { return of( Double.class ); }

    /**
     * This is a shortcut method for {@code Nda.of(Float.class)}
     * used to build {@link Nda}s storing {@link Float}s.
     * @return The next step of the {@link Nda} builder API which exposes methods for defining shapes.
     */
    static WithShapeOrScalarOrVector<Float> ofFloats() { return of( Float.class ); }

    /**
     * This is a shortcut method for {@code Nda.of(Long.class)}
     * used to build {@link Nda}s storing {@link Long}s.
     * @return The next step of the {@link Nda} builder API which exposes methods for defining shapes.
     */
    static WithShapeOrScalarOrVector<Long> ofLongs() { return of( Long.class ); }

    /**
     * This is a shortcut method for {@code Nda.of(Boolean.class)}
     * used to build {@link Nda}s storing {@link Boolean}s.
     * @return The next step of the {@link Nda} builder API which exposes methods for defining shapes.
     */
    static WithShapeOrScalarOrVector<Boolean> ofBooleans() { return of( Boolean.class ); }

    /**
     * This is a shortcut method for {@code Nda.of(Character.class)}
     * used to build {@link Nda}s storing {@link Character}s.
     * @return The next step of the {@link Nda} builder API which exposes methods for defining shapes.
     */
    static WithShapeOrScalarOrVector<Character> ofChars() { return of( Character.class ); }

    /**
     * This is a shortcut method for {@code Nda.of(Byte.class)}
     * used to build {@link Nda}s storing {@link Byte}s.
     * @return The next step of the {@link Nda} builder API which exposes methods for defining shapes.
     */
    static WithShapeOrScalarOrVector<Byte> ofBytes() { return of( Byte.class ); }

    /**
     * This is a <b>short</b>cut method for {@code Nda.of(Short.class)}
     * used to build {@link Nda}s storing {@link Short}s.
     * @return The next step of the {@link Nda} builder API which exposes methods for defining shapes.
     */
    static WithShapeOrScalarOrVector<Short> ofShorts() { return of( Short.class ); }

    /**
     * This is a shortcut method for {@code Nda.of(Object.class)}
     * used to build {@link Nda}s storing {@link Object}s.
     * @return The next step of the {@link Nda} builder API which exposes methods for defining shapes.
     */
    static WithShapeOrScalarOrVector<Object> ofObjects() { return of( Object.class ); }

    /**
     * This is a shortcut method for {@code Nda.of(Number.class)}
     * used to build {@link Nda}s storing {@link Number}s.
     * @return The next step of the {@link Nda} builder API which exposes methods for defining shapes.
     */
    static WithShapeOrScalarOrVector<Number> ofNumbers() { return of( Number.class ); }

    /**
     * This is a shortcut method for {@code Nda.of(BigDecimal.class)}
     * used to build {@link Nda}s storing {@link BigDecimal}s.
     * @return The next step of the {@link Nda} builder API which exposes methods for defining shapes.
     */
    static WithShapeOrScalarOrVector<BigDecimal> ofBigDecimals() { return of( BigDecimal.class ); }

    /**
     * @param value The scalar value which ought to be represented as nd-array.
     * @return A scalar double nd-array.
     */
    static Nda<Double> of( double value ) { return Tensor.of( Double.class, Shape.of( 1 ), value ); }

    /**
     *  Constructs a vector of floats based on the provided array.
     *
     * @param value The array of floats from which a 1D nd-array ought to be constructed.
     * @return A vector / 1D nd-array of floats.
     */
    static Nda<Float> of( float... value ) { return Tensor.of( Float.class, Shape.of( value.length ), value ); }

    /**
     *  Constructs a vector of doubles based on the provided array.
     *
     * @param value The array of doubles from which a 1D nd-array ought to be constructed.
     * @return A vector / 1D nd-array of doubles.
     */
    static Nda<Double> of( double... value ) { return Tensor.of( Double.class, Shape.of( value.length ), value ); }

    /**
     *  Constructs a vector of bytes based on the provided array.
     *
     * @param value The array of bytes from which a 1D nd-array ought to be constructed.
     * @return A vector / 1D nd-array of bytes.
     */
    static Nda<Byte> of( byte... value ) { return Tensor.of( Byte.class, Shape.of( value.length ), value ); }

    /**
     *  Constructs a vector of ints based on the provided array.
     *
     * @param value The array of ints from which a 1D nd-array ought to be constructed.
     * @return A vector / 1D nd-array of ints.
     */
    static Nda<Integer> of( int... value ) { return Tensor.of( Integer.class, Shape.of( value.length ), value ); }

    /**
     *  Constructs a vector of longs based on the provided array.
     *
     * @param value The array of longs from which a 1D nd-array ought to be constructed.
     * @return A vector / 1D nd-array of longs.
     */
    static Nda<Long> of( long... value ) { return Tensor.of( Long.class, Shape.of( value.length ), value ); }

    /**
     *  Constructs a vector of shorts based on the provided array.
     *
     * @param value The array of shorts from which a 1D nd-array ought to be constructed.
     * @return A vector / 1D nd-array of shorts.
     */
    static Nda<Short> of( short... value ) { return Tensor.of( Short.class, Shape.of( value.length ), value ); }

    /**
     *  Constructs a vector of booleans based on the provided array.
     *
     * @param value The array of booleans from which a 1D nd-array ought to be constructed.
     * @return A vector / 1D nd-array of shorts.
     */
    static Nda<Boolean> of( boolean... value ) { return Tensor.of( Boolean.class, Shape.of( value.length ), value ); }

    /**
     * Constructs a vector of objects based on the provided array.
     *
     * @param values The array of objects from which a 1D nd-array ought to be constructed.
     * @return A vector / 1D nd-array of objects.
     */
    @SafeVarargs
    static <T> Nda<T> of( T... values ) { return Tensor.of(values); }

    /**
     *  Use this to construct and return a double based nd-array of the specified shape and initial values.
     *  The length of the provided array does not have to match the number of elements
     *  defined by the provided shape, the nd-array will be populated based on repeated iteration over the
     *  provided double array.
     *
     * @param shape The shape of the resulting nd-array consisting of any number of axis-sizes.
     * @param values The values which ought to be used to populate the tensor.
     */
    static Nda<Double> of( Shape shape, double... values ) { return Tensor.ofAny( Double.class, shape, values ); }

    /**
     *  Use this to construct and return a float based nd-array of the specified shape and initial values.
     *  The length of the provided array does not have to match the number of elements
     *  defined by the provided shape, the nd-array will be populated based on repeated iteration over the
     *  provided float array.
     *
     * @param shape The shape of the resulting nd-array consisting of any number of axis-sizes.
     * @param values The values which ought to be used to populate the tensor.
     */
    static Nda<Float> of( Shape shape, float... values ) { return Tensor.ofAny( Float.class, shape, values ); }

    /**
     *  Use this to construct and return a byte based nd-array of the specified shape and initial values.
     *  The length of the provided array does not have to match the number of elements
     *  defined by the provided shape, the nd-array will be populated based on repeated iteration over the
     *  provided byte array.
     *
     * @param shape The shape of the resulting nd-array consisting of any number of axis-sizes.
     * @param values The values which ought to be used to populate the tensor.
     */
    static Nda<Byte> of( Shape shape, byte... values ) { return Tensor.ofAny( Byte.class, shape, values ); }

    /**
     *  Use this to construct and return a int based nd-array of the specified shape and initial values.
     *  The length of the provided array does not have to match the number of elements
     *  defined by the provided shape, the nd-array will be populated based on repeated iteration over the
     *  provided int array.
     *
     * @param shape The shape of the resulting nd-array consisting of any number of axis-sizes.
     * @param values The values which ought to be used to populate the tensor.
     */
    static Nda<Integer> of( Shape shape, int... values ) { return Tensor.ofAny( Integer.class, shape, values ); }

    /**
     *  Use this to construct and return a long based nd-array of the specified shape and initial values.
     *  The length of the provided array does not have to match the number of elements
     *  defined by the provided shape, the nd-array will be populated based on repeated iteration over the
     *  provided long array.
     *
     * @param shape The shape of the resulting nd-array consisting of any number of axis-sizes.
     * @param values The values which ought to be used to populate the tensor.
     */
    static Nda<Long> of( Shape shape, long... values ) { return Tensor.ofAny( Long.class, shape, values ); }

    /**
     *  Use this to construct and return a short based nd-array of the specified shape and initial values.
     *  The length of the provided array does not have to match the number of elements
     *  defined by the provided shape, the nd-array will be populated based on repeated iteration over the
     *  provided short array.
     *
     * @param shape The shape of the resulting nd-array consisting of any number of axis-sizes.
     * @param values The values which ought to be used to populate the tensor.
     */
    static Nda<Short> of( Shape shape, short... values ) { return Tensor.ofAny( Short.class, shape, values ); }

    /**
     *  Use this to construct and return a boolean based nd-array of the specified shape and initial values.
     *  The length of the provided array does not have to match the number of elements
     *  defined by the provided shape, the nd-array will be populated based on repeated iteration over the
     *  provided boolean array.
     *
     * @param shape The shape of the resulting nd-array consisting of any number of axis-sizes.
     * @param values The values which ought to be used to populate the tensor.
     */
    static Nda<Boolean> of( Shape shape, boolean... values ) { return Tensor.ofAny( Boolean.class, shape, values ); }

    /**
     *  Use this to construct and return an object based nd-array of the specified shape and initial values.
     *  The length of the provided array does not have to match the number of elements
     *  defined by the provided shape, the nd-array will be populated based on repeated iteration over the
     *  provided object array.
     *
     * @param shape The shape of the resulting nd-array consisting of any number of axis-sizes.
     * @param values The values which ought to be used to populate the tensor.
     */
    @SafeVarargs
    static <T> Nda<T> of( Shape shape, T... values ) { return (Nda<T>) Tensor.of( values ).reshape( shape.toIntArray() ); }

    /**
     * Constructs a vector of objects based on the provided iterable.
     *
     * @param values The iterable of objects from which a 1D nd-array ought to be constructed.
     * @return A vector / 1D nd-array of objects.
     */
    static <T> Nda<T> of( Iterable<T> values ) { return Tensor.of(values); }

    /**
     * Constructs a vector of objects based on the provided list.
     *
     * @param values The list of objects from which a 1D nd-array ought to be constructed.
     * @return A vector / 1D nd-array of objects.
     */
    static <T> Nda<T> of( List<T> values ) { return TensorImpl._of(values); }

    /**
     *  A nd-array can have a label. This label is used for example when printing the nd-array.
     *  When loading a CSV file for example the label of the nd-array
     *  will be taken from the cell where the header row and the first column intersect.
     *  @return The label/name of the nd-array.
     */
    default String getLabel() { return ((TensorImpl<?>) this).find(NDFrame.class).map(NDFrame::getLabel).orElse(""); }

    /**
     *  A nd-array can have a label. This label is used for example when printing the nd-array.
     *  When loading a CSV file for example the label of the nd-array
     *  will be taken from the cell where the header row and the first column intersect.
     *  This is a shorter version of {@link #getLabel()}.
     *  @return The label/name of the nd-array.
     */
    default String label() { return this.getLabel(); }

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

    /**
     * @return A new nd-array which is a shallow copy of this nd-array but with a different label.
     */
    Nda<V> withLabel( String label );

    /**
     *  This method receives a nested {@link String} array which
     *  ought to contain a label for the index of this nd-array.
     *  The index for a single element of this nd-array would be an array
     *  of numbers as long as the rank where every number is
     *  in the range of the corresponding shape dimension...
     *  Labeling an index means that for every dimension there
     *  must be a label for elements in this range array! <br>
     *  For example the shape (2,3) could be labeled as follows:    <br>
     *                                                              <br>
     *      dim 0 : ["A", "B"]                                      <br>
     *      dim 1 : ["1", "2", "3"]                                 <br>
     *                                                              <br>
     *
     * @param labels A nested String array containing labels for indexes of the nd-array dimensions.
     * @return This nd-array (method chaining).
     */
    Nda<V> withLabels( String[]... labels );

    /**
     *  This method receives a nested {@link String} list which
     *  ought to contain a label for the index of this nd-array.
     *  The index for a single element of this nd-array would be an array
     *  of numbers as long as the rank where every number is
     *  in the range of the corresponding shape dimension...
     *  Labeling an index means that for every dimension there
     *  must be a label for elements in this range array! <br>
     *  For example the shape (2,3) could be labeled as follows: <br>
     *                                                           <br>
     *      dim 0 : ["A", "B"]                                   <br>
     *      dim 1 : ["1", "2", "3"]                              <br>
     *                                                           <br>
     * @param labels A nested String list containing labels for indexes of the nd-array dimensions.
     * @return This nd-array (method chaining).
     */
    Nda<V> withLabels( List<List<Object>> labels );

    /**
     *  This method provides the ability to
     *  label not only the indices of the shape of this nd-array, but also
     *  the dimension of the shape.
     *  The first and only argument of the method expects a map instance
     *  where keys are the objects which ought to act as dimension labels
     *  and the values are lists of labels for the indices of said dimensions.
     *  For example the shape (2,3) could be labeled as follows:            <br>
     *  [                                                                   <br>
     *      "dim 0" : ["A", "B"],                                           <br>
     *      "dim 1" : ["1", "2", "3"]                                       <br>
     *  ]                                                                   <br>
     *                                                                      <br>
     * @param labels A map in which the keys are dimension labels and the values are lists of index labels for the dimension.
     * @return This nd-array (method chaining).
     */
    Nda<V> withLabels( Map<Object, List<Object>> labels );


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
     *  A convenience method for {@code stream().filter( predicate )}.
     *
     * @param predicate The predicate to filter the items of this {@link Nda}.
     * @return A {@link Stream} of the items in this {@link Nda} which match the predicate.
     */
    default Stream<V> filter( Predicate<V> predicate ) { return stream().filter( predicate ); }

    /**
     *  A convenience method for {@code nda.stream().flatMap( mapper )},
     *  which turns this {@link Nda} into a {@link Stream} of its items. <br>
     *  Here an example of how to use this method : <br>
     *  <pre>{@code
     *    var nda = Nda.of( -2, -1, 0, 1, 2 );
     *    var list = nda.flatMap( i -> Stream.of( i * 2, i * 3 ) ).toList();
     *    // list = [-4, -6, -2, -3, 0, 0, 2, 3, 4, 6, 6, 9]
     *  }</pre>
     *
     * @param mapper The mapper to map the items of this {@link Nda}.
     * @return A {@link Stream} of the items in this {@link Nda} which match the predicate.
     */
    default <R> Stream<R> flatMap( Function<V, Stream<R>> mapper ) {
        return stream().flatMap( v -> {
            Object o = mapper.apply( v );
            if ( o instanceof Iterable )
                return (Stream<R>) StreamSupport.stream( ((Iterable) o).spliterator(), false );
            else if ( o instanceof Stream )
                return (Stream<R>) o;
            else
                return Stream.of( (R) o );
        });
    }

    /**
     * Returns a {@code Collector} that accumulates the input elements into a
     * new {@link Nda} with the specified shape. <br>
     * Usage example : <br>
     * <pre>{@code
     *    var nda = Stream.of( 1, 2, 3, 4, 5, 6 )
     *                      .collect( Nda.shaped( 2, 3 ) );
     * }</pre>
     *
     * @param shape The shape of the nd-array to be returned.
     * @param <T> the type of the input elements
     * @return a {@code Collector} which collects all the input elements into a
     *          {@link Nda}, in encounter order.
     */
    static <T> Collector<T, ?, Nda<T>> shaped( int... shape ) {
        return Collector.of(
                (Supplier<List<T>>) ArrayList::new,
                List::add,
                (left, right) -> { left.addAll(right); return left; },
                list -> Tensor.of( Shape.of(shape), list )
            );
    }

    /**
     * Iterates over every element of this nd-array, and checks whether all
     * elements are <code>true</code> according to the provided lambda.
     * @param predicate The lambda to check each element against.
     * @return true if every item in the nd-array matches the predicate, false otherwise.
     */
    default boolean every( Predicate<V> predicate ) {
        if ( ((Tensor<V>)this).isVirtual() ) return predicate.test( this.item() );
        return stream().allMatch(predicate);
    }

    /**
     * Iterates over every element of this nd-array, and checks whether any
     * element matches the provided lambda.
     * @param predicate The lambda to check each element against.
     * @return true if any item in the nd-array matches the predicate, false otherwise.
     */
    default boolean any( Predicate<V> predicate ) {
        if ( ((Tensor<V>)this).isVirtual() ) return predicate.test( this.item() );
        return stream().anyMatch(predicate);
    }

    /**
     * Iterates over every element of this nd-array, and checks whether none
     * of the elements match the provided lambda.
     * @param predicate The lambda to check each element against.
     * @return true if none of the items in the nd-array match the predicate, false otherwise.
     */
    default boolean none( Predicate<V> predicate ) {
        if ( ((Tensor<V>)this).isVirtual() ) return !predicate.test( this.item() );
        return stream().noneMatch(predicate);
    }

    /**
     *  Iterates over every element of this nd-array, and counts the number of
     *  times the provided lambda matches the items of this array.
     *  <p>
     *  Here is an example of counting the number of items in the array that are
     *  greater than 5 :
     *  <pre>{@code
     *    var nda = Nda.of( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 );
     *    var count = nda.count( i -> i > 5 );
     *    System.out.println( count ); // prints 5
     *  }</pre>
     *
     *
     * @param predicate The lambda to check each element against.
     * @return The number of items in the nd-array that match the predicate.
     */
    default int count( Predicate<V> predicate ) {
        if ( ((Tensor<V>)this).isVirtual() ) return predicate.test( this.item() ) ? this.size() : 0;
        return (int) stream().filter(predicate).count();
    }

    /**
     * Returns the minimum item of this nd-array according to the provided
     * {@link Comparator}.  This is a special case of a reduction.
     *
     * @param comparator The {@link Comparator} to use to determine the order of the items in the nd-array.
     * @return The minimum value in the nd-array.
     */
    default V minItem( Comparator<V> comparator ) {
        if ( ((Tensor<V>)this).isVirtual() ) return this.item();
        return stream().min( comparator ).orElse(null);
    }

    /**
     * Returns the maximum item of this nd-array according to the provided
     * {@link Comparator}.  This is a special case of a reduction.
     *
     * @param comparator The {@link Comparator} to use to determine the order of the items in the nd-array.
     * @return The maximum value in the nd-array.
     */
    default V maxItem( Comparator<V> comparator ) {
        if ( ((Tensor<V>)this).isVirtual() ) return this.item();
        return stream().max( comparator ).orElse(null);
    }

    /**
     *  This returns an unprocessed version of the underlying data of this nd-array.
     *  If this nd-array is outsourced (stored on a device), then the data will be loaded
     *  into an array and returned by this method.
     *  Do not expect the returned array to be actually stored within the nd-array itself!
     *  Contrary to the {@link #getItems()} method, this one will
     *  return the data in an unbiased form, where for example a virtual (see {@link Tensor#isVirtual()})
     *  nd-array will have this method return an array of length 1.
     *
     * @return An unbiased copy of the underlying data of this nd-array.
     */
    Object getRawData();

    /**
     *  Use this to access elements of the underlying data array without any index
     *  transformation applied to it. This is usually similar to the {@link #item} method,
     *  however for nd-arrays which are sliced or permuted views of the data of another nd-array,
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

    /**
     * @return The items of this nd-array as a (if possible) primitive array.
     */
    Object getRawItems();

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
     *  Equivalent to the {@code #item(0)} and {@link #getItem()}.
     *
     * @return The first item of this nd-array.
     */
    default V item() { return item( 0 ); }

    /**
     *  Equivalent to the {@code #item(0)} and {@link #item()}.
     *
     * @return The first item of this nd-array.
     */
    default V getItem() { return item(); }

    /**
     * Use this to get the items of this nd-array as a primitive array
     * of the specified type.
     * @param arrayTypeClass The class of the array type to return.
     * @param <A> The type of the array to return.
     * @return The items of this nd-array as a primitive array of the specified type.
     */
    default <A> A getItemsAs( Class<A> arrayTypeClass ) {
        return DataConverter.get().convert( getRawItems(), arrayTypeClass );
    }

    /**
     * Use this to get the items of the underlying {@link Data} buffer
     * of this nd-array as a primitive array
     * of the specified type.
     * Note that the length of the returned array may be different from the
     * size of this nd-array. This is the case if this nd-array is a slice
     * of another larger nd-array.
     * @param arrayTypeClass The class of the array type to return.
     * @param <A> The type of the array to return.
     * @return The items of this nd-array as a primitive array of the specified type.
     */
    default  <A> A getDataAs( Class<A> arrayTypeClass ) {
        return DataConverter.get().convert( getRawData(), arrayTypeClass );
    }

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
     * This method concatenates the provided nd-arrays together with this nd-array along a specified axis.
     * The provided nd-arrays must have the same shape and data type as the current nd-array, except for the specified axis.
     *
     * @param axis The axis along which the provided nd-arrays should be concatenated.
     *             The axis must be within the range of the rank of the current nd-array.
     * @param other The other nd-arrays which should be concatenated with this nd-array.
     * @param ndArrays The non-null, non-empty nd-arrays which should be concatenated together with this and the other nd-array.
     *                 The nd-arrays all must have the same shape as this nd-array, except for the specified axis.
     *                 Also, it must have the same data type as the current nd-array.
     * @return A new nd-array which is the concatenation of the current nd-array and the provided nd-arrays.
     */
    Nda<V> concatAt( int axis, Nda<V> other, Nda<V>... ndArrays );

    /**
     * This method concatenates the provided nd-array together with this nd-array along a specified axis.
     * The provided nd-array must have the same shape and data type as this nd-array, except for the specified axis.
     *
     * @param axis The axis along which the provided nd-arrays should be concatenated.
     *             The axis must be within the range of the rank of the current nd-array.
     * @param other The other nd-arrays which should be concatenated with this nd-array.
     * @return A new nd-array which is the concatenation of the current nd-array and the provided nd-arrays.
     */
    Nda<V> concatAt( int axis, Nda<V> other );

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
     * @param rangToSteps A map where the keys define where axes should be sliced and values which define the steps for the specific axis.
     * @return A nd-array slice with an offset based on the provided map keys and
     *         steps based on the provided map values.
     */
    Nda<V> getAt( Map<?,Integer> rangToSteps );

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
     * <p>
     *     This is a convenience method for mapping a nd-array to a nd-array of new type
     *     based on a provided target item type and mapping lambda.
     *     Here a simple example:
     * </p>
     * <pre>{@code
     *     Nda<String>  a = Nda.of(String.class).vector("1", "2", "3");
     *     Nda<Integer> b = a.mapTo(Integer.class, s -> Integer.parseInt(s));
     * }</pre>
     * <p>
     *     Note: <br>
     *     The provided lambda cannot be executed anywhere else but the CPU.
     *     This is a problem if this nd-array lives somewhere other than the JVM.
     *     So, therefore, this method will temporally transfer this nd-array from
     *     where ever it may reside back to the JVM, execute the mapping lambda, and
     *     then transfer the result back to the original location.
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
     *     nd-array of the same type based on the provided lambda function, which will be applied
     *     to all items of this nd-array individually (element-wise).
     * </p>
     *  Here a simple example:
     *  <pre>{@code
     *  Nda<String> a = Nda.of(String.class).vector("1", "2", "3");
     *  Nda<String> b = a.map( s -> s + "!" );
     *  }</pre>
     *  Note: <br>
     *  The provided lambda cannot be executed anywhere else but the CPU.
     *  This is a problem if this nd-array lives somewhere other than the JVM.
     *  So, therefore, this method will temporally transfer this nd-array from where ever it may reside
     *  back to the JVM, execute the mapping lambda, and then transfer the result back to the original location.
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
     *  This method exposes an API for mutating the state of this tensor.
     *  The usage of methods exposed by this API is generally discouraged
     *  because the exposed state can easily lead to broken tensors and exceptional situations!<br>
     *  <br>
     *  <p><b>
     *  Only use this if you know what you are doing and
     *  performance is critical! <br>
     *  </b>
     *  (Like in custom backend extensions for example)
     *
     * @return The unsafe API exposes methods for mutating the state of the tensor.
     */
    MutateNda<V> getMut();

    /**
     *  This method exposes an API for mutating the state of this tensor.
     *  The usage of methods exposed by this API is generally discouraged
     *  because the exposed state can easily lead to broken tensors and exceptional situations!<br>
     *  <br>
     *  <p><b>
     *  Only use this if you know what you are doing and
     *  performance is critical! <br>
     *  </b>
     *  (Like custom backend extensions for example)
     *
     * @return The unsafe API exposes methods for mutating the state of the tensor.
     */
    default MutateNda<V> mut() { return getMut(); }

    /**
     *  Returns a nd-array with the same data and number of elements as this nd-array, but with the specified shape.
     *  When possible, the returned nd-array will be a view of this nd-array.
     *
     *  A single dimension may be -1, in which case it’s inferred from the remaining
     *  dimensions and the number of elements in input.
     *
     *  Keep in mind that the new shape must have the same number of elements as the original shape. <br>
     *  <br>
     *  This operation supports autograd.
     *
     * @param shape The new shape of the returned nd-array.
     * @return A new nd-array instance with the same underlying data (~shallow copy) but with a different shape.
     */
    Nda<V> reshape( int... shape );

    /**
     *  Returns a view of the original tensor input with its dimensions permuted.<br>
     *  Consider a 3-dimensional tensor x with shape (2×3×5),
     *  then calling x.permute(1, 0, 2) will return a 3-dimensional tensor of shape (3×2×5). <br>
     *
     * @param dims The desired ordering of dimensions
     * @return A new nd-array instance with the same underlying data (~shallow copy) but with a different shape.
     */
    Nda<V> permute( int... dims );

    /**
     * Returns a view of the original tensor input the targeted
     * axes are swapped / transposed.<br>
     *
     * @param dim1 The first dimension to be swapped.
     * @param dim2 The second dimension to be swapped.
     * @return A new nd-array instance with the same underlying data (~shallow copy) but with a different shape.
     */
    Nda<V> transpose( int dim1, int dim2 );

    /**
     *  This method exposes the {@link Item} API which allows you to get or set
     *  individual items within this nd-array targeted by an array of provided indices.
     * @param indices An array of indices targeting a particular position in this nd-array...
     * @return An object which allows you to get or set individual items within this nd-array.
     */
    Item<V> at( int... indices );

    /**
     *  Instances of this are being returned by the {@link #at(int...)} method,
     *  and they allow you to get individual nd-array items
     * @param <V> The type of the items of this nd-array.
     */
    interface Item<V>
    {
        /**
         *  Get the value at the targeted position or throw an exception if the item does not exist.
         *
         * @return The value at the targeted position.
         */
        default V get() {
            V item = orElseNull();
            if ( item == null )
                throw new IllegalArgumentException("No item at the targeted position!");
            return item;
        }

        /**
         *  Get the value at the targeted position or return the provided default value if the item does not exist.
         *
         * @param defaultValue The default value to return if the item does not exist.
         * @return The value at the targeted position or the provided default value.
         * @throws IllegalArgumentException If the provided default value is {@code null}.
         */
        default V orElse( V defaultValue ) {
            if ( defaultValue == null )
                throw new IllegalArgumentException("The provided default value must not be null! (Use orElseNull() instead)");
            V item = orElseNull();
            return item == null ? defaultValue : item;
        }

        /**
         *  Get the value at the targeted position or return {@code null} if the item does not exist.
         *
         * @return The value at the targeted position or {@code null}.
         */
        V orElseNull();

        /**
         *  Converts this item into an optional value.
         *  If the item exists, the resulting optional will contain the value.
         *  Otherwise, the resulting optional will be empty.
         * @return An optional value.
         */
        default Optional<V> toOptional() {
            V item = orElseNull();
            return item == null ? Optional.empty() : Optional.of( item );
        }

        /**
         *  Maps this item to an optional value based on the provided lambda.
         *  The lambda will be executed if the item exists.
         *  If the lambda returns {@code null} the resulting optional will be empty.
         *  Otherwise, the resulting optional will contain the value returned by the lambda.
         *
         *  @param mapper The lambda which maps the item to an optional value.
         *  @return An optional value based on the provided lambda.
         */
        default Optional<V> map( Function<V,V> mapper ) {
            V item = orElseNull();
            return item == null ? Optional.empty() : Optional.ofNullable( mapper.apply( item ) );
        }

        /**
         * @return {@code true} if the item exists, {@code false} otherwise.
         */
        default boolean exists() {
            return orElseNull() != null;
        }

        /**
         * @return {@code true} if the item does not exist, {@code false} otherwise.
         */
        default boolean doesNotExist() {
            return orElseNull() == null;
        }
    }

    /**
     *  Use this to turn this nd-array into a {@link String} instance based on the provided
     *  {@link NDPrintSettings} instance, which allows you to configure things
     *  like the number of chars per entry, delimiters, the number of items per line, etc.
     */
    default String toString( NDPrintSettings config ) {
        return NdaAsString.representing( this ).withConfig( config ).toString();
    }

    /**
     *  This allows you to provide a lambda which configures how this nd-array should be
     *  converted to {@link String} instances.
     *  The provided {@link Consumer} will receive a {@link NDPrintSettings} instance
     *  which allows you to change various settings with the help of method chaining.<br>
     *  Here is an example:
     *  <pre>{@code
     *       t.toString(it ->
     *           it.setHasSlimNumbers(false)
     *             .setIsScientific(true)
     *             .setIsCellBound(true)
     *             .setIsMultiline(true)
     *             .setCellSize(15)
     *          );
     *  }</pre>
     *
     * @param config A consumer of the {@link NDPrintSettings} ready to be configured.
     * @return The {@link String} representation of this nd-array.
     */
    default String toString( Consumer<NDPrintSettings> config ) {
        NDPrintSettings defaults = Neureka.get().settings().view().getNDPrintSettings().clone();
        config.accept(defaults);
        return NdaAsString.representing( this ).withConfig( defaults ).toString();
    }

    /**
     *  This method returns a {@link String} representation of this nd-array.
     *  The default settings are used for the conversion.
     * @return The {@link String} representation of this nd-array.
     */
    String toString();

}
