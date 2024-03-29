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

   'Any fool can write code that a computer can understand.
    Good programmers write code that humans can understand.'
    – Martin Fowler

    Use the following as search keys :)

    §(1) : CONSTRUCTION
    §(2) : FLAGS
    §(3) : COMPONENT SYSTEM
    §(4) : PROPERTIES
    §(5) : OBJECT STATE MODIFICATION
    §(6) : ND-ITERATOR LOGIC
    §(7) : COMPONENT SPECIFIC
    §(8) : (OVERLOADABLE) OPERATORS & OPERATIONS
    §(9) : SLICING, INDEXING & INJECTING
    §(10) : MAPPING

*/


package neureka;

import neureka.autograd.GraphNode;
import neureka.autograd.JITProp;
import neureka.backend.api.LazyRef;
import neureka.common.composition.Component;
import neureka.common.composition.ComponentOwner;
import neureka.common.utility.DataConverter;
import neureka.common.utility.ListReader;
import neureka.common.utility.LogUtil;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.dtype.DataType;
import neureka.dtype.NumericType;
import neureka.dtype.custom.UI16;
import neureka.dtype.custom.UI32;
import neureka.dtype.custom.UI8;
import neureka.fluent.building.NdaBuilder;
import neureka.fluent.building.states.IterByOrIterFromOrAllTensor;
import neureka.fluent.building.states.WithShapeOrScalarOrVector;
import neureka.fluent.building.states.WithShapeOrScalarOrVectorOnDevice;
import neureka.fluent.slicing.states.AxisOrGetTensor;
import neureka.framing.NDFrame;
import neureka.framing.Relation;
import neureka.math.Function;
import neureka.math.Functions;
import neureka.math.args.Arg;
import neureka.ndim.Filler;
import neureka.ndim.NDConstructor;
import neureka.ndim.NDUtil;
import neureka.ndim.config.NDConfiguration;
import neureka.optimization.Optimizer;
import neureka.optimization.OptimizerFactory;
import neureka.view.NDPrintSettings;
import neureka.view.NdaAsString;

import java.awt.image.BufferedImage;
import java.util.*;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.stream.Collector;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 *  A {@link Tensor} is a mathematical concept and type of multidimensional
 *  data-structure with certain transformation properties.
 *  Technically however, it is mostly a simple container / data-structure which can house data indexed by N dimensions.
 *  Therefore, it is often also described as a nd-array.
 *  Elements of a tensor are also mostly numeric.<br>
 *  This means that: <br>
 *  <i><b>
 *      ...a tensor of rank 0 is a scalar, a tensor of rank 1 is
 *      a vector and a tensor of rank 2 is a matrix, etc...
 *  </b></i>
 *  <br><br>
 *  Consequently, tensors are a perfect fit for applying various operations on them.
 *  Such operations might be simple element-wise operations or more complex linear operations like
 *  the dot-product, matrix- or even tensor multiplications. <br>
 *  <br>
 * @param <V> The type parameter for the individual value items within this tensor.
 */
public interface Tensor<V> extends Nda<V>, Component<Tensor<V>>, ComponentOwner<Tensor<V>>
{
    /*==================================================================================================================
    |
    |       §(1) : CONSTRUCTION
    |   ---------------------------
    */

    /**
     *  This static factory method creates and return a completely empty and undefined tensor
     *  which is void of any contents and meaning.
     *  The use case for this would be to use the produced {@link Tensor}
     *  instance as a target for an inline operations which fills the instance with an actual value. <br>
     *  An example of this approach would be to call the {@link MutateTensor#putAt(List, Nda)} method with an empty list as key.
     *  This will be interpreted as an inline copy of the contents of the
     *  second parameter into this {@link Tensor} instance.
     *
     * @return A new and completely empty / uninitialized {@link Tensor} instance.
     */
    static Tensor<Object> newInstance() { return new TensorImpl<>(); }

    /**
     *  Use this to conveniently operate on 2 tensors.
     *  A simple example would be: {@code of(a,'*',b)}.
     *
     * @param a The left operand.
     * @param o The operator, which may be '+', '-', '*'...
     * @param b The right operand.
     * @param <T> The value item type parameter for the involved tensors.
     * @return The result of the operation defined by the provided character.
     */
    static <T> Tensor<T> of(Tensor<T> a, char o, Tensor<T> b ) { return TensorImpl._of( a, String.valueOf(o), b ); }

    /**
     *  Use this to conveniently operate on 3 tensors.
     *  A simple example would be: {@code of(a,'*',b,'+',c)}.
     *
     * @param a The first and left most operand.
     * @param o1 The first operator, which may be '+', '-', '*'...
     * @param b The second operand.
     * @param o2 The second operator, which may also be '+', '-', '*'...
     * @param c The third and last operand.
     * @param <T> The value item type parameter for the involved tensors.
     * @return The result of the operations defined by the 2 provided characters.
     */
    static <T> Tensor<T> of(Tensor<T> a, char o1, Tensor<T> b, char o2, Tensor<T> c ) {
        return TensorImpl._of( a, String.valueOf(o1), b, String.valueOf(o2), c );
    }

    /**
     *  Use this to conveniently operate on a tensor.
     *  A simple example would be: {@code of("sig(tanh(",a,"))")}.
     *
     * @param e1 The first part of the string expression defining how the provided tensor should be processed.
     * @param a The tensor which ought to be sent to whatever is defined by the provided expressions.
     * @param e2 The latter part of the expression defining how the provided tensor should be executed.
     * @param <T> The value item type parameter for the involved tensor.
     * @return The result of the operation(s) defined by the provided strings.
     */
    static <T> Tensor<T> of(String e1, Tensor<T> a, String e2 ) { return TensorImpl._of( e1, a, e2 ); }

    /**
     *  Use this to conveniently operate on 2 tensors.
     *  A simple example would be: {@code of("relu(",a,'-',b,")*2")}.
     *
     * @param e1 The first part of the string expression defining how the provided tensor should be processed.
     * @param a The first tensor which ought to be sent to whatever function is defined by the provided expressions.
     * @param o An operator combining both {@code a} and {@code b} to form a result.
     * @param b The second tensor and right operand which ought to be sent to whatever function is defined by the provided expressions.
     * @param e2 The latter part of the expression defining how the provided tensor should be executed.
     * @param <T> The value item type parameter for the involved tensor.
     * @return The result of the operation(s) defined by the provided strings.
     *
     */
    static <T> Tensor<T> of(String e1, Tensor<T> a, char o, Tensor<T> b, String e2 ) {
        return TensorImpl._of( e1, a, String.valueOf(o), b, e2 );
    }

    /**
     *  Use this to conveniently operate on 3 tensors.
     *  A simple example would be:
     *  {@code of("abs((",a,"-",b,") * ",c,")")}.
     *
     * @param e1 The first part of the expression which would typically be used to define a function name.
     * @param a The first argument.
     * @param e2 The second part of the expression, which might be an operation.
     * @param b The second argument.
     * @param e3 The third part of the expression...
     * @param c The third argument.
     * @param e4 The last part of the expression which should syntactically match the other expression...
     * @param <T> The type parameter for the involved tensors.
     * @return The result of the calculation defined by the provided expressions and arguments.
     */
    static <T> Tensor<T> of(
            String e1, Tensor<T> a, String e2, Tensor<T> b, String e3, Tensor<T> c, String e4
    ) {
        LogUtil.nullArgCheck( e1, "e1", String.class, "The first expression must not be null." );
        LogUtil.nullArgCheck( a, "a", Tensor.class, "The first tensor must not be null." );
        LogUtil.nullArgCheck( e2, "e2", String.class, "The second expression part must not be null." );
        LogUtil.nullArgCheck( b, "b", Tensor.class, "The second tensor must not be null." );
        LogUtil.nullArgCheck( e3, "e3", String.class, "The third expression part must not be null." );
        LogUtil.nullArgCheck( c, "c", Tensor.class, "The third tensor must not be null." );
        LogUtil.nullArgCheck( e4, "e4", String.class, "The fourth expression part must not be null." );
        return TensorImpl._of( e1, a, e2, b, e3, c, e4 );
    }

    /**
     *  This static {@link Tensor} factory method tries to interpret the provided
     *  arguments to create the instance the use might wants.
     *
     * @param args The arguments which ought to be interpreted.
     * @return The result of the interpretation in the form of a {@link Tensor} instance of typ {@link Object}.
     */
    static <T> Tensor<T> of( Object... args ) {
        LogUtil.nullArgCheck( args, "args", Object[].class );
        return TensorImpl._of( args );
    }

    /**
     * Constructs a vector of objects based on the provided iterable.
     *
     * @param iterable The iterable of objects from which a 1D nd-array ought to be constructed.
     * @return A vector / 1D tensor of objects.
     */
    static <T> Tensor<T> of( Iterable<T> iterable ) {
        LogUtil.nullArgCheck( iterable, "iterable", Iterable.class );
        return TensorImpl._of( iterable );
    }

    /**
     *  This is a convenient factory method for creating {@link Tensor} instances for
     *  values of type {@link T} based on a list of integers
     *  defining a shape made up of axes sizes as well as a scalar value of type {@link T}
     *  which will fill out the data array spanned by the provided shape information.
     *
     * @param shape A list of integers whose values ought to define the size of the axes of the shape of the new {@link Tensor}.
     * @param item An object of type {@link T} which will populate the data array of the new instance.
     * @return A new {@link Tensor} instance for the generic type {@link T}.
     */
    static <T> Tensor<T> of( List<Integer> shape, T item ) {
        LogUtil.nullArgCheck( shape, "shape", List.class );
        LogUtil.nullArgCheck( item, "value", Object.class );
        return of( (Class<T>) item.getClass(), shape, item );
    }

    /**
     *  This is a convenient factory method for creating {@link Tensor} instances for
     *  representing items of type {@link T}. The factory method
     *  instantiates tensors based on a {@link Shape} tuple of integers
     *  defining axes sizes, and a scalar item of type {@link T}
     *  which will fill out the data array spanned by the provided shape information.
     *  A simple usage example would be:
     *  <pre>{@code
     *     Tensor.of(Shape.of( 4, 3, 6 ), 42);
     *  }</pre>
     *
     * @param shape An immutable tuple of integers whose values ought to define the size of the axes of the shape of the new {@link Tensor}.
     * @param value An object of type {@link T} which will populate the data array of the new instance.
     * @return A new {@link Tensor} instance for the generic type {@link T}.
     */
    static <T> Tensor<T> of( Shape shape, T value ) {
        LogUtil.nullArgCheck( shape, "shape", List.class );
        LogUtil.nullArgCheck( value, "value", Object.class );
        return ofAny( (Class<T>) value.getClass(), shape, value );
    }

    /**
     *  This factory method will create and return a {@link Tensor} instance
     *  based on a list of {@link Number} instances whose rounded values will be interpreted as
     *  the shape of this new {@link Tensor} instance and a seed which will serve
     *  as a source of pseudo randomness to generate the values for the new instance.
     *
     * @param shape A list of {@link Number} instances which will be interpreted as a shape array.
     * @param seed A source of pseudo randomness for the {@link Tensor} instance created by this method.
     * @return A new {@link Tensor} instance created based on a shape and a seed.
     */
    static Tensor<Double> of(List<? extends Number> shape, String seed ) {
        int[] shapeArray = new int[ shape.size() ];
        for ( int i = 0; i < shapeArray.length; i++ ) shapeArray[ i ] = shape.get( i ).intValue();
        return of( Double.class, Shape.of(shapeArray), Arg.Seed.of(seed) );
    }

    /**
     *  Creates a new {@link Tensor} instance based on a list of numbers representing the shape,
     *  and a list of values representing the value of the resulting tensor.
     *
     * @param shape A list of numbers whose integer values will be used to form the shape of the resulting {@link Tensor}.
     * @param items A list of values which will be used to populate the data array of the resulting {@link Tensor}.
     * @param <V> The type parameter of the value list and returned tensor.
     * @return A new {@link Tensor} instance constructed based on the provided shape and value list.
     */
    static <V> Tensor<V> of( List<? extends Number> shape, List<V> items ) {
        LogUtil.nullArgCheck( shape, "shape", List.class, "Null is not a valid shape!" );
        LogUtil.nullArgCheck( items, "value", List.class, "Null is not a valid value list!" );
        Class<V> typeClass = (Class<V>) Object.class;
        if ( items.size() > 0 ) typeClass = (Class<V>) items.get(0).getClass();
        return of(
                    DataType.of(typeClass),
                    Shape.of(shape),
                    items
                );
    }

    /**
     *  Creates a new {@link Tensor} instance based on a shape tuple of numbers representing the nd-array shape,
     *  and a list of items representing the value of the resulting tensor. <br>
     *  A simple usage example would be:
     *  <pre>{@code
     *     Tensor.of(Shape.of( 2, 3, 4 ), myListOfItems);
     *  }</pre>
     *
     * @param shape A shape tuple of numbers whose integer values will be used to form the shape of the resulting {@link Tensor}.
     * @param items A list of values which will be used to populate the data array of the resulting {@link Tensor}.
     * @param <V> The type parameter of the value list and returned tensor.
     * @return A new {@link Tensor} instance constructed based on the provided shape and value list.
     */
    static <V> Tensor<V> of( Shape shape, List<V> items ) {
        Class<V> typeClass = (Class<V>) Object.class;
        if ( items.size() > 0 ) typeClass = (Class<V>) items.get(0).getClass();
        return of( DataType.of(typeClass), shape, items );
    }

    /**
     *  This factory method will turn a list of values or nested lists of values into a {@link Tensor}
     *  instance with the corresponding rank and shape.
     *
     * @param conf A list of either values or nested lists which are themselves either or.
     * @return A new {@link Tensor} instance whose shape and data is based on the provided list structure.
     */
    static Tensor<Object> of( List<Object> conf ) { return of( (Class<Object>) null, conf ); }

    /**
     *  This factory method will turn a list of values or nested lists of values into a {@link Tensor}
     *  instance with the corresponding rank and shape and whose values
     *  are of the provided type.
     *
     * @param type The type of the tensor produced by this factory method.
     * @param conf A list of either values or nested lists which are themselves either or.
     * @param <T> The type parameter of the tensor returned by this factory method.
     * @return A new {@link Tensor} instance whose shape and data is based on the provided list structure.
     */
    static <T> Tensor<T> of( Class<T> type, List<Object> conf ) {
        ListReader.Result result = null;
        try {
            result = ListReader.read( conf, o -> o );
        } catch (Exception e) {
            // We don't care about the first attempt...
        }
        if ( result == null )
            result = ListReader.read( conf, o -> ( o instanceof Number ? ((Number)o).doubleValue() : o ) );
        Class<T> resultType;
        Object[] resultData;
        Shape shape = Shape.of(result.getShape());
        if ( type == null ) {
            resultType = (Class<T>) result.getType();
            resultData = result.getData().toArray();
        } else {
            DataConverter converter = DataConverter.get();
            resultType = type;
            resultData = result.getData().parallelStream().map( v -> converter.convert(v, type) ).toArray();
        }
        return of( DataType.of(resultType), shape, resultData );
    }

    /**
     *  This is the entry point to the fluent tensor builder API for building
     *  {@link Tensor} instances in a readable and type safe fashion.
     *  The returned {@link WithShapeOrScalarOrVector} is the next step in the
     *  fluent {@link Tensor} builder API which will lead to the creation
     *  of a tensor storing values defined by the provided type class.
     *  A simple usage example would be:
     *   <pre>{@code
     *      Tensor.of(Double.class)
     *            .withShape( 2, 3, 4 )
     *            .andFill( 5, 3, 5 )
     *   }</pre>
     *
     *   It is also possible to define a range using the API to populate the tensor with values:
     *   <pre>{@code
     *      Tensor.of(Double.class)
     *            .withShape( 2, 3, 4 )
     *            .andFillFrom( 2 ).to( 9 ).step( 2 )
     *   }</pre>
     *
     *   If one needs a simple scalar then the following shortcut is possible:
     *   <pre>{@code
     *      Tensor.of(Float.class).scalar( 3f )
     *   }</pre>
     *
     *   This principle works for vectors as well:
     *   <pre>{@code
     *       Tensor.of(Byte.class).vector( 2, 5, 6, 7, 8 )
     *   }</pre>
     *   For more fine-grained control over the initialization one can
     *   pass an initialization lambda to the API:
     *   <pre>{@code
     *       Tensor.of(Byte.class).withShape(2, 3).andWhere( (i, indices) -> i * 5 - 30 )
     *   }</pre>
     *   <br>
     *   Consider using the following convenience methods:
     *   {@link #ofFloats()}, {@link #ofDoubles()}, {@link #ofInts()}, {@link #ofBytes()}, {@link #ofShorts()}
     *
     * @param type The type class of the items stored by the tensor built by the exposed builder API.
     * @return The next step of the {@link Tensor} builder API which exposes methods for defining shapes.
     */
    static <V> WithShapeOrScalarOrVectorOnDevice<V> of( Class<V> type ) { return new NdaBuilder<>( type ); }

    /**
     *  This is a simple convenience method which is simply calling the {@link Tensor#of(Class)}
     *  method like so: {@code of(Double.class)}.
     *  The returned {@link WithShapeOrScalarOrVector} is the next step in the
     *  fluent {@link Tensor} builder API which in this case will lead to the creation
     *  of a tensor storing doubles. <br>
     *  A simple usage example would be:
     *  <pre>{@code
     *     Tensor.ofDoubles()
     *           .withShape( 2, 3, 4 )
     *           .andFill( 5d, 3d, 5d )
     *  }</pre>
     *
     * @return The next step of the {@link Tensor} builder API which exposes methods for defining shapes.
     */
    static WithShapeOrScalarOrVectorOnDevice<Double> ofDoubles() { return of(Double.class); }

    /**
     *  This is a simple convenience method which is simply calling the {@link Tensor#of(Class)}
     *  method like so: {@code of(Float.class)}.
     *  The returned {@link WithShapeOrScalarOrVector} is the next step in the
     *  fluent {@link Tensor} builder API which in this case will lead to the creation
     *  of a tensor storing floats.<br>
     *  A simple usage example would be:
     *  <pre>{@code
     *     Tensor.ofFloats()
     *           .withShape( 2, 3, 4 )
     *           .andFill( 5f, 7f, 11f )
     *  }</pre>
     *
     *
     * @return The next step of the {@link Tensor} builder API which exposes methods for defining shapes.
     */
    static WithShapeOrScalarOrVectorOnDevice<Float> ofFloats() { return of(Float.class); }

    /**
     *  This is a simple convenience method which is simply calling the {@link Tensor#of(Class)}
     *  method like so: {@code of(Integer.class)}.
     *  The returned {@link WithShapeOrScalarOrVector} is the next step in the
     *  fluent {@link Tensor} builder API which in this case will lead to the creation
     *  of a tensor storing integers.
     *
     * @return The next step of the {@link Tensor} builder API which exposes methods for defining shapes.
     */
    static WithShapeOrScalarOrVectorOnDevice<Integer> ofInts() { return of(Integer.class); }

    /**
     *  This is a simple convenience method which is simply calling the {@link Tensor#of(Class)}
     *  method like so: {@code of(Short.class)}.
     *  The returned {@link WithShapeOrScalarOrVector} is the next step in the
     *  fluent {@link Tensor} builder API which in this case will lead to the creation
     *  of a tensor storing shorts.
     *
     * @return The next step of the {@link Tensor} builder API which exposes methods for defining shapes.
     */
    static WithShapeOrScalarOrVectorOnDevice<Short> ofShorts() { return of(Short.class); }

    /**
     *  This is a simple convenience method which is simply calling the {@link Tensor#of(Class)}
     *  method like so: {@code of(Byte.class)}.
     *  The returned {@link WithShapeOrScalarOrVector} is the next step in the
     *  fluent {@link Tensor} builder API which in this case will lead to the creation
     *  of a tensor storing bytes.
     *
     * @return The next step of the {@link Tensor} builder API which exposes methods for defining shapes.
     */
    static WithShapeOrScalarOrVectorOnDevice<Byte> ofBytes() { return of(Byte.class); }

    /**
     *  Constructs a vector of doubles based on the provided array.
     *
     * @param value The array of doubles from which a 1D tensor ought to be constructed.
     * @return A vector / 1D tensor of doubles.
     */
    static Tensor<Double> of( double... value ) { return of( Double.class, Shape.of( value.length ), value ); }

    /**
     * @param value The scalar value which ought to be represented as tensor.
     * @return A scalar double tensor.
     */
    static Tensor<Double> of( double value ) { return of( Double.class, Shape.of( 1 ), value ); }

    /**
     *  Constructs a vector of floats based on the provided array.
     *
     * @param value The array of floats from which a 1D tensor ought to be constructed.
     * @return A vector / 1D tensor of floats.
     */
    static Tensor<Float> of( float... value ) { return of( Float.class, Shape.of( value.length ), value ); }

    /**
     * @param value The scalar value which ought to be represented as tensor.
     * @return A scalar float tensor.
     */
    static Tensor<Float> of( float value ) { return of( Float.class, Shape.of( 1 ), value ); }

    /**
     *  Constructs a vector of bytes based on the provided array.
     *
     * @param value The array of bytes from which a 1D tensor ought to be constructed.
     * @return A vector / 1D tensor of bytes.
     */
    static Tensor<Byte> of( byte... value ) { return of( Byte.class, Shape.of( value.length ), value ); }

    /**
     * @param value The scalar value which ought to be represented as tensor.
     * @return A scalar byte tensor.
     */
    static Tensor<Byte> of( byte value ) { return of( Byte.class, Shape.of( 1 ), value ); }

    /**
     *  Constructs a vector of ints based on the provided array.
     *
     * @param value The array of ints from which a 1D tensor ought to be constructed.
     * @return A vector / 1D tensor of ints.
     */
    static Tensor<Integer> of( int... value ) { return of( Integer.class, Shape.of( value.length ), value ); }

    /**
     * @param value The scalar value which ought to be represented as tensor.
     * @return A scalar int tensor.
     */
    static Tensor<Integer> of( int value ) { return of( Integer.class, Shape.of( 1 ), value ); }

    /**
     *  Constructs a vector of longs based on the provided array.
     *
     * @param value The array of longs from which a 1D tensor ought to be constructed.
     * @return A vector / 1D tensor of longs.
     */
    static Tensor<Long> of( long... value ) { return of( Long.class, Shape.of( value.length ), value ); }

    /**
     * @param value The scalar value which ought to be represented as tensor.
     * @return A scalar long tensor.
     */
    static Tensor<Long> of( long value ) { return of( Long.class, Shape.of( 1 ), value ); }

    /**
     *  Constructs a vector of shorts based on the provided array.
     *
     * @param value The array of shorts from which a 1D tensor ought to be constructed.
     * @return A vector / 1D tensor of shorts.
     */
    static Tensor<Short> of( short... value ) { return of( Short.class, Shape.of( value.length ), value ); }

    /**
     * @param value The scalar value which ought to be represented as tensor.
     * @return A scalar short tensor.
     */
    static Tensor<Short> of( short value ) { return of( Short.class, Shape.of( 1 ), value ); }

    /**
     *  Constructs a vector of booleans based on the provided array.
     *
     * @param value The array of booleans from which a 1D tensor ought to be constructed.
     * @return A vector / 1D tensor of shorts.
     */
    static Tensor<Boolean> of( boolean... value ) { return of( Boolean.class, Shape.of( value.length ), value ); }

    /**
     *  Use this to construct and return a seeded tensor of the specified type.
     *
     * @param valueType The type class of the items stored by the resulting tensor.
     * @param shape The shape of the resulting tensor consisting of any number of axis-sizes.
     * @param seed An arbitrary {@link String} whose hash will be used to as a seed.
     * @param <V> The type parameter of individual tensor items.
     * @return A newly created and seeded tensor of the provided type and shape.
     */
    static <V> Tensor<V> of( Class<V> valueType, Shape shape, Arg.Seed seed ) { return TensorImpl._of( valueType, NDConstructor.of(shape), seed ); }

    /**
     *  Use this to construct and return a homogeneously populated double tensor of the specified shape.
     *
     * @param shape The shape of the resulting tensor consisting of any number of axis-sizes.
     * @param value The value which ought to be used to populate the tensor homogeneously.
     * @return A new tensor instance with the provided shape and initial value.
     */
    static Tensor<Double> of( Shape shape, double value ) { return ofAny( Double.class, shape, value ); }

    /**
     *  Use this to construct and return a double tensor of the specified shape and initial values.
     *  The length of the provided array does not have to match the number of elements
     *  defined by the provided shape, the tensor will be populated based on repeated iteration over the
     *  provided double array.
     *
     * @param shape The shape of the resulting tensor consisting of any number of axis-sizes.
     * @param values The values which ought to be used to populate the tensor.
     */
    static Tensor<Double> of( Shape shape, double[] values ) { return ofAny( Double.class, shape, values ); }

    /**
     *  Use this to construct and return an int tensor of the specified shape and initial values.
     *  The length of the provided array does not have to match the number of elements
     *  defined by the provided shape, the tensor will be populated based on repeated iteration over the
     *  provided int array.
     *
     * @param shape The shape of the resulting tensor consisting of any number of axis-sizes.
     * @param values The values which ought to be used to populate the tensor.
     */
    static Tensor<Integer> of( Shape shape, int[] values ) { return ofAny( Integer.class, shape, values ); }

    /**
     *  Use this to construct and return a byte tensor of the specified shape and initial values.
     *  The length of the provided array does not have to match the number of elements
     *  defined by the provided shape, the tensor will be populated based on repeated iteration over the
     *  provided byte array..
     *
     * @param shape The shape of the resulting tensor consisting of any number of axis-sizes.
     * @param values The values which ought to be used to populate the tensor.
     */
    static Tensor<Byte> of( Shape shape, byte[] values ) { return ofAny( Byte.class, shape, values ); }

    /**
     *  Use this to construct and return a long tensor of the specified shape and initial values.
     *  The length of the provided array does not have to match the number of elements
     *  defined by the provided shape, the tensor will be populated based on repeated iteration over the
     *  provided long array..
     *
     * @param shape The shape of the resulting tensor consisting of any number of axis-sizes.
     * @param values The values which ought to be used to populate the tensor.
     */
    static Tensor<Long> of( Shape shape, long[] values ) { return ofAny( Long.class, shape, values ); }

    /**
     *  Use this to construct and return a short tensor of the specified shape and initial values.
     *  The length of the provided array does not have to match the number of elements
     *  defined by the provided shape, the tensor will be populated based on repeated iteration over the
     *  provided short array..
     *
     * @param shape The shape of the resulting tensor consisting of any number of axis-sizes.
     * @param values The values which ought to be used to populate the tensor.
     */
    static Tensor<Short> of( Shape shape, short[] values ) { return ofAny( Short.class, shape, values ); }

    /**
     *  Use this to construct and return a float tensor of the specified shape and initial values.
     *  The length of the provided array does not have to match the number of elements
     *  defined by the provided shape, the tensor will be populated based on repeated iteration over the
     *  provided float array..
     *
     * @param shape The shape of the resulting tensor consisting of any number of axis-sizes.
     * @param values The values which ought to be used to populate the tensor.
     */
    static Tensor<Float> of( Shape shape, float[] values ) { return ofAny( Float.class, shape, values ); }

    /**
     *  Use this to construct and return a homogeneously populated float tensor of the specified shape.
     *
     * @param shape The shape of the resulting tensor consisting of any number of axis-sizes.
     * @param value The value which ought to be used to populate the tensor homogeneously.
     * @return A new tensor instance with the provided shape and initial value.
     */
    static Tensor<Float> of( Shape shape, float value ) { return ofAny( Float.class, shape, value ); }

    /**
     *  Use this to construct and return a boolean tensor of the specified shape and initial values.
     *  The length of the provided array does not have to match the number of elements
     *  defined by the provided shape, the tensor will be populated based on repeated iteration over the
     *  provided boolean array..
     *
     * @param shape The shape of the resulting tensor consisting of any number of axis-sizes.
     * @param values The values which ought to be used to populate the tensor.
     */
    static Tensor<Boolean> of( Shape shape, boolean[] values ) { return ofAny( Boolean.class, shape, values ); }

    /**
     *  Use this to construct and return a tensor of the specified shape and data object.<br>
     *  This method is typically used like this:<br>
     *  <pre>{@code
     *      Tsr<Integer> tensor = Tsr.of( Shape.of(2,3), Data.of(1,2,3,4,5,6) );
     *  }</pre>
     *  The resulting tensor will have the shape {@code [2,3]} and the values {@code [1,2,3,4,5,6]}.
     *
     * @param shape The shape of the resulting tensor consisting of any number of axis-sizes.
     * @param data The data object which contains the values to be used to populate the tensor.
     * @return A newly created tensor of the provided shape and data.
     * @param <V> The type parameter of individual tensor items.
     */
    static <V> Tensor<V> of( Shape shape, Data<V> data ) {
        return Tensor.of( data.dataType().getItemTypeClass(), shape, data.getOrNull() );
    }

    /**
     *  Use this to construct and return a tensor of the specified type and shape.
     *
     * @param type The type of the items stored by the resulting tensor.
     * @param shape The shape of the resulting tensor consisting of any number of axis-sizes.
     * @param <V> The type parameter of individual tensor items.
     * @return A newly created tensor of the provided type and shape.
     */
    static <V> Tensor<V> of( DataType<V> type, Shape shape ) { return TensorImpl._of( NDConstructor.of(shape), type ); }

    /**
     *  Use this to construct and return a tensor of the specified type, shape and data object.
     *
     * @param type The type of the items stored by the resulting tensor.
     * @param shape The shape of the resulting tensor consisting of an array of axis-sizes.
     * @param data The data object which will be used to populate the tensor.
     * @param <V> The type parameter of individual tensor items.
     * @return A newly created tensor of the provided type, shape and data.
     */
    static <V> Tensor<V> of( Class<V> type, Shape shape, Object data ) { return of( DataType.of(type), shape, data ); }

    /**
     *  Use this to construct and return a tensor of the specified type, shape and data object.
     *
     * @param type The type of the items stored by the resulting tensor.
     * @param shape The shape of the resulting tensor consisting of list of axis-sizes.
     * @param data The data object which will be used to populate the tensor.
     * @param <V> The type parameter of individual tensor items.
     * @return A newly created tensor of the provided type, shape and data.
     */
    static <V> Tensor<V> of( Class<V> type, List<Integer> shape, Object data ) {
        return of( DataType.of(type), Shape.of(shape), data );
    }

    /**
     *  Use this to construct and return a tensor of the specified type, shape and number.
     *
     * @param type The type of the items stored by the resulting tensor.
     * @param shape The shape of the resulting tensor consisting of a immutable tuple of axis-sizes.
     * @param data The data object which will be used to populate the tensor.
     * @param <V> The type parameter of individual tensor items.
     * @return A newly created tensor of the provided type, shape and data.
     */
    static <V extends Number> Tensor<V> of( Class<V> type, Shape shape, Number data ) {
        return of( DataType.of(type), shape, data );
    }

    /**
     *  Use this to construct and return a tensor of the specified type, shape and data object.
     *
     * @param type The type of the items stored by the resulting tensor.
     * @param shape The shape of the resulting tensor consisting of a immutable tuple of axis-sizes.
     * @param data The data object which will be used to populate the tensor.
     * @param <V> The type parameter of individual tensor items.
     * @return A newly created tensor of the provided type, shape and data.
     */
    static <V> Tensor<V> ofAny( Class<V> type, Shape shape, Object data ) {
        return of( DataType.of(type), shape, data );
    }

    /**
     *  Use this to construct and return a tensor of the specified type, shape and data object.
     *
     * @param type The type of the items stored by the resulting tensor.
     * @param shape The shape of the resulting tensor consisting of a list of axis-sizes.
     * @param data The list of items which will be used to populate the tensor.
     * @param <V> The type parameter of individual tensor items.
     * @return A newly created tensor of the provided type, shape and data.
     */
    static <V> Tensor<V> of( Class<V> type, List<Integer> shape, List<V> data ) {
        return of( DataType.of( type ), Shape.of(shape), data );
    }

    /**
     *  Use this to construct and return a tensor of the specified type, shape and list of items.
     *  Here a simple usage example:  <br>
     *  <pre>{@code
     *      Tsr<Float> tensor = Tsr.of( Float.class, Shape.of(2,3), List.of(1f,2f,3f,4f,5f,6f) );
     *  }</pre>
     *
     * @param type The type of the items stored by the resulting tensor.
     * @param shape The shape of the resulting tensor consisting of an immutable tuple of axis-sizes.
     * @param data The list of items which will be used to populate the tensor.
     * @param <V> The type parameter of individual tensor items.
     * @return A newly created tensor of the provided type, shape and data.
     */
    static <V> Tensor<V> of( Class<V> type, Shape shape, List<V> data ) {
        return of( DataType.of( type ), shape, data );
    }

    /**
     *  Use this to construct and return a tensor of the specified type, shape and data object.
     *
     * @param dataType The type of the items stored by the resulting tensor.
     * @param shape The shape of the resulting tensor consisting of a list of axis-sizes.
     * @param data The data object which will be used to populate the tensor.
     * @param <V> The type parameter of individual tensor items.
     * @return A newly created tensor of the provided type, shape and data.
     */
    static <V> Tensor<V> of( DataType<V> dataType, List<Integer> shape, List<V> data ) {
        return of( dataType, Shape.of(shape), data.toArray() );
    }

    /**
     *  Use this to construct and return a tensor of the specified type, shape and a list of items.
     *  Here a simple usage example:  <br>
     *  <pre>{@code
     *      Tsr<Integer> tensor = Tsr.of( DataType.F32, Shape.of(2,3), List.of(1,2,3,4,5,6) );
     *  }</pre>
     *
     * @param dataType The type of the items stored by the resulting tensor.
     * @param shape The shape of the resulting tensor consisting of an immutable tuple of axis-sizes.
     * @param data The list of items which will be used to populate the tensor.
     * @param <V> The type parameter of individual tensor items.
     * @return A newly created tensor of the provided type, shape and data.
     */
    static <V> Tensor<V> of( DataType<V> dataType, Shape shape, List<V> data ) {
        LogUtil.nullArgCheck( dataType, "dataType", DataType.class, "Null is not a valid data type!" );
        LogUtil.nullArgCheck( shape, "shape", Shape.class, "Null is not a valid shape!" );
        LogUtil.nullArgCheck( data, "data", List.class, "Null is not a valid data object!" );
        return of( dataType, shape, data.toArray() );
    }

    /**
     *  This factory method is among the most flexible and forgiving ways to create a {@link Tensor} instance.
     *  It receives a {@link DataType} for type safety and to ensure that the produced {@link Tensor} instance
     *  will contain elements of the correct type, and a {@link Shape} tuple which stores the sizes of the axes that the
     *  instance ought to possess, and finally it receives a data {@link Object} which can be anything ranging from
     *  a {@link List} to an array or simply a single value which ought to fill out the entire {@link Tensor}.
     *
     * @param dataType The data type of the data represented by {@link Tensor} instance created by this method.
     * @param shape An immutable tuple of axis sizes describing the dimensionality of the {@link Tensor} created by this method.
     * @param data The data for the {@link Tensor} that is about to be created, which can be a list, an array or scalar.
     * @return A new {@link Tensor} instance of the specified type, shape and containing the provided data.
     */
    static <V> Tensor<V> of( DataType<V> dataType, Shape shape, Object data ) {
        return TensorImpl._of( NDConstructor.of(shape), CPU.get(), dataType, data );
    }

    /**
     *  This factory method is among the most flexible and forgiving ways to create a {@link Tensor} instance.
     *  It receives a {@link DataType} for type safety and to ensure that the produced {@link Tensor} instance
     *  will contain elements of the correct type, and a {@link Shape} tuple which stores the sizes of the axes that the
     *  instance ought to possess, and finally it receives a data {@link Object} which can be anything ranging from
     *  a {@link List} to an array or simply a single value which ought to fill out the entire {@link Tensor}.
     *
     * @param dataType The data type of the data represented by {@link Tensor} instance created by this method.
     * @param device The device on which the tensor will be stored.
     * @param shape An immutable tuple of axis sizes describing the dimensionality of the {@link Tensor} created by this method.
     * @param data The data for the {@link Tensor} that is about to be created, which can be a list, an array or scalar.
     * @return A new {@link Tensor} instance of the specified type, shape and containing the provided data.
     */
    static <V extends N, N> Tensor<V> of( DataType<V> dataType, Device<N> device, Shape shape, Object data ) {
        return TensorImpl._of( NDConstructor.of(shape), device, dataType, data );
    }

    /**
     *  This factory method a raw tensor constructor which will not perform any type checking
     *  or data conversion on the data provided to it.
     *  It constructs the tensor expecting that the data provided to it is of the correct type
     *  and an array of axis sizes.
     *
     * @param dataType The data type of the data represented by {@link Tensor} instance created by this method.
     * @param ndConstructor The {@link NDConstructor} that will be used to construct the {@link Tensor} instance.
     * @param data The data for the {@link Tensor} that is about to be created, which is expected to be an array.
     * @return A new {@link Tensor} instance of the specified type, shape and containing the provided data.
     * @param <V> The type parameter of individual tensor items.
     */
    static <V> Tensor<V> of( DataType<V> dataType, NDConstructor ndConstructor, Data<V> data ) { return TensorImpl._of( ndConstructor, dataType, data ); }

    /**
     *  This factory method allows the creation of tensors with an additional initialization
     *  lambda for filling the underlying data array with desired values.
     *  Other than regular numeric types it is also possible to initialize the
     *  tensor with regular Objects like String instances or custom data types like complex
     *  numbers for example... <br>
     *  Therefore the constructor requires not only a shape as argument but also
     *  the data type which ought to be allocated as well as the initialization
     *  lambda which will be called iteratively.
     *
     * @param type The data type this tensor ought to have.
     * @param shape The shape of this new tensor ought to have.
     * @param filler The lambda Object which ought to fill this tensor with the appropriate data.
     * @param <T> The type parameter for the actual data array items.
     */
    static <T> Tensor<T> of( DataType<T> type, List<Integer> shape, Filler<T> filler ) {
        LogUtil.nullArgCheck( shape, "shape", List.class );
        return of( type, Shape.of(shape), filler );
    }

    /**
     *  This factory method allows the creation of tensors with an additional initialization
     *  lambda for filling the underlying data array with desired values.
     *  Other than regular numeric types it is also possible to initialize the
     *  tensor with regular Objects like String instances or custom data types like complex
     *  numbers for example... <br>
     *  Therefore the constructor requires not only a shape as argument but also
     *  the data type which ought to be allocated as well as the initialization
     *  lambda which will be called iteratively.
     *  Here a simple usage example:  <br>
     *  <pre>{@code
     *      Tsr<Double> tensor = Tsr.of( DataType.F64, Shape.of(2, 3), (i, j) -> i + j );
     *  }</pre>
     *
     * @param type The data type this tensor ought to have.
     * @param shape The shape of this new tensor ought to have.
     * @param filler The lambda Object which ought to fill this tensor with the appropriate data.
     * @param <T> The type parameter for the actual data array items.
     */
    static <T> Tensor<T> of( DataType<T> type, Shape shape, Filler<T> filler) {
        LogUtil.nullArgCheck( shape, "shape", Shape.class );
        return TensorImpl._of( NDConstructor.of(shape), type, filler );
    }

    /**
     *  This factory method allows the creation of tensors with an additional initialization
     *  lambda for filling the underlying data array with desired values.
     *  Other than regular numeric types it is also possible to initialize the
     *  tensor with regular Objects like String instances or custom data types like complex
     *  numbers for example... <br>
     *  Therefore the constructor requires not only a shape as argument but also
     *  the data type which ought to be allocated as well as the initialization
     *  lambda which will be called iteratively.
     *
     * @param type The data type class the items of this tensor ought to have.
     * @param shape The shape of this new tensor ought to have.
     * @param filler The lambda Object which ought to fill this tensor with the appropriate data.
     * @param <T> The type parameter for the actual data array items.
     */
    static <T> Tensor<T> of( Class<T> type, Shape shape, Filler<T> filler ) {
        return of( DataType.of(type), shape, filler );
    }

    /**
     *  This factory method allows for the creation and execution of {@link Function} instances
     *  without actually instantiating them manually,
     *  where the result will then be returned by this factory method. <br><br>
     *  The passed {@link String} will be parsed into a {@link Function} AST which will be cached
     *  using the expression as key in case it will be used in future constructor calls
     *  like this one, or elsewhere...
     *  The created / retrieved {@link Function} will then be called with the supplied input list
     *  in order to trigger an execution.
     *  The result of which will be used for the population of the fields of this
     *  very instance.                                                                      <br>
     *  An example would be the following :                                                 <br>
     * <ul>
     *      <li><i> 'var a = Tsr.of( "sin( I[0] ) / I[1]", 12f, -6.34f )'</i></li>
     * </ul>
     *
     * @param expression A String which will be used for parsing a Function AST.
     * @param inputs An array of inputs which can be tensors or numeric types.
     */
    @SafeVarargs
    static <V extends Number> Tensor<V> of( String expression, V... inputs ) {
        return Function.of( expression, true ).call( Arrays.stream(inputs).map(args -> TensorImpl._of(args)).toArray(Tensor[]::new) );
    }

    /**
     *  This factory method allows for the creation and execution of {@link Function} instances
     *  without actually instantiating them manually,
     *  where the result will then be returned by this factory method. <br><br>
     *  The passed {@link String} will be parsed into a {@link Function} AST which will be cached
     *  using the expression as key in case it will be used in future constructor calls
     *  like this one, or elsewhere...
     *  The created / retrieved {@link Function} will then be called with the supplied input list
     *  in order to trigger an execution.
     *  The result of which will be used for the population of the fields of this
     *  very instance.                                                                      <br>
     *  An example would be the following :                                                 <br>
     * <ul>
     *      <li><i> 'var a = Tsr.of( "sin( I[0] ) / I[1]", List.of(b, c) )'</i></li>
     * </ul>
     *
     * @param expression A String which will be used for parsing a Function AST.
     * @param inputs A list of inputs which can be tensors or numeric types.
     */
    static <V> Tensor<V> of( String expression, List<Tensor<V>> inputs ) {
        return Function.of( expression, true ).call( inputs );
    }

    /**
     *  This method takes a list of tensors and a String expression describing
     *  operations which ought to be applied to the tensors in said list.
     *  It also receives a boolean flag which determines if the defined function
     *  should be executed with autograd enabled.
     *  The provided expression will be parsed to a {@link Function} instance expecting as many inputs
     *  as there are array entries, namely : "I[0]", "I[1]", "I[2]", ...                    <br>
     *  An example would be the following :                                                 <br>
     * <ul>
     *      <li><i> 'var a = Tsr.of( "sin( I[0] ) / I[1]", true, List.of(b, c) )'</i></li>
     * </ul>
     *  Which takes the tensor 'b' and 'c' and applies the function "f(x,y) = sin(x) / y"
     *  element-wise to produce a new tensor 'a'!
     *  Additionally, there is a helpful flag which allows one to specify if the
     *  parsed {@link Function} instance emerging from the provided expression
     *  should also allow the tracking of computations via a computation graph ({@link GraphNode} instances).
     *  This history tracking then enables auto-differentiation. <br>
     *
     * @param expression The expression describing operations applied to the provided tensors.
     * @param doAD A flag which when set to true commands the creation of a computation graph during operation execution.
     * @param tensors A list of tensors used as inputs to the Function instance parsed from the provided expression.
     */
    static <V> Tensor<V> of( String expression, boolean doAD, List<Tensor<V>> tensors ) {
        return Function.of( expression, doAD ).call( tensors );
    }

    /**
     *  This method takes a tensor and a String expression describing
     *  operations which ought to be applied to said tensor.
     *  This expression will be parsed to a {@link Function} instance expecting one input,
     *  namely : "I[0]" <br>
     *  An example would be the following :
     * <ul>
     *      <li><i> 'var a = Tsr.of( "sin( I[0] ) * 2", b )'</i></li>
     * </ul>
     *
     *  Which takes the tensor 'b' and applies the function "f(x) = sin(x) * 2"
     *  element-wise to produce a new tensor 'a'! <br>
     *  <br>
     *
     * @param tensor A tensor which serves as input to the Function instance parsed from the given expression.
     * @param expression The expression describing operations applied to the provided tensor.
     */
    static <V> Tensor<V> of( String expression, Tensor<V> tensor ) {
        return Function.of( expression, true ).call( tensor );
    }

    /**
     *  This method takes an array of tensors and a String expression describing
     *  operations which ought to be applied to the tensors in said array.
     *  This expression will be parsed to a {@link Function} instance expecting as many inputs
     *  as there are array entries, namely : "I[0]", "I[1]", "I[2]", ... <br>
     *  An example would be the following :
     * <ul>
     *      <li><i> 'var a = Tsr.of( "sin( I[0] ) / I[1]", b, c )'</i></li>
     * </ul>
     *
     *  Which takes the tensor 'b' and 'c' and applies the function "f(x,y) = sin(x) / y"
     *  element-wise to produce a new tensor 'a'! <br>
     *
     * @param expression The expression describing operations applied to the provided tensors.
     * @param tensors An array of tensors used as inputs to the Function instance parsed from the provided expression.
     */
    @SafeVarargs
    static <V> Tensor<V> of( String expression, Tensor<V>... tensors ) {
        return Function.of( expression, true ).call( tensors );
    }

    /**
     *  This method takes an array of tensors and a String expression describing
     *  operations which ought to be applied to the tensors in said array.
     *  It also receives a boolean flag which determines if the defined function
     *  should be executed with autograd enabled.
     *  The provided expression will be parsed to a {@link Function} instance expecting as many inputs
     *  as there are array entries, namely : "I[0]", "I[1]", "I[2]", ...                    <br>
     *  An example would be the following :                                                 <br>
     * <ul>
     *      <li><i> 'var a = Tsr.of( "sin( I[0] ) / I[1]", true, b, c )'</i></li>
     * </ul>
     *  Which takes the tensor 'b' and 'c' and applies the function "f(x,y) = sin(x) / y"
     *  element-wise to produce a new tensor 'a'!
     *  Additionally, there is a helpful flag which allows one to specify if the
     *  parsed {@link Function} instance emerging from the provided expression
     *  should also allow the tracking of computations via a computation graph ({@link GraphNode} instances).
     *  This history tracking then enables auto-differentiation. <br>
     *
     * @param expression The expression describing operations applied to the provided tensors.
     * @param doAD A flag which when set to true commands the creation of a computation graph during operation execution.
     * @param tensors An array of tensors used as inputs to the Function instance parsed from the provided expression.
     */
    @SafeVarargs
    static <V> Tensor<V> of( String expression, boolean doAD, Tensor<V>... tensors ) {
        return Function.of( expression, doAD ).call( tensors );
    }

    /**
     *  This factory method produces a randomly populated tensor of the provided
     *  type and shape using a hard coded default seed.
     *  If the provided type class is representing a
     *  floating point number type (like {@link Double} or {@link Float}) then the random numbers will
     *  be gaussian ("normally") distributed values with mean {@code 0.0} and standard
     *  deviation {@code 1.0}.
     *
     * @param valueTypeClass The type class of the values stored by the returned tensor.
     * @param shape The shape of the tensor produced by this factory method.
     * @param <V> The type parameter of the values stored by the returned tensor.
     * @return A randomly filled tensor of the provided type.
     */
    static <V> Tensor<V> ofRandom( Class<V> valueTypeClass, int... shape ) {
        return of( valueTypeClass )
                .withShape( shape )
                .andSeed( 8701252152903546L );// If the user does not provide a seed, we use this.
    }

    /**
     *  Use this factory method to instantiate a new tensor with the same data type, shape
     *  and memory location ({@link Device} instance) as the provided template tensor.
     *
     * @param template The template tensor whose type, shape and location should be taken to construct a new tensor.
     * @param <V> The type parameter defining the value type of the provided as well as returned tensor.
     * @return A new {@link Tensor} instance with the same data type, shape and memory location as the provided template.
     */
    static <V> IterByOrIterFromOrAllTensor<V> like( Tensor<V> template ) {
        return of( template.getDataType().getItemTypeClass() )
                .on( template.getDevice() )
                .withShape( template.getNDConf().shape() );
    }

    /**
     * Returns a {@code Collector} that accumulates the input elements into a
     * new {@link Tensor} with the specified shape. <br>
     * Usage example : <br>
     * <pre>{@code
     *    var tensor = Stream.of( 1, 2, 3, 4, 5, 6 )
     *                      .collect( Tsr.shaped( 2, 3 ) );
     * }</pre>
     *
     * @param shape The shape of the tensor to be returned.
     * @param <T> the type of the input elements
     * @return a {@code Collector} which collects all the input elements into a
     *          {@link Tensor}, in encounter order.
     */
    static <T> Collector<T, ?, Tensor<T>> shaped( int... shape ) { return shaped( Shape.of(shape) ); }

    /**
     * Returns a {@code Collector} that accumulates the input elements into a
     * new {@link Tensor} with the specified shape. <br>
     * Usage example : <br>
     * <pre>{@code
     *    var tensor = Stream.of( 1, 2, 3, 4, 5, 6 )
     *                      .collect( Tsr.shaped( otherTensor.shape() ) );
     * }</pre>
     *
     * @param shape The shape of the tensor to be returned.
     * @param <T> the type of the input elements
     * @return a {@code Collector} which collects all the input elements into a
     *          {@link Tensor}, in encounter order.
     */
    static <T> Collector<T, ?, Tensor<T>> shaped(Shape shape ) {
        return Collector.of(
                    (Supplier<List<T>>) ArrayList::new,
                    List::add,
                    (left, right) -> { left.addAll(right); return left; },
                    list -> Tensor.of( shape, list )
                );
    }

    /*==================================================================================================================
    |
    |       §(2) : FLAGS
    |   ----------------------
    */

    /**
     *  Setting this flag to {@code true} will tell the autograd system to accumulate gradients at this tensor.
     *  This is achieved by allowing for the recording of a computation graph
     *  for when this tensor is used in any autograd supporting operations.
     *  This allows the autograd / auto-differentiation system to traverse said graph
     *  for when the {@link #backward()} method is called
     *  on any descendant tensor at the most recent end of the computation graph.
     *
     * @param rqsGradient The truth value determining if this tensor ought to receive gradients via
     *                     the built-in automatic backpropagation system.
     * @return This very {@link Tensor} instance in order to enable method chaining.
     */
    Tensor<V> setRqsGradient(boolean rqsGradient );

    /**
     *  This flag will indirectly trigger the activation of the autograd / auto-differentiation system of this library!
     *  If the flag is set to 'true' and the tensor is used for computation then
     *  it will also receive gradients when the {@link #backward()} method is being called
     *  on any descendant tensor within the computation graph.
     *
     * @return The truth value determining if this tensor ought to receive gradients via
     *         the built-in automatic backpropagation system.
     */
    boolean rqsGradient();

    /**
     *  Intermediate tensors are internal non-user tensors which may be eligible
     *  for deletion when further consumed by a {@link Function}.
     *  For the casual user of Neureka, this flag should always be false!
     *
     * @return The truth value determining if this tensor is not a user tensor but an internal
     *         tensor which may be eligible for deletion by {@link Function}s consuming it.
     */
    boolean isIntermediate();

    /**
     *  Outsourced means that the tensor is stored on a {@link Device} implementation instance which is not the {@link CPU}.
     *
     * @return The truth value determining if the data of this tensor is not actually stored inside it
     *         in the form of a traditional primitive JVM array!
     */
    default boolean isOutsourced() { return !(this.getDevice() instanceof CPU); }

    /**
     *  A Virtual tensor is a tensor whose underlying data array is of size 1, holding only a single value. <br>
     *  This only makes sense for homogeneously populated tensors.
     *  An example of such a tensor would be: <br>
     *  {@code Tsr.ofInts().withShape(x,y).all(n)}                           <br><br>
     *  The reasons for this feature is that it greatly improves performance in certain cases.
     *  In essence this feature is a form of lazy loading.
     *  <p>
     *  Use {@link MutateTensor#setIsVirtual(boolean)} to "actualize" a "virtual" tensor, and vise versa.
     *
     * @return The truth value determining if this tensor is "virtual" or "actual".
     */
    boolean isVirtual();

    /**
     *  This will check if the {@link MutateTensor#delete()} method was previously called on this tensor.
     *  This means that the tensor data was freed on every device
     *  and any references inside the tensor are null (to be eligable for garbage collection).
     *
     * @return The truth value determining if the {@link MutateTensor#delete()} method has been called oin this instance.
     */
    boolean isDeleted();

    /**
     *  A tensor is empty if it's {@link Data} storage is null.
     *  This is true for deleted tensors or tensors which have not been initialized yet.
     *
     * @return The truth value determining if this tensor has no {@link Data}.
     */
    default boolean isEmpty() { return getMut().getData() == null || getMut().getData().getOrNull() == null; }

    /**
     *  A tensor is "undefined" if it has either no {@link NDConfiguration} implementation instance
     *  or this instance does not have a shape set for this {@link Tensor} which is needed for
     *  a tensor to also have a rank and dimensionality...
     *
     * @return The truth value determining if this tensor has an {@link NDConfiguration} stored internally.
     */
    default boolean isUndefined() { return getNDConf() == null || getNDConf().shape() == null; }

    /** {@inheritDoc} */
    @Override
    default boolean isSlice() {
        return this.find(Relation.class).map(Relation::hasParent).orElse(false);
    }

    /** {@inheritDoc} */
    @Override
    default boolean isShallowCopy() {
        return this
                .find( Relation.class )
                .map( r -> (Relation<V>) r )
                .map( child ->
                        child.getParent()
                                .map( p -> p.getNDConf().equals(this.getNDConf()) )
                                .orElse(false)
                    /*
                        Note:
                        A shallow copy is conceptually always a "full slice" of the parent tensor.
                        This means that the parent tensor and the shallow copy
                        share the same nd-configurations (shape and data access pattern).
                     */
                )
                .orElse(false);
    }

    /** {@inheritDoc} */
    @Override
    default boolean isPartialSlice() {
        return this
                .find( Relation.class )
                .map( r -> (Relation<V>) r )
                .map( child -> child.getParent()
                                    .map( p -> p.size() > this.size() )
                                    .orElse(false)
                    /*
                        Note:
                        A partial slice is a slice which does not have the same size as the parent tensor
                        but still sharing the same underlying data as the parent tensor.
                     */
                )
                .orElse(false);
    }

    /** {@inheritDoc} */
    @Override
    default int sliceCount() { return this.find(Relation.class).map(Relation::childCount).orElse(0); }

    /** {@inheritDoc} */
    @Override
    default boolean isSliceParent() { return this.find( Relation.class ).map(Relation::hasChildren).orElse(false); }

    /**
     *  Tensors which are used or produced by the autograd system will have a {@link GraphNode} component attached to them.
     *  This is because autograd requires recording a computation graph for back-prop traversal.
     *  This autograd system however, will only be triggered by {@link Function} implementations which
     *  are not "detached", meaning they have their "{@link Function#isDoingAD()}" flags set to true! <br>
     *  Detached functions (like those pre-instantiated in Function.Detached.*) will not attach {@link GraphNode}
     *  instances to involved tensors which will prevent the formation of a computation graph.
     *
     * @return The truth value determining if this tensor belongs to a recorded computation graph.
     */
    default boolean belongsToGraph() { return this.graphNode().isPresent(); }

    /**
     *  Tensors which are used or produced by the autograd system will have a {@link GraphNode} component attached to them.
     *  This is because autograd requires recording a computation graph for back-prop traversal.
     *  This autograd system however, will only be triggered by {@link Function} implementations which
     *  are not "detached", meaning they have their "{@link Function#isDoingAD()}" flags set to true! <br>
     *  A tensor is a leave if it is attached to a computation graph in which it is not an intermediate / branch node
     *  but input / branch node.
     *
     * @return The truth value determining if this tensor is attached to a computation graph as leave node.
     */
    default boolean isLeave() { return (!this.belongsToGraph() || this.graphNode().map(GraphNode::isLeave).orElse(false) ); }

    /**
     *  Tensors which are used or produced by the autograd system will have a {@link GraphNode} component attached to them.
     *  This is because autograd requires recording a computation graph for back-prop traversal.
     *  This autograd system however, will only be triggered by {@link Function} implementations which
     *  are not "detached", meaning they have their "{@link Function#isDoingAD()}" flags set to true! <br>
     *  A tensor is a branch if it is attached to a computation graph in which it is not an input / leave node
     *  but intermediate / branch node.
     *
     * @return The truth value determining if this tensor is attached to a computation graph as branch node.
     */
     default boolean isBranch() { return !this.isLeave(); }

    /**
     *  Tensors can be components of other tensors which makes the
     *  implicitly their gradients.
     *
     * @return The truth value determining if this tensor has another tensor attached to it (which is its gradient).
     */
    default boolean hasGradient() { return this.has( Tensor.class ); }

    /**
     *  This flag works alongside two autograd features which can be enabled inside the library settings.
     *  They will come into effect when flipping their feature flags, <br>
     *  namely: <i>'isApplyingGradientWhenRequested'</i> and <i>'isApplyingGradientWhenTensorIsUsed'</i><br>
     *  As the first flag name suggests gradients will be applied to their tensors when it is set to true,
     *  however this will only happen when the second flag is set to true as well, because otherwise gradients
     *  wouldn't be applied to their tensors automatically in the first place... <br>
     *  <br>
     *  Setting both flags to true will inhibit the effect of the second setting <i>'isApplyingGradientWhenTensorIsUsed'</i>
     *  unless a form of "permission" is being signaled to the autograd system.
     *  This signal comes in the form of a "request" flag which marks a tensor as <b>allowed to
     *  be updated by its gradient</b>.<br>
     *  <br>
     * @return The truth value determining if the application of the gradient of this tensor is requested.
     */
    boolean gradientApplyRequested();

    /**
     *  This flag works alongside two autograd features which can be enabled inside the library settings.
     *  They will come into effect when flipping their feature flags, <br>
     *  namely: <i>'isApplyingGradientWhenRequested'</i> and <i>'isApplyingGradientWhenTensorIsUsed'</i><br>
     *  As the first flag name suggests gradients will be applied to their tensors when it is set to true,
     *  however this will only happen when the second flag is set to true as well, because otherwise gradients
     *  wouldn't be applied to their tensors automatically in the first place... <br>
     *  <br>
     *  Setting both flags to true will inhibit effect of the second setting <i>'isApplyingGradientWhenTensorIsUsed'</i>
     *  unless a form of "permission" is being signaled to the autograd system.
     *  This signal comes in the form of a "request" flag which marks a tensor as <b>allowed to
     *  be updated by its gradient</b>.<br>
     *  <br>
     * @param applyRequested The truth value determining if the application of the gradient of this tensor is requested.
     * @return This very tensor instance in order to enable method chaining.
     */
    Tensor<V> setGradientApplyRequested( boolean applyRequested );

    /*==================================================================================================================
    |
    |       §(3) : COMPONENT SYSTEM
    |   --------------------------------
    */

    /**
     *  Important : Components of type {@link Tensor} are simply gradients!
     *  Currently, this method is used only to catch illegal arguments which
     *  is for example the case when trying to attach a gradient with a different shape...
     *  (Otherwise the gradient tensor "does not mind" an owner change...)
     */
    @Override
    default boolean update( OwnerChangeRequest<Tensor<V>> changeRequest ) {
        if ( changeRequest.type() == IsBeing.ADDED ) {
            if (
                changeRequest.getNewOwner().shape().hashCode() != this.shape().hashCode() ||
                Arrays.hashCode(changeRequest.getNewOwner().getNDConf().shape()) != Arrays.hashCode( getNDConf().shape() )
            ) {
                throw new IllegalArgumentException(
                        "Trying to attach a tensor as gradient component to a tensor with different shape."
                );
            }
            // If a tensor becomes a gradient, we need to make sure that it does not get deleted.
            this.getMut().setIsIntermediate( false ); // So we mark it as non-intermediate.
        }
        changeRequest.executeChange(); // This can be an 'add', 'remove' or 'transfer' of this component!
        // If the change request type is set to "REPLACED" then
        // this is means that this tensor is a gradient that is being
        // transferred to another tensor to serve as gradient...
        // No update task needs to occur. (This might change in the future...)
        return true;
    }

    /*==================================================================================================================
    |
    |       §(4) : PROPERTIES :
    |   ---------------------------------------
    */

    /**
     *  The version number is tracking how often this tensor has been mutated.
     *  This is especially useful for checking the correcting of auto-grad!
     */
    int getVersion();

    /**
     *  This method returns the {@link DataType} instance of this {@link Tensor}, which is
     *  a wrapper object for the actual type class representing the value items stored inside
     *  the underlying data array of this tensor.
     *
     * @return The {@link DataType} instance of this {@link Tensor} storing important type information.
     */
    DataType<V> getDataType();

    /**
     *  The {@link Class} returned by this method is the representative {@link Class} of the
     *  value items of a concrete {@link AbstractNda} but not necessarily the actual {@link Class} of
     *  a given value item, this is especially true for numeric types, which are represented by
     *  implementations of the {@link NumericType} interface.                                        <br>
     *  For example in the case of a tensor of type {@link Double}, this method would
     *  return {@link neureka.dtype.custom.F64} which is the representative class of {@link Double}. <br>
     *  Calling the {@link #getItemType()} method instead of this method would return the actual value
     *  type class, namely: {@link Double}.
     *
     * @return The representative type class of individual value items within this concrete {@link AbstractNda}
     *         extension instance which might also be subclasses of the {@link NumericType} interface
     *         to model unsigned types or other JVM foreign numeric concepts.
     */
    Class<?> getRepresentativeItemClass();

    /*==================================================================================================================
    |
    |       §(5) : OBJECT STATE MODIFICATION :
    |   ------------------------------------------
    */

    /** {@inheritDoc} */
    @Override
    MutateTensor<V> getMut();

    /** {@inheritDoc} */
    @Override default MutateTensor<V> mut() { return getMut(); }

    /** {@inheritDoc} */
    @Override default Tensor<V> reshape( int... shape ) {
        return Neureka.get()
                .backend()
                .getAutogradFunction()
                .reshape()
                .with(Arg.Shape.of(shape))
                .call(this);
    }

    /** {@inheritDoc} */
    @Override
    default Tensor<V> permute( int... dims ) {
        return Neureka.get()
                .backend()
                .getAutogradFunction()
                .permute()
                .with(Arg.Indices.of(dims))
                .call(this);
    }

    /** {@inheritDoc} */
    @Override
    default Tensor<V> transpose( int dim1, int dim2 ) {
        // Transpose is based on permute, so we can just call permute with the correct arguments!
        int[] dims = new int[ this.rank() ];
        for ( int i = 0; i < dims.length; i++ ) dims[i] = i;
        dims[dim1] = dim2;
        dims[dim2] = dim1;
        return this.permute( dims );
    }

    /*==================================================================================================================
    |
    |       §(6) : ND-ITERATOR LOGIC :
    |   ---------------------------------------
    */

    // See Nda

    /*==================================================================================================================
    |
    |       §(7) : COMPONENT SPECIFIC :
    |   ---------------------------------------
    */

    /**
     * This method takes a {@link Device} and tries to migrate the contents of this {@link Tensor}
     * instance to that {@link Device}!
     *
     * @param device The {@link Device} which should host this {@link Tensor} as well as be added to its components list.
     * @return This very class to enable method chaining.
     */
    Tensor<V> to( Device<?> device );

    /**
     * @param deviceType A search key identifying the device onto which this tensor should be stored.
     * @return This very tensor instance in order to enable method chaining.
     */
    default Tensor<V> to( String deviceType ) { return this.to(Device.get(deviceType)); }

    /**
     *  Configures an {@link Optimizer} for this tensor based on the given {@link OptimizerFactory}
     *  which will be used to create a new {@link Optimizer} instance specific to this tensor.
     *  The {@link Optimizer} instance will be attached to this tensor as a component
     *  and then called to perform the actual optimization when the {@link #applyGradient()} method is called.
     *  <p>
     *  Here a simple example of how to use this method:
     *  <pre>{@code
     *  var t = Tsr.of( 1.0, 2.0, 3.0 ).set( Optimizer.ADAM );
     *  }</pre>
     *  <p>
     *  As you can see, the {@link Optimizer} interface exposes various types of popular
     *  optimization algorithm factories which can be used to quickly and conveniently create
     *  an {@link Optimizer} instance for a particular tensor.
     *
     * @param optimizerFactory The {@link OptimizerFactory} which will be used to create a new {@link Optimizer} instance.
     * @return This tensor instance to allow for method chaining.
     */
    default Tensor<V> set( OptimizerFactory optimizerFactory ) {
        this.set( optimizerFactory.create( (Tensor) this ) );
        return this;
    }

    /**
     *  Tensors which are used or produced by the autograd system will have a {@link GraphNode} component attached to them.
     *  This is because autograd requires recording a computation graph for back-prop traversal.
     *  If this tensor is part of a computation graph then this method
     *  will traverse an error backward in the recorded history towards tensors which require
     *  the accumulation of gradients.
     *
     * @param error A tensor which is back-propagated to gradients. Must match the size og this tensor.
     * @return This tensor, to allow for method chaining.
     */
    default Tensor<V> backward(Tensor<V> error ) {
        LogUtil.nullArgCheck(error, "error", Tensor.class, "Cannot back-propagate 'null'!");
        ((TensorImpl<V>)this)._backward( LazyRef.of( () -> error ) );
        return this;
    }

    /**
     *  Tensors which are used or produced by the autograd system will have a {@link GraphNode} component attached to them.
     *  This is because autograd requires recording a computation graph for back-prop traversal.
     *  If this tensor is part of a computation graph then this method
     *  will traverse an error backward in the recorded history towards tensors which require
     *  the accumulation of gradients.<br>
     *  <br>
     *  This method turns the given scalar value and
     *  turns it into a matching tensor ( with the same shape)
     *  which will then be back-propagated through the
     *  recorded computation graph.
     *
     * @param value A scalar which is back-propagated to gradients. Must match the size og this tensor.
     * @return The tensor, to allow for method chaining.
     */
    default Tensor<V> backward( double value ) {
        ((TensorImpl<V>)this)._backward( LazyRef.of( () -> Tensor.of( this.getItemType(), shape(), value )) );
        return this;
    }

    /**
     *  Use this to back-propagate an error signal of 1.0 through the recorded computation graph.
     *  Tensors which are used or produced by operations supporting the autograd system
     *  will have this graph defined by {@link GraphNode} components attached to them.
     *  This is because autograd requires recording a computation graph for back-prop traversal.
     *  If this tensor is part of a computation graph then this method
     *  will traverse an error backward in the recorded history towards tensors which require
     *  the accumulation of gradients. <br>
     *  <br>
     *  This method assumes that the user wants to back-propagate
     *  an error of "1" having the same shape as
     *  this tensor.
     *
     * @return This tensor to allow for method chaining.
     */
    default Tensor<V> backward() {
        backward( 1 ); // By default, we back-propagate an error signal of 1.
        return this;
    }

    /**
     * @return The gradient of this tensor which is internally stored as component.
     */
    default Optional<Tensor<V>> getGradient() { return this.find( Tensor.class ).map(t -> (Tensor<V>) t ); }

    /**
     *  This is a functionally identical alternative to the {@link #getGradient()} method.
     *
     * @return The gradient of this tensor which is internally stored as component.
     */
    default Optional<Tensor<V>> gradient() { return getGradient(); }

    /**
     *  If this tensor owns a gradient tensor as component, then it can be applied by this method. <br>
     *  "Applying" a gradient to a tensor simply means adding the values inside the gradient element-wise
     *  to the owning host tensor via an inline operation. <br>
     */
    default void applyGradient(){
        /*
           If the tensor has a JITProp component then it will trigger the continuation of the back-propagation which
           has been put on hold by saving the pending graph nodes inside the component. <br>
           This is because the gradient most likely has not yet been fully calculated.
         */
        this.find( JITProp.class ).ifPresent( JITProp::execute );
        // Afterwards the JITProp component is not needed anymore! So we remove it.
        this.remove( JITProp.class );
        // Now the gradient can be applied (Gradients are also tensors, which is why we provide its class as key).
        this.find( Tensor.class ).ifPresent(g -> {
                // If an optimizer is present then we also optimize the gradient first!
                g = this.find( Optimizer.class ).map( o -> o.optimize( this ) ).orElse( g );
                // And then we remove the gradient because it is no longer needed.
                this.remove( Tensor.class );
                // We are now ready to apply the gradient to the tensor. This is an inline operation!
                // Therefore, we need to turn off the inline operation safety net:
                boolean inlineSafety = Neureka.get().settings().autograd().isPreventingInlineOperations();
                if ( inlineSafety ) Neureka.get().settings().autograd().setIsPreventingInlineOperations( false );
                // INLINE OPERATION :
                Neureka.get().backend().getFunction().plusAssign().call( this, g ); //-> Finally applying the gradient!
                // INLINE END ! -> We can now revert to the previous setting:
                if ( inlineSafety ) Neureka.get().settings().autograd().setIsPreventingInlineOperations( true );
            }
        );
    }

    /**
     * @return The device on which this tensor is stored or {@link CPU} if it is not outsourced.
     */
    default Device<V> getDevice() {
        Device device = this.get( Device.class );
        if ( device == null )
            if ( !this.isDeleted() && mut().getData() != null )
                return mut().getData().owner();
            else
                return (Device<V>) CPU.get();
        else
            return device;
    }

    /**
     * @return The graph node optional of the computation graph to which this tensor belongs
     *         or an empty optional if not part of a graph.
     */
    default Optional<GraphNode<V>> getGraphNode() { return find( GraphNode.class ).map( g-> (GraphNode<V>) g ); }

    /**
     *  This is a functionally identical alternative to {@link #getGraphNode()}.
     *
     * @return The graph node optional of the computation graph to which this tensor belongs
     *         or an empty optional if not part of a graph.
     */
    default Optional<GraphNode<V>> graphNode() { return getGraphNode(); }

    /**
     * @return An instance of the {@link NDFrame} component if present.
     */
    default Optional<NDFrame<V>> getFrame() { return (Optional<NDFrame<V>>) ((Optional)find( NDFrame.class )); }

    /**
     *  This is a functionally identical alternative to {@link #getFrame()}.
     *
     * @return An instance of the {@link NDFrame} component if present.
     */
    default Optional<NDFrame<V>> frame() { return getFrame(); }

    /**
     *  <b>This method returns a new tensor detached from any underlying computation-graph
     *  or simply does nothing if no graph is present.</b> <br>
     *  Nodes within a computation graph are instances of the "{@link GraphNode}" class which are also
     *  simple components of the tensors they represent in the graph. <br>
     *  Therefore, a "detached" clone of this tensor is
     *  simply a tensor without a {@link GraphNode} component.
     *
     * @return This very instance in order to allow for a more streamline usage of this method.
     */
    default Tensor<V> detached() {
        if ( this.has( GraphNode.class ) )
            return this.shallowCopy().remove( GraphNode.class );
        return this;
    }

    /** {@inheritDoc} */
    @Override
    Tensor<V> withLabel(String label );

    /** {@inheritDoc} */
    @Override
    Tensor<V> withLabels(String[]... labels );

    /** {@inheritDoc} */
    @Override
    Tensor<V> withLabels(List<List<Object>> labels );

    /** {@inheritDoc} */
    @Override
    Tensor<V> withLabels(Map<Object, List<Object>> labels );


    /*==================================================================================================================
    |
    |       §(8) : (OVERLOADABLE) OPERATORS & OPERATIONS :
    |   -----------------------------------------------------
    |       ...for more context see package 'math'...
    |*/

    /**
     *  This method compares the passed class with the underlying data-type of this NDArray.
     *  If the data-type of this NDArray is equivalent to the passed class then the returned
     *  boolean will be true, otherwise the method returns false.
     *
     * @param typeClass The class which ought to be compared to the underlying data-type of this NDArray.
     * @return The truth value of the question: Does this NDArray implementation hold the data of the passed type?
     */
    boolean is( Class<?> typeClass );

    /**
     *  This method will produce the addition of
     *  two tensors with the same rank (or two ranks which can be made compatible with padding ones),
     *  where the left operand is this {@link Tensor}
     *  instance and the right operand is the tensor passed to the method.
     *  If the shapes of both of the involved tensors is identical then
     *  the result will be a regular element-wise addition.
     *  Otherwise, the method will also be able to perform broadcasting, however only if
     *  for every pair of shape dimensions the following is true:
     *  Either the dimensions have the same size or one of them has size 1. <br>
     *  Here is an example of 2 matching shapes: (1, 4, 1) and (3, 4, 1)       <br>
     *  And here is an example of a mismatch: (2, 4, 1) and (3, 4, 1)         <br>
     *
     * @param other The right operand of the addition.
     * @return The sum of this instance as the left and the passed {@link Tensor} instance as right operand.
     */
    default Tensor<V> plus(Tensor<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tensor.class, "Cannot add 'null' to a tensor!");
        return Neureka.get().backend().getAutogradFunction().plus().call( this, other );
    }

    /**
     *  This method will create a new {@link Tensor}
     *  with the provided double scalar added to all elements of this {@link Tensor}.
     * <p>
     *  The shapes of this tensor is irrelevant as the provided value will
     *  simply be broadcast to any possible shape.
     *
     * @param value The right operand of the addition.
     * @return The sum between this instance as the left and the passed double as right operand.
     */
    default Tensor<V> plus(V value ) { return plus( ofAny( itemType(), this.shape(), value ) ); }

    /**
     *  Performs subtraction on
     *  two tensors with the same rank (or two ranks which can be made compatible with padding ones),
     *  where the left operand is this {@link Tensor}
     *  instance and the right operand is the tensor passed to the method.
     *  If the shapes of both of the involved tensors are identical then
     *  the result will be a regular element-wise subtraction.
     *  Otherwise, the method will also be able to perform broadcasting, however only if
     *  for every pair of shape dimensions the following is true:
     *  Either the dimensions have the same size or one of them has size 1.    <br>
     *  Here is an example of 2 matching shapes: (1, 4, 1) and (3, 4, 1)       <br>
     *  And here is an example of a mismatch: (2, 4, 1) and (3, 4, 1)          <br>
     *
     * @param other The right operand of the subtraction.
     * @return The difference between this instance as the left and the passed {@link Tensor} instance as right operand.
     */
    default Tensor<V> minus(Tensor<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tensor.class, "Cannot subtract 'null' from a tensor!");
        return Neureka.get().backend().getAutogradFunction().minus().call( this, other );
    }

    /**
     *  This method will create a new {@link Tensor}
     *  with the provided item subtracted from all elements of this {@link Tensor}.
     * <p>
     *  The shapes of this tensor is irrelevant as the provided item will
     *  simply be broadcast to all items od this tensor, irrespective of any shape.
     *
     * @param other The right operand of the subtraction, which is an item of the same type as this tensor.
     * @return The difference between this instance as the left and the passed item as right operand.
     */
    default Tensor<V> minus(V other ) {
        LogUtil.nullArgCheck(other, "other", this.getItemType(), "Cannot subtract 'null' from a tensor!");
        return minus(
                    of( this.getDataType().getItemTypeClass() )
                        .withShape(this.getNDConf().shape())
                        .all(other)
                );
    }

    /**
     * @return A clone of this tensor where the signs of all elements are flipped.
     */
    default Tensor<V> negative() { return Neureka.get().backend().getAutogradFunction().neg().call( this ); }

    /**
     *  Creates and returns a new {@link Tensor} instance which is a transposed twin of this instance.<br>
     *  This is a shorter alternative to the functionally identical {@link #getT()} method.
     *
     * @return A new transposed tensor with the same underlying {@link Data} as this tensor.
     */
    default Tensor<V> T() {
        if ( this.rank() == 1 ) return this;
        else if ( this.rank() == 2 )
            return Neureka.get().backend().getAutogradFunction().transpose2D().call(this);

        StringBuilder operation = new StringBuilder();
        for ( int i = rank() - 1; i >= 0; i-- ) operation.append( i ).append( i == 0 ? "" : ", " );
        operation = new StringBuilder( "[" + operation + "]:(I[ 0 ])" );
        return Function.of( operation.toString(), true ).call( this );
    }

    /**
     *  A method which returns a new {@link Tensor} instance which is a transposed twin of this instance.<br>
     *  This is an alternative to the functionally identical {@link #T()} method.
     *
     * @return A new transposed tensor with the same underlying {@link Data} as this tensor.
     */
    default Tensor<V> getT() { return this.T(); } // Transposed

    /**
     *  Calculate the mean value of all values
     *  within this tensor and returns it
     *  in the form of a scalar tensor. <br>
     *  This operation supports autograd.
     *
     * @return A scalar tensor which wraps the mean value of all values of this tensor.
     */
    default Tensor<V> mean() {
        Functions functions = Neureka.get().backend().getAutogradFunction();
        Tensor<V> sum = this.sum();
        Tensor<V> result = functions.div().call( sum, of( this.getItemType(), Shape.of( 1 ), this.size() ) );
        if ( sum != this ) sum.getMut().delete(); // This is a temporary tensor which is not needed anymore! (not even for back propagation)
        return result;
    }

    /**
     *  Calculate the sum value of all values
     *  within this tensor and returns it
     *  in the form of a scalar tensor. <br>
     *  This operation supports autograd.
     *
     * @return A scalar tensor which wraps the sum of all values of this tensor.
     */
    default Tensor<V> sum() {
        Functions functions = Neureka.get().backend().getAutogradFunction();
        Tensor<V> sum = functions.sum().call( this );
        if ( sum == null )
            throw new IllegalStateException(
                    "Failed to calculate sum using function! Shape: "+
                    Arrays.toString(this.getNDConf().shape())
                );
        return sum;
    }

    /**
     *  Calculate the sum value of all values
     *  within this tensor along the specified axis and returns it
     *  in the form of a tensor. <br>
     *  For example, if this tensor has a shape of (2, 3, 4) and the axis is 1,
     *  then the result will be a tensor with a shape of (2, 1, 4) because the
     *  sum of all values along the axis 1 is a single value for each of the two
     *  first dimensions. <br>
     *  This operation supports autograd.
     *
     * @param axis The axis along which the sum should be calculated.
     * @return A tensor which wraps the sum of all values of this tensor along the specified axis.
     */
    default Tensor<V> sum(int axis ) {
        int toBeReduced = this.shape(axis);
        Tensor<V> current = this.slice().axis(axis).at(0).get();
        for ( int i = 0; i < toBeReduced; i++ ) {
            if ( i > 0 )
                current = this.slice().axis(axis).at(i).get().plus(current);
        }
        return current;
    }

    /**
     *  Calculate the sum value of all values
     *  within this tensor along the specified axes and returns it
     *  in the form of a tensor. <br>
     *  For example, if this tensor has a shape of (2, 3, 4) and the axes are 1 and 2,
     *  then the result will be a tensor with a shape of (2, 1, 1) because the
     *  sum of all values along the axis 1 and 2 is a single value for each of the two
     *  first dimensions. <br>
     *  This operation supports autograd.
     *
     * @param axes The axes along which the sum should be calculated.
     * @return A tensor which wraps the sum of all values of this tensor along the specified axes.
     */
    default Tensor<V> sum(int... axes ) {
        Tensor<V> current = this;
        for ( int axis : axes )
            current = current.sum( axis );

        return current;
    }

    /**
     *  Calculate the min value of all values
     *  within this tensor and returns it
     *  in the form of a scalar tensor. <br>
     *  This operation supports autograd.
     *
     * @return A scalar tensor which wraps the smallest of all values of this tensor.
     */
    default Tensor<V> min() {
        Functions functions = Neureka.get().backend().getAutogradFunction();
        Tensor<V> min = functions.min().call( this );
        if ( min == null )
            throw new IllegalStateException(
                "Failed to calculate min using min function! Shape: "+
                Arrays.toString(this.getNDConf().shape())
            );
        return min;
    }

    /**
     *  Calculate the max value of all values
     *  within this tensor and returns it
     *  in the form of a scalar tensor. <br>
     *  This operation supports autograd.
     *
     * @return A scalar tensor which wraps the largest of all values of this tensor.
     */
    default Tensor<V> max() {
        Functions functions = Neureka.get().backend().getAutogradFunction();
        Tensor<V> max = functions.max().call( this );
        if ( max == null )
            throw new IllegalStateException(
                "Failed to calculate max using max function! Shape: "+
                Arrays.toString(this.getNDConf().shape())
            );
        return max;
    }

    /**
     *  This method performs a convolutional based dot product between the last dimension of this tensor
     *  and the first dimension of the passed tensor.
     *
     * @param other The tensor which is the right part of the dot product operation.
     * @return A new tensor which is the dot product of this tensor and the passed one.
     */
    default Tensor<V> convDot(Tensor<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tensor.class);
        Tensor<V> a = this;
        int[][] fitter = TensorImpl.makeFit( a.getNDConf().shape(), other.getNDConf().shape() );
        boolean doReshape = false;
        for ( int i = 0; i < fitter[ 0 ].length && !doReshape; i++ ) if ( fitter[ 0 ][ i ] != i ) doReshape = true;
        for ( int i = 0; i < fitter[ 1 ].length && !doReshape; i++ ) if ( fitter[ 1 ][ i ] != i ) doReshape = true;
        if ( doReshape ) {
            a = Function.of( NDUtil.shapeString( fitter[ 0 ] ) + ":(I[ 0 ])" ).call( a );
            other = Function.of( NDUtil.shapeString( fitter[ 1 ] ) + ":(I[ 0 ])" ).call( other );
        }
        return Neureka.get()
                .backend()
                .getAutogradFunction()
                .conv()
                .call( a, other )
                .dimtrim();
    }

    /**
     *  Performs a dot product between the last dimension of this tensor
     *  and the first dimension of the provided tensor.
     *  However, currently this method can only handle matrices which means
     *  that it is functionally completely identical to the {@link #matMul(Tensor)} method.
     *
     * @param other The tensor which is the right part of the dot product operation.
     * @return A new tensor which is the dot product of this tensor and the passed one.
     */
    default Tensor<V> dot(Tensor<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tensor.class, "Cannot perform dot operation when second operand is 'null'!");
        return Neureka.get().backend().getAutogradFunction().dot().call( this, other );
    }

    /**
     *  This will produce the matrix product of
     *  two tensors with rank 2 (matrices), where the left operand is this {@link Tensor}
     *  instance and the right operand is the argument passed to the method.
     *
     * @param other The right operand of the matrix multiplication.
     * @return The matrix product of this instance as the left and the passed {@link Tensor} instance as right operand.
     */
    default Tensor<V> matMul(Tensor<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tensor.class, "Cannot perform matrix multiplication operation when second operand is 'null'!");
        if ( this.rank() != 2 || other.rank() != 2 )
            throw new IllegalArgumentException(
                    "Cannot perform matrix multiplication for tensors whose ranks are not both 2!\n" +
                    "Encountered ranks: " + this.rank() + ", " + other.rank() + ";"
                );

        return Neureka.get().backend().getAutogradFunction().matMul().call( this, other );
    }

    /**
     * This method performs convolution between this tensor and the one passed as argument.
     * The convolution is performed by the {@link Function} which is registered under the name "conv".
     * @param other The tensor which is the right operand of the convolutional operation.
     * @return A new tensor which is the result of the convolutional operation.
     */
    default Tensor<V> conv(Tensor<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tensor.class, "Cannot perform convolution operation when second operand is 'null'!");
        return Neureka.get().backend().getAutogradFunction().conv().call( this, other );
    }

    /**
     *  This creates a new tensor with the same underlying {@link Data} and whose shape is trimmed.
     *  A trimmed shape is simply a shape without preceding and trailing ones. <br>
     *  For example the shape (1x4x1x2x1) would be trimmed to (4x1x2).
     *  The underlying operation does not perform a removal of redundant ones all together.
     *  Only ones at the start and the beginning will be removed.
     *  A scalar tensor will not be affected by this operation.
     *
     * @return A tensor with the same underlying data but possibly trimmed shape without preceding or trailing ones.
     */
    default Tensor<V> dimtrim() { return Neureka.get().backend().getAutogradFunction().dimTrim().call( this ); }

    /**
     *  This method name translates to the "in" keyword in Groovy!
     *  The same is true for the "contains" method in Kotlin.
     *  Both methods do the exact same thing, however they exist
     *  for better language support.
     *
     * @param other The tensor which will be checked.
     * @return The answer to the following question: Is the data of the provided tensor a subset of the data of this tensor?
     */
    boolean isCase( Tensor<V> other );

    /**
     *  This method name translates to the "in" keyword in Kotlin!
     *  The same is true for the "isCase" method in Groovy.
     *  Both methods do the exact same thing, however they exist
     *  for better language support.
     *
     * @param other The tensor which will be checked.
     * @return The answer to the following question: Is the data of the provided tensor a subset of the data of this tensor?
     */
    default boolean contains( Tensor<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tensor.class, "Cannot perform 'contains' operation when second operand is 'null'!");
        return this.isCase( other );
    }

    /**
     *  This method is synonymous to the {@link #times(Tensor)} method.
     *  Both of which will produce the product of
     *  two tensors with the same rank (or two ranks which can be made compatible with padding ones),
     *  where the left operand is this {@link Tensor}
     *  instance and the right operand is the tensor passed to the method.
     *  If the shapes of both of the involved tensors is identical then
     *  the result will be a regular element-wise product.
     *  Otherwise, the method will also be able to perform broadcasting, however only if
     *  for every pair of shape dimensions the following is true:
     *  Either the dimensions have the same size or one of them has size 1. <br>
     *  Here is an example of 2 matching shapes: (1, 4, 1) and (3, 4, 1)       <br>
     *  And here is an example of a mismatch: (2, 4, 1) and (3, 4, 1)         <br>
     *
     * @param other The right operand of the multiplication.
     * @return The product of this instance as the left and the passed {@link Tensor} instance as right operand.
     */
    default Tensor<V> multiply(Tensor<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tensor.class, "Cannot multiply 'null' with a tensor!");
        return Neureka.get().backend().getAutogradFunction().mul().call( this, other );
    }

    /**
     * @param other The value which should be broadcast to all elements of a clone of this tensor.
     * @return A new tensor where all elements are multiplied by the provided value.
     */
    default Tensor<V> multiply(V other ) {
        LogUtil.nullArgCheck(other, "other", this.getItemType(), "Cannot multiply 'null' with a tensor!");
        return multiply(
                of( this.getDataType().getItemTypeClass() )
                        .withShape( this.getNDConf().shape() )
                        .all( other )
        );
    }

    /**
     *  This is a functionally identical synonym to the {@link #multiply(Tensor)} method.
     *  Both of which will produce the product of
     *  two tensors with the same rank (or two ranks which can be made compatible with padding ones),
     *  where the left operand is this {@link Tensor}
     *  instance and the right operand is the tensor passed to the method.
     *  If the shapes of both of the involved tensors is identical then
     *  the result will be a regular element-wise product.
     *  Otherwise, the method will also be able to perform broadcasting, however only if
     *  for every pair of shape dimensions the following is true:
     *  Either the dimensions have the same size or one of them has size 1. <br>
     *  Here is an example of 2 matching shapes: (1, 4, 1) and (3, 4, 1)       <br>
     *  And here is an example of a mismatch: (2, 4, 1) and (3, 4, 1)         <br>
     *
     * @param other The right operand of the multiplication.
     * @return The product of this instance as the left and the passed {@link Tensor} instance as right operand.
     */
    default Tensor<V> times(Tensor<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tensor.class, "Cannot multiply 'null' with a tensor!");
        return multiply( other );
    }

    /**
     * @param other The value which should be broadcast to all elements of a clone of this tensor.
     * @return A new tensor where all elements are multiplied by the provided value.
     */
    default Tensor<V> times(V other ) {
        LogUtil.nullArgCheck(other, "other", getItemType(), "Cannot multiply 'null' with a tensor!");
        return multiply( other );
    }

    /**
     * @param value The value which should be broadcast to all elements of a clone of this tensor.
     * @return A new tensor where all elements are multiplied by the provided value.
     */
    default Tensor<V> multiply(double value ) {
        Tensor<V> other = of( getItemType(), shape(), value );
        Tensor<V> result = multiply( other );
        if ( !other.graphNode().map(GraphNode::isUsedAsDerivative).orElse(false) )
            other.mut().delete();
        return result;
    }

    /**
     *  This method will produce the quotient of
     *  two tensors with the same rank (or two ranks which can be made compatible with padding ones),
     *  where the left operand is this {@link Tensor}
     *  instance and the right operand is the tensor passed to the method.
     *  If the shapes of both of the involved tensors are identical then
     *  the result will be a regular element-wise division.
     *  Otherwise, the method will also be able to perform broadcasting, however only if
     *  for every pair of shape dimensions the following is true:
     *  Either the dimensions have the same size or one of them has size 1. <br>
     *  Here is an example of 2 matching shapes: (1, 4, 1) and (3, 4, 1)       <br>
     *  And here is an example of a mismatch: (2, 4, 1) and (3, 4, 1)         <br>
     *
     * @param other The right operand of the division.
     * @return The quotient of this instance as the left and the passed {@link Tensor} instance as right operand.
     */
    default Tensor<V> div(Tensor<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tensor.class, "Cannot divide a tensor by 'null' (In any sense of the word)!");
        return Neureka.get().backend().getAutogradFunction().div().call( this, other );
    }
    default Tensor<V> div(V value ) { return div( of( getItemType(), shape(), value ) ); }

    /**
     *  Produces the modulus of
     *  two tensors with the same rank (or two ranks which can be made compatible with padding ones),
     *  where the left operand is this {@link Tensor}
     *  instance and the right operand is the tensor passed to the method.
     *  If the shapes of these 2 tensors are identical then
     *  the result will be a regular element-wise modulo operation.
     *  Otherwise, the method will also be able to perform broadcasting, however only if
     *  for every pair of shape dimensions the following is true:
     *  Either the dimensions have the same size or one of them has size 1. <br>
     *  Here is an example of 2 matching shapes: (1, 4, 1) and (3, 4, 1)       <br>
     *  And here is an example of a mismatch: (2, 4, 1) and (3, 4, 1)         <br>
     *
     * @param other The right operand of the modulo operation.
     * @return The modulus of this instance as the left and the passed {@link Tensor} instance as right operand.
     */
    default Tensor<V> mod(Tensor<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tensor.class, "Cannot perform tensor modulo 'null'!");
        return Neureka.get().backend().getAutogradFunction().mod().call( this, other );
    }

    /**
     * @param other The value which should be broadcast to all elements of a clone of this tensor.
     * @return A new tensor where the modulo operation is applied to all
     *          elements using the provided int as right operand.
     */
    default Tensor<V> mod(int other ) { return mod(of(getItemType(), shape(), other)); }

    /**
     *  This method is synonymous to the {@link #mod(int)} method.
     */
    default Tensor<V> rem(int other ) { return this.mod(other); }

    /**
     *  This will produce the power of
     *  two tensors with the same rank (or two ranks which can be made compatible with padding ones),
     *  where the left operand is this {@link Tensor}
     *  instance and the right operand is the tensor passed to the method.
     *  If the shapes of the involved tensors are identical then
     *  the result will be a regular element-wise exponentiation.
     *  Otherwise, the method will also be able to perform broadcasting, however only if
     *  for every pair of shape dimensions the following is true:
     *  Either the dimensions have the same size or one of them has size 1. <br>
     *  Here is an example of 2 matching shapes: (1, 4, 1) and (3, 4, 1)       <br>
     *  And here is an example of a mismatch: (2, 4, 1) and (3, 4, 1)         <br>
     *
     * @param other The right operand, also known as exponent, of the exponentiation.
     * @return The power of this instance as the left and the passed {@link Tensor} instance as right operand.
     */
    default Tensor<V> power(Tensor<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tensor.class, "Cannot raise a tensor to the power of 'null'!");
        return Neureka.get().backend().getAutogradFunction().pow().call( this, other );
    }

    /**
     *  Raises all items of this tensor to the power of the provided value.
     *  The returned tensor is a new instance which will have the same shape as this tensor.
     * 
     * @param value The value which should be used to raise all items of this tensor to the power of.
     * @return A new tensor where all items are raised to the power of the provided value.
     */
    default Tensor<V> power(V value ) {
        LogUtil.nullArgCheck(value, "value", getItemType(), "Cannot raise a tensor to the power of 'null'!");
        return power( ofAny( this.itemType(), this.shape(), value ) );
    }
    
    /**
     *  This method is a functionally identical synonym to the {@link #power(Tensor)} method.
     */
    default Tensor<V> xor(Tensor<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tensor.class, "Cannot raise a tensor to the power of 'null'!");
        return Neureka.get().backend().getAutogradFunction().pow().call( this, other );
    }

    /**
     *  This method is a functionally identical synonym to the {@link #power(Tensor)} method.
     */
    default Tensor<V> xor(double value ) { return xor( ofAny( this.itemType(), this.shape(), value ) ); }

    /**
     *  This method is a functionally identical to the following alternatives:
     *  <pre>{@code
     *      // Pre-instantiated:
     *      var out1 = Neureka.get().backend().getAutogradFunction().sigmoid().call( myTensor );
     *      // Dynamically parsed and instantiated:
     *      var out2 = Function.of("sig(I[0])").call(myTensor);
     *  }</pre>
     *
     * @return A new tensor whose items are the result of the <b>sigmoid function</b> applied to the items of this tensor.
     */
    default Tensor<V> sig() { return Neureka.get().backend().getAutogradFunction().sigmoid().call( this ); }

    /**
     *  This method is a functionally identical to the following alternatives:
     *  <pre>{@code
     *      // Pre-instantiated:
     *      var out1 = Neureka.get().backend().getAutogradFunction().tanh().call( myTensor );
     *      // Dynamically parsed and instantiated:
     *      var out2 = Function.of("tanh(I[0])").call(myTensor);
     *  }</pre>
     *
     * @return A new tensor whose items are the result of the <b>tanh function</b> applied to the items of this tensor.
     */
    default Tensor<V> tanh() { return Neureka.get().backend().getAutogradFunction().tanh().call( this ); }

    /**
     *  This method is a functionally identical to the following alternatives:
     *  <pre>{@code
     *      // Pre-instantiated:
     *      var out1 = Neureka.get().backend().getAutogradFunction().relu().call( myTensor );
     *      // Dynamically parsed and instantiated:
     *      var out2 = Function.of("relu(I[0])").call(myTensor);
     *  }</pre>
     *
     * @return A new tensor whose items are the result of the <b>relu function</b> applied to the items of this tensor.
     */
    default Tensor<V> relu() { return Neureka.get().backend().getAutogradFunction().relu().call( this ); }

    /**
     *  This method is a functionally identical to the following alternatives:
     *  <pre>{@code
     *      // Pre-instantiated:
     *      var out1 = Neureka.get().backend().getAutogradFunction().sin().call( myTensor );
     *      // Dynamically parsed and instantiated:
     *      var out2 = Function.of("sin(I[0])").call(myTensor);
     *  }</pre>
     *
     * @return A new tensor whose items are the result of the <b>sin function</b> applied to the items of this tensor.
     */
    default Tensor<V> sin() { return Neureka.get().backend().getAutogradFunction().sin().call( this ); }

    /**
     *  This method is a functionally identical to the following alternatives:
     *  <pre>{@code
     *      // Pre-instantiated:
     *      var out1 = Neureka.get().backend().getAutogradFunction().cos().call( myTensor );
     *      // Dynamically parsed and instantiated:
     *      var out2 = Function.of("cos(I[0])").call(myTensor);
     *  }</pre>
     *
     * @return A new tensor whose items are the result of the <b>cos function</b> applied to the items of this tensor.
     */
    default Tensor<V> cos() { return Neureka.get().backend().getAutogradFunction().cos().call( this ); }

    /**
     *  This method is a functionally identical to the following alternatives:
     *  <pre>{@code
     *      // Pre-instantiated:
     *      var out1 = Neureka.get().backend().getAutogradFunction().ln().call( myTensor );
     *      // Dynamically parsed and instantiated:
     *      var out2 = Function.of("ln(I[0])").call(myTensor);
     *  }</pre>
     *
     * @return A new tensor whose items are the result of the <b>ln function</b> applied to the items of this tensor.
     */
    default Tensor<V> ln() { return Neureka.get().backend().getAutogradFunction().ln().call( this ); }

    /**
     *  This method is a functionally identical to the following alternatives:
     *  <pre>{@code
     *      // Pre-instantiated:
     *      var out1 = Neureka.get().backend().getAutogradFunction().softplus().call( myTensor );
     *      // Dynamically parsed and instantiated:
     *      var out2 = Function.of("softplus(I[0])").call(myTensor);
     *  }</pre>
     *
     * @return A new tensor whose items are the result of the <b>softplus function</b> applied to the items of this tensor.
     */
    default Tensor<V> softplus() { return Neureka.get().backend().getAutogradFunction().softplus().call( this ); }

    /**
     *  This method is a functionally identical to the following alternatives:
     *  <pre>{@code
     *      // Pre-instantiated:
     *      var out1 = Neureka.get().backend().getAutogradFunction().exp().call( myTensor );
     *      // Dynamically parsed and instantiated:
     *      var out2 = Function.of("exp(I[0])").call(myTensor);
     *  }</pre>
     *
     * @return A new tensor whose items are the result of the <b>exp function</b> applied to the items of this tensor.
     */
    default Tensor<V> exp() { return Neureka.get().backend().getAutogradFunction().exp().call( this ); }

    /**
     *  This method is a functionally identical to the following alternatives:
     *  <pre>{@code
     *      // Pre-instantiated:
     *      var out1 = Neureka.get().backend().getAutogradFunction().sqrt().call( myTensor );
     *      // Dynamically parsed and instantiated:
     *      var out2 = Function.of("sqrt(I[0])").call(myTensor);
     *  }</pre>
     *
     * @return A new tensor whose items are the result of the <b>sqrt function</b> applied to the items of this tensor.
     */
    default Tensor<V> sqrt() { return Neureka.get().backend().getAutogradFunction().sqrt().call( this ); }

    /**
     *  This method is a functionally identical to the following alternatives:
     *  <pre>{@code
     *      // Pre-instantiated:
     *      var out1 = Neureka.get().backend().getAutogradFunction().log10().call( myTensor );
     *      // Dynamically parsed and instantiated:
     *      var out2 = Function.of("log10(I[0])").call(myTensor);
     *  }</pre>
     *
     * @return A new tensor whose items are the result of the <b>log10 function</b> applied to the items of this tensor.
     */
    default Tensor<V> log10() { return Neureka.get().backend().getAutogradFunction().log10().call( this ); }

    /**
     *  This method is a functionally identical to the following alternatives:
     *  <pre>{@code
     *      // Pre-instantiated:
     *      var out1 = Neureka.get().backend().getAutogradFunction().cbrt().call( myTensor );
     *      // Dynamically parsed and instantiated:
     *      var out2 = Function.of("cbrt(I[0])").call(myTensor);
     *  }</pre>
     *
     * @return A new tensor whose items are the result of the <b>cbrt function</b> applied to the items of this tensor.
     */
    default Tensor<V> cbrt() { return Neureka.get().backend().getAutogradFunction().cbrt().call( this ); }

    /**
     *  This method is a functionally identical to the following alternatives:
     *  <pre>{@code
     *      // Pre-instantiated:
     *      var out1 = Neureka.get().backend().getAutogradFunction().abs().call( myTensor );
     *      // Dynamically parsed and instantiated:
     *      var out2 = Function.of("abs(I[0])").call(myTensor);
     *  }</pre>
     *
     * @return A new tensor whose items are the result of the <b>abs function</b> applied to the items of this tensor.
     */
    default Tensor<V> abs() { return Neureka.get().backend().getAutogradFunction().abs().call( this ); }

    /**
     *  This method is a functionally identical to the following alternatives:
     *  <pre>{@code
     *      // Pre-instantiated:
     *      var out1 = Neureka.get().backend().getAutogradFunction().neg().call( myTensor );
     *      // Dynamically parsed and instantiated:
     *      var out2 = Function.of("neg(I[0])").call(myTensor);
     *  }</pre>
     *
     * @return A new tensor whose items are the result of the <b>neg function</b> applied to the items of this tensor.
     */
    default Tensor<V> neg() { return Neureka.get().backend().getAutogradFunction().neg().call( this ); }

    /**
     * @return A new tensor whose items are the result of the <b>softmax function</b> applied to the items of this tensor.
     */
    default Tensor<V> softmax() {
        // Currently the softmax function is not implemented as Function instance, we simply calculate it using exp and div:
        return exp().div( exp().sum() );
    }

    /**
     * @return A new tensor whose items are the result of the <b>softmax function</b> applied to the items of this tensor.
     */
    default Tensor<V> softmax(int axis ) {
        // Currently the softmax function is not implemented as Function instance, we simply calculate it using exp and div:
        Tensor<V> exp = exp();
        return exp.div( exp.sum(axis) );
    }

    /**
     *  Calculates the softmax function along the specified axes. <br>
     *  For example, if this tensor has a shape of (2, 3, 4) and the axes 0 and 2 are chosen,
     *  then the result will be a tensor of the same size where all elements summed up alongside
     *  axis 0 and 2 would be 1.
     *  Ao calling {@code sum(0, 2)} would in this example be a tensor of shape of (1, 3, 1) where every item is 1. <br>
     *  This operation supports autograd.
     *
     * @param axes The axes along which the softmax function should be applied.
     * @return A new tensor whose items are the result of the <b>softmax function</b> applied to the items of this tensor.
     */
    default Tensor<V> softmax(int... axes ) {
        // Currently the softmax function is not implemented as Function instance, we simply calculate it using exp and div:
        return exp().div( exp().sum(axes) );
    }

    /**
     * @return A new tensor whose items are the result of the <b>sigmoid function</b> applied to the items of this tensor.
     */
    default Tensor<V> sigmoid() { return Neureka.get().backend().getAutogradFunction().sigmoid().call( this ); }


    /*==================================================================================================================
    |
    |       §(9) : SLICING, INDEXING & INJECTING :
    |   -----------------------------------------------------
    |       ...for more context see package 'ndim.config'...
    */

    /** {@inheritDoc} */
    @Override
    AxisOrGetTensor<V> slice();

    /** {@inheritDoc} */
    @Override default Tensor<V> concatAt(int axis, Nda<V> other, Nda<V>... ndArrays ) {
        String args = IntStream.range(0,ndArrays.length+2).mapToObj(i->"I["+ i +"]").collect(Collectors.joining(", "));
        Function concat = Function.of( "concat("+ args +")" );
        Tensor<V>[] allArgs = new Tensor[ndArrays.length+2];
        allArgs[0] = this;
        allArgs[1] = (Tensor<V>) other;
        System.arraycopy( ndArrays, 0, allArgs, 2, ndArrays.length );
        return concat.with(Arg.Axis.of(axis)).call( allArgs );
    }

    /** {@inheritDoc} */
    @Override default Tensor<V> concatAt(int axis, Nda<V> other ) {
        return Neureka.get()
                .backend()
                .getAutogradFunction()
                .concat()
                .with(Arg.Axis.of(axis))
                .call( this, (Tensor<V>) other );
    }


    /** {@inheritDoc} */
    @Override
    Tensor<V> getAt(int... indices );

    /** {@inheritDoc} */
    @Override
    default Tensor<V> getAt(Number i ) {
        return getAt( Collections.singletonList( getNDConf().indicesOfIndex( (i).intValue() ) ).toArray() );
    }

    /** {@inheritDoc} */
    @Override
    default Tensor<V> get(int... indices ) { return getAt( indices ); }

    /** {@inheritDoc} */
    @Override
    default Tensor<V> getAt(Object... args ) {
        List<Object> argsList = Arrays.asList( args );
        return getAt( argsList );
    }

    /** {@inheritDoc} */
    @Override
    default Tensor<V> get(Object... args ) { return getAt( args ); }

    /** {@inheritDoc} */
    @Override
    default Tensor<V> getAt(int i ) { return getAt( indicesOfIndex(i) ); }

    /** {@inheritDoc} */
    @Override
    default Tensor<V> get(int i ) { return getAt( i ); }

    /** {@inheritDoc} */
    @Override
    default Tensor<V> get(Number i ) { return getAt( i ); }

    /** {@inheritDoc} */
    @Override
    default Tensor<V> get(Object key ) { return getAt( key ); }

    /** {@inheritDoc} */
    @Override
    Tensor<V> getAt(Map<?,Integer> rangToSteps);

    /** {@inheritDoc} */
    @Override
    Tensor<V> getAt(List<?> key );

    /*==================================================================================================================
    |
    |       §(10) : Mapping :
    |   -----------------------------------------------------
    |       ...transformation and modification...
    */

    /** {@inheritDoc} */
    @Override default <T> Tensor<T> mapTo(
        Class<T> typeClass,
        java.util.function.Function<V,T> mapper
    ) {
        if ( this.isEmpty() )
            throw new IllegalArgumentException("Trying to map an empty tensor!");
        /*
           The provided lambda cannot be executed anywhere but the CPU (Note: Maybe we should consider Aparapi here)
           This is a problem if this tensor here lives somewhere other than the JVM.
           So, therefore, we invite it back home for dinner!
         */
        return CPU.get() // This little API will temporarily migrate this to the JVM.
                .borrow( (Tensor<Object>) this )
                .in( () -> {
                    Object data = getMut().getData().getOrNull();
                    DataConverter.ForTensor map = new DataConverter.ForTensor( this );
                    if ( data == null ) {
                        if ( this.isOutsourced() )
                            throw new IllegalStateException("Encountered an outsourced tensor! Only local tensors stored in RAM can be mapped.");
                        else
                            throw new IllegalStateException("Invalid tensor state encountered! Cannot map a tensor without data.");
                    }
                    Object newData;
                    String failMessage = "Conversion to type "+typeClass+" not yet supported.";
                    if ( Number.class.isAssignableFrom(typeClass) ) {
                        java.util.function.Function<Integer, Number> access;
                        if ( this.getItemType() == Integer.class ) {
                            int[] sourceData = getMut().getData().as(int[].class);
                            access = (i -> (Number) mapper.apply((V) Integer.valueOf(sourceData[i])));
                        } else if (this.getItemType() == Double.class) {
                            double[] sourceData = getMut().getData().as(double[].class);
                            access = (i -> (Number) mapper.apply((V) Double.valueOf(sourceData[i])));
                        } else if (this.getItemType() == Float.class) {
                            float[] sourceData = getMut().getData().as(float[].class);
                            access = (i -> (Number) mapper.apply((V) Float.valueOf(sourceData[i])));
                        } else if (this.getItemType() == Short.class) {
                            short[] sourceData = getMut().getData().as(short[].class);
                            access = (i -> (Number) mapper.apply((V) Short.valueOf(sourceData[i])));
                        } else if (this.getItemType() == Byte.class) {
                            byte[] sourceData = getMut().getData().as(byte[].class);
                            access = (i -> (Number) mapper.apply((V) Byte.valueOf(sourceData[i])));
                        } else
                            throw new IllegalArgumentException(failMessage);

                        if (typeClass == Double.class) newData = map.toDoubleArray(access);
                        else if ( typeClass == Integer.class ) newData = map.toIntArray(access);
                        else if ( typeClass == Long.class    ) newData = map.toLongArray(access);
                        else if ( typeClass == Byte.class    ) newData = map.toByteArray(access);
                        else if ( typeClass == Float.class   ) newData = map.toFloatArray(access);
                        else if ( typeClass == Short.class   ) newData = map.toShortArray(access);
                        else
                            throw new IllegalArgumentException(failMessage);
                    } else {
                        java.util.function.Function<Integer, Object> access = null;
                        if ( this.getItemType() == Integer.class ) {
                            int[] sourceData = getMut().getData().as(int[].class);
                            access = (i -> mapper.apply((V) Integer.valueOf(sourceData[i])));
                        } else if ( this.getItemType() == Double.class ) {
                            double[] sourceData = getMut().getData().as(double[].class);
                            access = (i -> mapper.apply((V) Double.valueOf(sourceData[i])));
                        } else if ( this.getItemType() == Float.class ) {
                            float[] sourceData = getMut().getData().as(float[].class);
                            access = (i -> mapper.apply((V) Float.valueOf(sourceData[i])));
                        } else if ( this.getItemType() == Short.class ) {
                            short[] sourceData = getMut().getData().as(short[].class);
                            access = (i -> mapper.apply((V) Short.valueOf(sourceData[i])));
                        } else if ( this.getItemType() == Byte.class ) {
                            byte[] sourceData = getMut().getData().as(byte[].class);
                            access = (i -> mapper.apply((V) Byte.valueOf(sourceData[i])));
                        } else if ( typeClass == itemType() ) {
                            Object[] sourceData = getMut().getData().as(Object[].class);
                            access = (i -> mapper.apply( (V) sourceData[i] ));
                        } else
                            throw new IllegalArgumentException(failMessage);

                        newData = map.toObjectArray(access);
                    }
                    return Tensor.of( typeClass, this.shape(), newData );
                });
    }

    /** {@inheritDoc} */
    @Override default Tensor<V> map(java.util.function.Function<V,V> mapper ) {
        return mapTo( this.getItemType(), mapper );
    }

    /**
     *  Turns this tensor into a {@link BufferedImage} based on the provided
     *  {@link Tensor.ImageType} formatting choice.
     *
     * @param type The type of format used to create the buffered image.
     * @return A {@link BufferedImage} populated with the contents of this tensor.
     */
    BufferedImage asImage( Tensor.ImageType type );

    /**
     * @param typeClass The class which is the target of the type conversion.
     * @param <T> The type parameter of the type that will be returned.
     * @return An instance of the supplied type class.
     */
    <T> T asType( Class<T> typeClass );

    default String toString( String conf ) {
        if ( this.isDeleted() ) return "deleted";
        else if ( this.isEmpty() ) return "empty";
        else if ( this.isUndefined() ) return "undefined";
        return NdaAsString.representing( this ).withConfig( conf ).toString();
    }

    /** {@inheritDoc} */
    @Override default String toString( NDPrintSettings config ) {
        if ( this.isDeleted() ) return "deleted";
        else if ( this.isEmpty() ) return "empty";
        else if ( this.isUndefined() ) return "undefined";
        return Nda.super.toString( config );
    }

    /** {@inheritDoc} */
    @Override default String toString( Consumer<NDPrintSettings> configurator ) {
        if ( this.isDeleted() ) return "deleted";
        return Nda.super.toString( configurator );
    }

    /** {@inheritDoc} */
    @Override
    Tensor<V> deepCopy();

    /** {@inheritDoc} */
    @Override default Tensor<V> shallowCopy() {
        if ( this.isEmpty() || this.isUndefined() ) return this; // Maybe throw an exception here...
        return slice().detached();
    }

    /**
     *  This is almost identical to the {@link Tensor#deepCopy()} method except that
     *  the returned tensor will have autograd support, meaning that the cloning
     *  will be part of the autograd computation graph, and backpropagation
     *  will traverse the cloned tensor as well.
     *
     * @return A deep clone of this tensor with autograd support.
     */
    Tensor<V> deepClone();

    /**
     * @return A shallow copy of this tensor with autograd support.
     */
    default Tensor<V> shallowClone() {
        if ( this.isEmpty() || this.isUndefined() ) return this; // Maybe throw an exception here...
        return slice().get();
    }

    /**
     * Use this enum as argument for the {@link Tensor#asImage(Tensor.ImageType)} method to
     * specify the type of image that should be returned.
     */
    enum ImageType
    {
        RGB_1INT(1, UI32.class, 1),
        ARGB_1INT(2, UI32.class, 1),
        ARGB_PRE_1INT(3, UI32.class, 1),
        BGR_1INT(4, UI32.class, 1),
        BGR_3BYTE(5, UI8.class, 3),
        ABGR_4BYTE(6, UI8.class, 4),
        ABGR_PRE_4BYTE(7, UI8.class, 4),
        RGB_565_USHORT(8, UI16.class, 1),
        RGB_555_USHORT(9, UI16.class, 1),
        GRAY_BYTE(0, UI8.class, 1),
        GRAY_USHORT(1, UI16.class, 1);

        public final int bufferType;
        public final DataType<?> dataType;
        public final int numberOfChannels;

        ImageType( int bufferType, Class<?> valueTypeClass, int numberOfChannels ) {
            this.bufferType = bufferType;
            this.dataType = DataType.of( valueTypeClass );
            this.numberOfChannels = numberOfChannels;
        }
    }

}
