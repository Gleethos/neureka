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

 __________
 \__    ___\
    |  |____ _ __
    | /  ___/ '___\
    | \___  \ |
     \/_____/_|      A long yet shallow class.

    This is the the core work-horse class of Neureka. The 'Tsr' class!
    It is a three-letter abbreviation of the word "Tensor"!

------------------------------------------------------------------------------------------------------------------------

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
import neureka.backend.api.ExecutionCall;
import neureka.calculus.Function;
import neureka.calculus.Functions;
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
import neureka.fluent.building.states.IterByOrIterFromOrAllTsr;
import neureka.fluent.building.states.WithShapeOrScalarOrVector;
import neureka.fluent.building.states.WithShapeOrScalarOrVectorOnDevice;
import neureka.fluent.slicing.states.AxisOrGetTsr;
import neureka.framing.NDFrame;
import neureka.framing.Relation;
import neureka.ndim.Filler;
import neureka.ndim.NDConstructor;
import neureka.ndim.NDUtil;
import neureka.ndim.config.NDConfiguration;
import neureka.optimization.Optimizer;
import neureka.view.NDPrintSettings;
import neureka.view.NdaAsString;

import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

/**
 *  {@link Tsr} is a 3 letter abbreviation of the word "tensor", a mathematical concept.
 *  A tensor is a type of multidimensional data-structure with certain transformation properties.
 *  Technically however, it is mostly a simple container / data-structure which can house data indexed by N dimensions.
 *  Therefore, it is often also described as a nd-array.
 *  Elements of a tensor are also mostly numeric.<br>
 *  This means that: <br>
 *  <i><b>...a tensor of rank 0 is a scalar, a tensor of rank 1 is a vector and a tensor of rank 2 is a matrix, etc...</b></i>
 *  <br><br>
 *  Consequently, tensors are a perfect fit for applying various operations on them.
 *  Such operations might be simple element-wise operations or more complex linear operations like
 *  the dot-product, matrix- or even tensor multiplications. <br>
 *  <br>
 * @param <V> The type parameter for the individual value items within this tensor.
 */
public interface Tsr<V> extends Nda<V>, Component<Tsr<V>>, ComponentOwner<Tsr<V>>
{
    /*==================================================================================================================
    |
    |       §(1) : CONSTRUCTION
    |   ---------------------------
    */

    /**
     *  This static factory method creates and return a completely empty and undefined tensor
     *  which is void of any contents and meaning.
     *  The use case for this would be to use the produced {@link Tsr}
     *  instance as a target for an inline operations which fills the instance with an actual value. <br>
     *  An example of this approach would be to call the {@link #putAt(List, Nda)} method with an empty list as key.
     *  This will be interpreted as an inline copy of the contents of the
     *  second parameter into this {@link Tsr} instance.
     *
     * @return A new and completely empty / unitialized {@link Tsr} instance.
     */
    static Tsr<Object> newInstance() { return new TsrImpl<>(); }

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
    static <T> Tsr<T> of( Tsr<T> a, char o, Tsr<T> b ) { return TsrImpl._of( a, String.valueOf(o), b ); }

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
    static <T> Tsr<T> of( Tsr<T> a, char o1, Tsr<T> b, char o2, Tsr<T> c ) {
        return TsrImpl._of( a, String.valueOf(o1), b, String.valueOf(o2), c );
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
    static <T> Tsr<T> of( String e1, Tsr<T> a, String e2 ) { return TsrImpl._of( e1, a, e2 ); }

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
    static <T> Tsr<T> of( String e1, Tsr<T> a, char o, Tsr<T> b, String e2 ) {
        return TsrImpl._of( e1, a, String.valueOf(o), b, e2 );
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
    static <T> Tsr<T> of(
            String e1, Tsr<T> a, String e2, Tsr<T> b, String e3, Tsr<T> c, String e4
    ) {
        return TsrImpl._of( e1, a, e2, b, e3, c, e4 );
    }


    /**
     *  This static {@link Tsr} factory method tries to interpret the provided
     *  arguments to create the instance the use might wants.
     *
     * @param args The arguments which ought to be interpreted.
     * @return The result of the interpretation in the form of a {@link Tsr} instance of typ {@link Object}.
     */
    static <T> Tsr<T> of( Object... args ) { return TsrImpl._of( args ); }

    /**
     *  This is a convenient factory method for creating {@link Tsr} instances for
     *  values of type {@link T} based on a list of integers
     *  defining a shape made up of axes sizes as well as a scalar value of type {@link T}
     *  which will fill out the data array spanned by the provided shape information.
     *
     * @param shape A list of integers whose values ought to define the size of the axes of the shape of the new {@link Tsr}.
     * @param value An object of type {@link T} which will populate the data array of the new instance.
     * @return A new {@link Tsr} instance for the generic type {@link T}.
     */
    static <T> Tsr<T> of( List<Integer> shape, T value ) {
        LogUtil.nullArgCheck( shape, "shape", List.class );
        LogUtil.nullArgCheck( value, "value", Object.class );
        return of( (Class<T>) value.getClass(), shape, value );
    }

    /**
     *  This factory method will create and return a {@link Tsr} instance
     *  based on a list of {@link Number} instances whose rounded values will be interpreted as
     *  the shape of this new {@link Tsr} instance and a seed which will serve
     *  as a source of pseudo randomness to generate the values for the new instance.
     *
     * @param shape A list of {@link Number} instances which will be interpreted as a shape array.
     * @param seed A source of pseudo randomness for the {@link Tsr} instance created by this method.
     * @return A new {@link Tsr} instance created based on a shape and a seed.
     */
    static Tsr<Double> of( List<? extends Number> shape, String seed ) {
        int[] shapeArray = new int[ shape.size() ];
        for ( int i = 0; i < shapeArray.length; i++ ) shapeArray[ i ] = shape.get( i ).intValue();
        return of( Double.class, shapeArray, seed );
    }

    /**
     *  Creates a new {@link Tsr} instance based on a list of numbers representing the shape,
     *  and a list of values representing the value of the resulting tensor.
     *
     * @param shape A list of numbers whose integer values will be used to form the shape of the resulting {@link Tsr}.
     * @param value A list of values which will be used to populate the data array of the resulting {@link Tsr}.
     * @param <V> The type parameter of the value list and returned tensor.
     * @return A new {@link Tsr} instance constructed based on the provided shape and value list.
     */
    static <V> Tsr<V> of( List<? extends Number> shape, List<V> value ) {
        Class<V> typeClass = (Class<V>) Object.class;
        if ( value.size() > 0 ) typeClass = (Class<V>) value.get(0).getClass();
        return of(
                DataType.of(typeClass),
                shape.stream().mapToInt(Number::intValue).toArray(),
                value
        );
    }

    /**
     *  Creates a new {@link Tsr} instance based on an array of integers representing the shape,
     *  and a list of values representing the value of the resulting tensor.
     *
     * @param shape An array of integers will be used to form the shape of the resulting {@link Tsr}.
     * @param value A list of values which will be used to populate the data array of the resulting {@link Tsr}.
     * @param <V> The type parameter of the value list and returned tensor.
     * @return A new {@link Tsr} instance constructed based on the provided shape and value list.
     */
    static <V> Tsr<V> of( int[] shape, List<V> value ) {
        Class<V> typeClass = (Class<V>) Object.class;
        if ( value.size() > 0 ) typeClass = (Class<V>) value.get(0).getClass();
        return of( DataType.of(typeClass), shape, value );
    }

    /**
     *  This factory method will turn a list of values or nested lists of values into a {@link Tsr}
     *  instance with the corresponding rank and shape.
     *
     * @param conf A list of either values or nested lists which are themselves either or.
     * @return A new {@link Tsr} instance whose shape and data is based on the provided list structure.
     */
    static Tsr<Object> of( List<Object> conf ) { return of( (Class<Object>) null, conf ); }

    /**
     *  This factory method will turn a list of values or nested lists of values into a {@link Tsr}
     *  instance with the corresponding rank and shape and whose values
     *  are of the provided type.
     *
     * @param targetType The type of the tensor produced by this factory method.
     * @param conf A list of either values or nested lists which are themselves either or.
     * @param <T> The type parameter of the tensor returned by this factoy method.
     * @return A new {@link Tsr} instance whose shape and data is based on the provided list structure.
     */
    static <T> Tsr<T> of( Class<T> targetType, List<Object> conf ) {
        ListReader.Result result = ListReader.read( conf, o -> ( o instanceof Number ? ((Number)o).doubleValue() : o ) );
        Class<T> resultType;
        Object[] resultData;
        int[] shape = result.getShape().stream().mapToInt(i -> i).toArray();
        if ( targetType == null ) {
            resultType = (Class<T>) result.getType();
            resultData = result.getData().toArray();
        } else {
            DataConverter converter = DataConverter.get();
            resultType = targetType;
            resultData = result.getData().parallelStream().map( v -> converter.convert(v, targetType) ).toArray();
        }
        return of( DataType.of(resultType), shape, resultData );
    }

    /**
     *  This is the entry point to the fluent tensor builder API for building
     *  {@link Tsr} instances in a readable and type safe fashion.
     *  The returned {@link WithShapeOrScalarOrVector} is the next step in the
     *  fluent {@link Tsr} builder API which will lead to the creation
     *  of a tensor storing values defined by the provided type class.
     *
     * @return The next step of the {@link Tsr} builder API which exposes methods for defining shapes.
     */
    static <V> WithShapeOrScalarOrVectorOnDevice<V> of( Class<V> typeClass ) { return new NdaBuilder<>( typeClass ); }

    /**
     *  This is a simple convenience method which is simply calling the {@link Tsr#of(Class)}
     *  method like so: {@code of(Double.class)}.
     *  The returned {@link WithShapeOrScalarOrVector} is the next step in the
     *  fluent {@link Tsr} builder API which in this case will lead to the creation
     *  of a tensor storing doubles.
     *
     * @return The next step of the {@link Tsr} builder API which exposes methods for defining shapes.
     */
    static WithShapeOrScalarOrVectorOnDevice<Double> ofDoubles() { return of(Double.class); }

    /**
     *  This is a simple convenience method which is simply calling the {@link Tsr#of(Class)}
     *  method like so: {@code of(Float.class)}.
     *  The returned {@link WithShapeOrScalarOrVector} is the next step in the
     *  fluent {@link Tsr} builder API which in this case will lead to the creation
     *  of a tensor storing floats.
     *
     * @return The next step of the {@link Tsr} builder API which exposes methods for defining shapes.
     */
    static WithShapeOrScalarOrVectorOnDevice<Float> ofFloats() { return of(Float.class); }

    /**
     *  This is a simple convenience method which is simply calling the {@link Tsr#of(Class)}
     *  method like so: {@code of(Integer.class)}.
     *  The returned {@link WithShapeOrScalarOrVector} is the next step in the
     *  fluent {@link Tsr} builder API which in this case will lead to the creation
     *  of a tensor storing integers.
     *
     * @return The next step of the {@link Tsr} builder API which exposes methods for defining shapes.
     */
    static WithShapeOrScalarOrVectorOnDevice<Integer> ofInts() { return of(Integer.class); }

    /**
     *  This is a simple convenience method which is simply calling the {@link Tsr#of(Class)}
     *  method like so: {@code of(Short.class)}.
     *  The returned {@link WithShapeOrScalarOrVector} is the next step in the
     *  fluent {@link Tsr} builder API which in this case will lead to the creation
     *  of a tensor storing shorts.
     *
     * @return The next step of the {@link Tsr} builder API which exposes methods for defining shapes.
     */
    static WithShapeOrScalarOrVectorOnDevice<Short> ofShorts() { return of(Short.class); }

    /**
     *  This is a simple convenience method which is simply calling the {@link Tsr#of(Class)}
     *  method like so: {@code of(Byte.class)}.
     *  The returned {@link WithShapeOrScalarOrVector} is the next step in the
     *  fluent {@link Tsr} builder API which in this case will lead to the creation
     *  of a tensor storing bytes.
     *
     * @return The next step of the {@link Tsr} builder API which exposes methods for defining shapes.
     */
    static WithShapeOrScalarOrVectorOnDevice<Byte> ofBytes() { return of(Byte.class); }

    /**
     * @param value The scalar value which ought to be represented as tensor.
     * @return A scalar double tensor.
     */
    static Tsr<Double> of( double value ) { return of( Double.class, new int[]{ 1 }, value ); }

    /**
     *  Constructs a vector of floats based on the provided array.
     *
     * @param value The array of floats from which a 1D tensor ought to be constructed.
     * @return A vector / 1D tensor of floats.
     */
    static Tsr<Float> of( float... value ) { return of( Float.class, new int[]{ value.length }, value ); }

    /**
     *  Constructs a vector of doubles based on the provided array.
     *
     * @param value The array of doubles from which a 1D tensor ought to be constructed.
     * @return A vector / 1D tensor of doubles.
     */
    static Tsr<Double> of( double... value ) { return of( Double.class, new int[]{ value.length }, value ); }

    /**
     *  Constructs a vector of bytes based on the provided array.
     *
     * @param value The array of bytes from which a 1D tensor ought to be constructed.
     * @return A vector / 1D tensor of bytes.
     */
    static Tsr<Byte> of( byte... value ) { return of( Byte.class, new int[]{ value.length }, value ); }

    /**
     *  Constructs a vector of ints based on the provided array.
     *
     * @param value The array of ints from which a 1D tensor ought to be constructed.
     * @return A vector / 1D tensor of ints.
     */
    static Tsr<Integer> of( int... value ) { return of( Integer.class, new int[]{ value.length }, value ); }

    /**
     *  Constructs a vector of longs based on the provided array.
     *
     * @param value The array of longs from which a 1D tensor ought to be constructed.
     * @return A vector / 1D tensor of longs.
     */
    static Tsr<Long> of( long... value ) { return of( Long.class, new int[]{ value.length }, value ); }

    /**
     *  Constructs a vector of shorts based on the provided array.
     *
     * @param value The array of shorts from which a 1D tensor ought to be constructed.
     * @return A vector / 1D tensor of shorts.
     */
    static Tsr<Short> of( short... value ) { return of( Short.class, new int[]{ value.length }, value ); }

    /**
     *  Constructs a vector of booleans based on the provided array.
     *
     * @param value The array of booleans from which a 1D tensor ought to be constructed.
     * @return A vector / 1D tensor of shorts.
     */
    static Tsr<Boolean> of( boolean... value ) { return of( Boolean.class, new int[]{ value.length }, value ); }

    /**
     *  Use this to construct and return a seeded tensor of the specified type.
     *
     * @param valueType The type class of the items stored by the resulting tensor.
     * @param shape The shape of the resulting tensor consisting of any number of axis-sizes.
     * @param seed An arbitrary {@link String} whose hash will be used to as a seed.
     * @param <V> The type parameter of individual tensor items.
     * @return A newly created and seeded tensor of the provided type and shape.
     */
    static <V> Tsr<V> of( Class<V> valueType, int[] shape, String seed ) { return new TsrImpl<>( valueType, NDConstructor.of(shape), seed ); }

    static Tsr<Double> of( int[] shape, double value ) { return of( Double.class, shape, value ); }

    static Tsr<Double> of( int[] shape, double[] value ) { return of( Double.class, shape, value ); }

    static Tsr<Integer> of( int[] shape, int[] value ) { return of( Integer.class, shape, value ); }

    static Tsr<Byte> of( int[] shape, byte[] value ) { return of( Byte.class, shape, value ); }

    static Tsr<Long> of( int[] shape, long[] value ) { return of( Long.class, shape, value ); }

    static Tsr<Short> of( int[] shape, short[] value ) { return of( Short.class, shape, value ); }

    static Tsr<Float> of( int[] shape, float[] value ) { return of( Float.class, shape, value ); }

    static Tsr<Boolean> of( int[] shape, boolean[] value ) { return of( Boolean.class, shape, value ); }

    static <V> Tsr<V> of( DataType<V> type, int[] shape ) { return new TsrImpl<>( NDConstructor.of(shape), type ); }

    static <V> Tsr<V> of( Class<V> typeClass, int[] shape, Object data ) {
        return of( DataType.of(typeClass), shape, data );
    }

    static <V> Tsr<V> of( Class<V> typeClass, List<Integer> shape, Object data ) {
        return of( DataType.of(typeClass), shape.stream().mapToInt(i -> i).toArray(), data );
    }

    static <V> Tsr<V> of( Class<V> typeClass, List<Integer> shape, List<V> data ) {
        return of( DataType.of( typeClass ), shape.stream().mapToInt( e -> e ).toArray(), data );
    }

    static <V> Tsr<V> of( DataType<V> dataType, List<Integer> shape,  List<V> data ) {
        return of( dataType, shape.stream().mapToInt( i -> i ).toArray(), data.toArray() );
    }

    /**
     *  This factory method is among the most flexible and forgiving ways to create a {@link Tsr} instance.
     *  It receives a {@link DataType} for type safety and to ensure that the produced {@link Tsr} instance
     *  will contain elements of the correct type, a shape array which stores the sizes of the axes that the
     *  instance ought to possess, and finally it receives a data {@link Object} which can be anything ranging from
     *  a {@link List} to an array or simply a single value which ought to fill out the entire {@link Tsr}.
     *
     * @param dataType The data type of the data represented by {@link Tsr} instance created by this method.
     * @param shape An array of axis sizes describing the dimensionality of the {@link Tsr} created by this method.
     * @param data The data for the {@link Tsr} that is about to be created, which can be a list, an array or scalar.
     * @return A new {@link Tsr} instance of the specified type, shape and containing the provided data.
     */
    static <V> Tsr<V> of( DataType<V> dataType, int[] shape, Object data ) { return new TsrImpl<>( NDConstructor.of(shape), dataType, data ); }

    /**
     *  This factory method a raw tensor constructor which will not perform any type checking
     *  or data conversion on the data provided to it.
     *  It constructs the tensor expecting that the data provided to it is of the correct type
     *  and an array of axis sizes.
     *
     * @param dataType The data type of the data represented by {@link Tsr} instance created by this method.
     * @param ndConstructor The {@link NDConstructor} that will be used to construct the {@link Tsr} instance.
     * @param data The data for the {@link Tsr} that is about to be created, which is expected to be an array.
     * @return A new {@link Tsr} instance of the specified type, shape and containing the provided data.
     * @param <V> The type parameter of individual tensor items.
     */
    static <V> Tsr<V> of( DataType<V> dataType, NDConstructor ndConstructor, Data<V> data ) { return new TsrImpl<>( ndConstructor, dataType, data ); }

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
    static <T> Tsr<T> of( DataType<T> type, List<Integer> shape, Filler<T> filler) {
        LogUtil.nullArgCheck( shape, "shape", List.class );
        return of( type, shape.stream().mapToInt( e -> e ).toArray(), filler );
    }

    /**
     *  This factory method allows the creation of tensors with an additional initialization
     *  lambda for filling the underlying data array with desired values.
     *  Besides regular numeric types it is also possible to initialize the
     *  tensor with regular objects like {@link String} instances or custom data types like complex
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
    static <T> Tsr<T> of( DataType<T> type, int[] shape, Filler<T> filler) {
        return new TsrImpl<>( NDConstructor.of(shape), type, filler );
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
     *      <li><i> 'Tsr a = of( "sin( I[0] ) / I[1]", 12f, -6.34f )'</i></li>
     * </ul>
     *
     * @param expression A String which will be used for parsing a Function AST.
     * @param inputs An array of inputs which can be tensors or numeric types.
     */
    @SafeVarargs
    static <V extends Number> Tsr<V> of( String expression, V... inputs ) {
        return Function.of( expression, true ).call( Arrays.stream(inputs).map(args -> TsrImpl._of(args)).toArray(Tsr[]::new) );
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
     *      <li><i> 'Tsr a = of( "sin( I[0] ) / I[1]", List.of(b, c) )'</i></li>
     * </ul>
     *
     * @param expression A String which will be used for parsing a Function AST.
     * @param inputs A list of inputs which can be tensors or numeric types.
     */
    static <V> Tsr<V> of( String expression, List<Tsr<V>> inputs ) {
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
     *      <li><i> 'Tsr a = of( "sin( I[0] ) / I[1]", true, List.of(b, c) )'</i></li>
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
    static <V> Tsr<V> of( String expression, boolean doAD, List<Tsr<V>> tensors ) {
        return Function.of( expression, doAD ).call( tensors );
    }

    /**
     *  This method takes a tensor and a String expression describing
     *  operations which ought to be applied to said tensor.
     *  This expression will be parsed to a {@link Function} instance expecting one input,
     *  namely : "I[0]" <br>
     *  An example would be the following :
     * <ul>
     *      <li><i> 'Tsr a = of( "sin( I[0] ) * 2", b )'</i></li>
     * </ul>
     *
     *  Which takes the tensor 'b' and applies the function "f(x) = sin(x) * 2"
     *  element-wise to produce a new tensor 'a'! <br>
     *  <br>
     *
     * @param tensor A tensor which serves as input to the Function instance parsed from the given expression.
     * @param expression The expression describing operations applied to the provided tensor.
     */
    static <V> Tsr<V> of( String expression, Tsr<V> tensor ) {
        return Function.of( expression, true ).call( tensor );
    }

    /**
     *  This method takes an array of tensors and a String expression describing
     *  operations which ought to be applied to the tensors in said array.
     *  This expression will be parsed to a {@link Function} instance expecting as many inputs
     *  as there are array entries, namely : "I[0]", "I[1]", "I[2]", ... <br>
     *  An example would be the following :
     * <ul>
     *      <li><i> 'Tsr a = of( "sin( I[0] ) / I[1]", b, c )'</i></li>
     * </ul>
     *
     *  Which takes the tensor 'b' and 'c' and applies the function "f(x,y) = sin(x) / y"
     *  element-wise to produce a new tensor 'a'! <br>
     *
     * @param expression The expression describing operations applied to the provided tensors.
     * @param tensors An array of tensors used as inputs to the Function instance parsed from the provided expression.
     */
    @SafeVarargs
    static <V> Tsr<V> of( String expression, Tsr<V>... tensors ) {
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
     *      <li><i> 'Tsr a = of( "sin( I[0] ) / I[1]", true, b, c )'</i></li>
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
    static <V> Tsr<V> of( String expression, boolean doAD, Tsr<V>... tensors ) {
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
    static <V> Tsr<V> ofRandom( Class<V> valueTypeClass, int... shape ) {
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
     * @return A new {@link Tsr} instance with the same data type, shape and memory location as the provided template.
     */
    static <V> IterByOrIterFromOrAllTsr<V> like( Tsr<V> template ) {
        return of( template.getDataType().getItemTypeClass() )
                .on( template.getDevice() )
                .withShape( template.getNDConf().shape() );
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
     * @return This very {@link Tsr} instance in order to enable method chaining.
     */
    Tsr<V> setRqsGradient( boolean rqsGradient );

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
    default boolean isOutsourced() {
        return !(this.getDevice() instanceof CPU);
    }

    /**
     *  A Virtual tensor is a tensor whose underlying data array is of size 1, holding only a single value. <br>
     *  This only makes sense for homogeneously populated tensors.
     *  An example of such a tensor would be: <br>
     *  {@code Tsr.ofInts().withShape(x,y).all(n)}                           <br><br>
     *  The reasons for this feature is that it greatly improves performance in certain cases.
     *  In essence this feature is a form of lazy loading.
     *
     *  Use {@link #setIsVirtual(boolean)} to "actualize" a "virtual" tensor, and vise versa.
     *
     * @return The truth value determining if this tensor is "virtual" or "actual".
     */
    boolean isVirtual();

    /**
     *  Virtualizing is the opposite to actualizing a tensor.
     *  A tensor is virtual if the size of the underlying data is not actually equal to
     *  the number of elements which the tensor claims to store, aka its size.
     *  This is for example the case when initializing a tensor filled with a single
     *  value continuously. In that case the tensor will flag itself as virtual and only allocate the
     *  underlying data array to hold a single item even though the tensor might actually hold
     *  many more items.
     *  The reasons for this feature is that it greatly improves performance in certain cases.
     *  In essence this feature is a form of lazy loading.
     *  <br><br>
     *  WARNING! Virtualizing is the process of compacting the underlying data array
     *  down to an array holding a single value item.
     *  This only makes sense for homogeneously populated tensors.
     *  Passing {@code false} to this method will "actualize" a "virtual" tensor.
     *  Meaning the underlying data array will at least become as large as the size of the tensor
     *  as is defined by {@link #size()}.
     *
     * @param isVirtual The truth value determining if this tensor should be "virtual" or "actual".
     * @return This concrete instance, to allow for method chaining.
     */
    Tsr<V> setIsVirtual( boolean isVirtual );

    /**
     *  This will check if the {@link Unsafe#delete()} method was previously called on this tensor.
     *  This means that the tensor data was freed on every device
     *  and any references inside the tensor are null (to be eligable for garbage collection).
     *
     * @return The truth value determining if the {@link Unsafe#delete()} method has been called oin this instance.
     */
    boolean isDeleted();

    /**
     *  A tensor is empty if it's {@link Data} storage is null.
     *  This is true for deleted tensors or tensors which have not been initialized yet.
     *
     * @return The truth value determining if this tensor has no {@link Data}.
     */
    default boolean isEmpty() { return getUnsafe().getData() == null; }

    /**
     *  A tensor is "undefined" if it has either no {@link NDConfiguration} implementation instance
     *  or this instance does not have a shape set for this {@link Tsr} which is needed for
     *  a tensor to also have a rank and dimensionality...
     *
     * @return The truth value determining if this tensor has an {@link NDConfiguration} stored internally.
     */
    default boolean isUndefined() { return getNDConf() == null || getNDConf().shape() == null; }

    /** {@inheritDoc} */
    @Override
    default boolean isSlice() {
        Relation<V> child = get( Relation.class );
        return ( child != null && child.hasParent() );
    }

    /** {@inheritDoc} */
    @Override
    default int sliceCount() {
        Relation<V> child = this.get( Relation.class );
        return ( child != null ) ? child.childCount() : 0;
    }

    /** {@inheritDoc} */
    @Override
    default boolean isSliceParent() {
        Relation<V> parent = this.get( Relation.class );
        return ( parent != null && parent.hasChildren() );
    }

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
    default boolean belongsToGraph() { return this.has( GraphNode.class ); }

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
    default boolean isLeave() { return (!this.belongsToGraph() || getGraphNode().isLeave()); }

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
    default boolean hasGradient() { return this.has( Tsr.class ); }

    /**
     *  This flag works alongside two autograd features which can be enables inside the library settings.
     *  They will come into effect when flipping their feature flags, <br>
     *  namely: <i>'isApplyingGradientWhenRequested'</i> and <i>'isApplyingGradientWhenTensorIsUsed'</i><br>
     *  As the first flag name suggests gradients will be applied to their tensors when it is set to true,
     *  however this will only happened when the second flag is set to true as well, because otherwise gradients
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
     *  This flag works alongside two autograd features which can be enables inside the library settings.
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
    Tsr<V> setGradientApplyRequested( boolean applyRequested );

    /*==================================================================================================================
    |
    |       §(3) : COMPONENT SYSTEM
    |   --------------------------------
    */

    /**
     *
     *  Important : Components of type {@link Tsr} are simply gradients!
     *  Currently, this method is used only to catch illegal arguments which
     *  is for example the case when trying to attach a gradient with a different shape...
     *  (Otherwise the gradient tensor "does not mind" an owner change...)
     */
    @Override
    default boolean update( OwnerChangeRequest<Tsr<V>> changeRequest ) {
        if ( changeRequest.type() == IsBeing.ADDED ) {
            if (
                changeRequest.getNewOwner().shape().hashCode() != this.shape().hashCode() ||
                Arrays.hashCode(changeRequest.getNewOwner().getNDConf().shape()) != Arrays.hashCode( getNDConf().shape() )
            ) {
                throw new IllegalArgumentException(
                        "Trying to attach a tensor as gradient component to a tensor with different shape."
                );
            }
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
     *  This method returns the {@link DataType} instance of this {@link Tsr}, which is
     *  a wrapper object for the actual type class representing the value items stored inside
     *  the underlying data array of this tensor.
     *
     * @return The {@link DataType} instance of this {@link Tsr} storing important type information.
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
     *         extension instance which might also be sub-classes of the {@link NumericType} interface
     *         to model unsigned types or other JVM foreign numeric concepts.
     */
    Class<?> getRepresentativeItemClass();

    /*==================================================================================================================
    |
    |       §(5) : OBJECT STATE MODIFICATION :
    |   ------------------------------------------
    */

    /**
     *  This method exposes an API for mutating the state of this tensor.
     *  The usage of methods exposed by this API is generally discouraged
     *  because the exposed state can easily lead to broken tensors and exceptional situations!<br>
     *  <br><b>
     *
     *  Only use this if you know what you are doing and
     *  performance is critical! <br>
     *  </b>
     *  (Like custom backend extensions for example)
     *
     * @return The unsafe API exposes methods for mutating the state of the tensor.
     */
    Unsafe<V> getUnsafe();

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
     * This method takes a {@link Device} and tries to migrate the contents of this {@link Tsr}
     * instance to that {@link Device}!
     *
     * @param device The {@link Device} which should host this {@link Tsr} as well as be added to its components list.
     * @return This very class to enable method chaining.
     */
    Tsr<V> to( Device<?> device );

    /**
     * @param deviceType A search key identifying the device onto which this tensor should be stored.
     * @return This very tensor instance in order to enable method chaining.
     */
    default Tsr<V> to( String deviceType ) { return this.to(Device.get(deviceType)); }

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
    default Tsr<V> backward( Tsr<V> error ) {
        LogUtil.nullArgCheck(error, "error", Tsr.class, "Cannot back-propagate 'null'!");
        if ( this.isOutsourced() )
            error = error.deepCopy().to(this.getDevice());

        Tsr<V> finalError = error;
        if ( !forComponent( GraphNode.class, node -> node.backward(finalError) ) && this.rqsGradient() ) {
            addToGradient( error );
        }
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
    default Tsr<V> backward( double value ) {
        backward( Tsr.of( this.getItemType(), getNDConf().shape(), value ) );
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
    default Tsr<V> backward() {
        backward( 1 ); // By default we back-propagate a base factor of 1.
        return this;
    }

    /**
     * @return The gradient of this tensor which is internally stored as component.
     */
    default Tsr<V> getGradient() { return this.get( Tsr.class ); }

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
        forComponent( JITProp.class, JITProp::execute );
        // Afterwards the JITProp component is not needed anymore! So we remove it.
        remove( JITProp.class );
        // Now the gradient can be applied (Gradients are also tensors, which is why we provide its class as key).
        forComponent(
                Tsr.class,
                g -> {
                    // If an optimizer is present then we also optimize the gradient first!
                    if ( this.has( Optimizer.class ) )
                        g = this.get(Optimizer.class).optimize( this );
                    // And then we remove the gradient because it is no longer needed.
                    remove( Tsr.class );
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
            return (Device<V>) CPU.get();
        else
            return device;
    }

    /**
     * @return The graph node of the computation graph to which this tensor belongs or null if not part of a graph.
     */
    default GraphNode<V> getGraphNode() { return get( GraphNode.class ); }

    /**
     * @return An instance of the {@link NDFrame} component if present.
     */
    default NDFrame<V> frame() { return get( NDFrame.class ); }

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
    default Tsr<V> detached() {
        if ( this.has( GraphNode.class ) )
            return this.shallowCopy().remove( GraphNode.class );
        return this;
    }

    /**
     *  This method receives a nested {@link String} array which
     *  ought to contain a label for the index of this tensor.
     *  The index for a single element of this tensor would be an array
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
     * @param labels A nested String array containing labels for indexes of the tensor dimensions.
     * @return This tensor (method chaining).
     */
    default Tsr<V> label( String[]... labels ) {
        label( null, labels );
        return this;
    }

    /**
     *  This method receives a label for this tensor and a
     *  nested {@link String} array which ought to contain a
     *  label for the index of this tensor.
     *  The index for a single element of this tensor would be an array
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
     * @param tensorName A label for this tensor itself.
     * @param labels A nested String array containing labels for indexes of the tensor dimensions.
     * @return This tensor (method chaining).
     */
    Tsr<V> label( String tensorName, String[]... labels );

    /**
     *  This method receives a nested {@link String} list which
     *  ought to contain a label for the index of this tensor.
     *  The index for a single element of this tensor would be an array
     *  of numbers as long as the rank where every number is
     *  in the range of the corresponding shape dimension...
     *  Labeling an index means that for every dimension there
     *  must be a label for elements in this range array! <br>
     *  For example the shape (2,3) could be labeled as follows: <br>
     *                                                           <br>
     *      dim 0 : ["A", "B"]                                   <br>
     *      dim 1 : ["1", "2", "3"]                              <br>
     *                                                           <br>
     * @param labels A nested String list containing labels for indexes of the tensor dimensions.
     * @return This tensor (method chaining).
     */
    Tsr<V> label( List<List<Object>> labels );

    /**
     *  This method receives a label for this tensor and a nested
     *  {@link String} list which ought to contain a label for the index of
     *  this tensor The index for a single element of this tensor would
     *  be an array of numbers as long as the rank where every number is
     *  in the range of the corresponding shape dimension...
     *  Labeling an index means that for every dimension there
     *  must be a label for elements in this range array! <br>
     *  For example the shape (2,3) could be labeled as follows: <br>
     *                                                           <br>
     *      dim 0 : ["A", "B"]                                   <br>
     *      dim 1 : ["1", "2", "3"]                              <br>
     *                                                           <br>
     * @param tensorName A label for this tensor itself.
     * @param labels A nested String list containing labels for indexes of the tensor dimensions.
     * @return This tensor (method chaining).
     */
    Tsr<V> label( String tensorName, List<List<Object>> labels );

    /**
     *  This method provides the ability to
     *  label not only the indices of the shape of this tensor, but also
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
     * @return This tensor (method chaining).
     */
    Tsr<V> label( Map<Object, List<Object>> labels );

    /**
     *  This method provides the ability to
     *  label not only the indices of the shape of this tensor, but also
     *  the dimension of the shape.
     *  The first and only argument of the method expects a map instance
     *  where keys are the objects which ought to act as dimension labels
     *  and the values are lists of labels for the indices of said dimensions.
     *  For example the shape (2,3) could be labeled as follows:            <br>
     *  [                                                                   <br>
     *     "dim 0" : ["A", "B"],                                            <br>
     *     "dim 1" : ["1", "2", "3"]                                        <br>
     *  ]                                                                   <br>
     *                                                                      <br>
     * @param tensorName A label for this tensor itself.
     * @param labels A map in which the keys are dimension labels and the values are lists of index labels for the dimension.
     * @return This tensor (method chaining).
     */
    Tsr<V> label( String tensorName, Map<Object, List<Object>> labels );



    /*==================================================================================================================
    |
    |       §(8) : (OVERLOADABLE) OPERATORS & OPERATIONS :
    |   -----------------------------------------------------
    |       ...for more context see package 'calculus'...
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
     *  where the left operand is this {@link Tsr}
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
     * @return The sum of this instance as the left and the passed {@link Tsr} instance as right operand.
     */
    default Tsr<V> plus( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot add 'null' to a tensor!");
        return Neureka.get().backend().getAutogradFunction().plus().call( this, other );
    }

    /**
     *  Performs an addition of the passed tensor to this tensor.
     *  The result of the addition will be stored in this tensor (inline operation).
     *
     * @param other The tensor which ought to be added to this tensor.
     * @return This tensor.
     */
    default Tsr<V> plusAssign( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot add-assign 'null' to a tensor!");
        return Neureka.get().backend().getFunction().plusAssign().call( this, other );
    }

    /**
     *  This method will create a new {@link Tsr}
     *  with the provided double scalar added to all elements of this {@link Tsr}.
     *
     *  The shapes of this tensor is irrelevant as the provided value will
     *  simply be broadcast to any possible shape.
     *
     * @param value The right operand of the addition.
     * @return The sum between this instance as the left and the passed double as right operand.
     */
    default Tsr<V> plus( V value ) { return plus( of( itemType(), this.shape(), value ) ); }

    /**
     *  Performs subtraction on
     *  two tensors with the same rank (or two ranks which can be made compatible with padding ones),
     *  where the left operand is this {@link Tsr}
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
     * @return The difference between this instance as the left and the passed {@link Tsr} instance as right operand.
     */
    default Tsr<V> minus( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot subtract 'null' from a tensor!");
        return Neureka.get().backend().getAutogradFunction().minus().call( this, other );
    }

    default Tsr<V> minus( V other ) {
        LogUtil.nullArgCheck(other, "other", this.getItemType(), "Cannot subtract 'null' from a tensor!");
        return minus(
                of( this.getDataType().getItemTypeClass() )
                        .withShape(this.getNDConf().shape())
                        .all(other)
        );
    }
    default Tsr<V> minusAssign( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot subtract-assign 'null' from a tensor!");
        return Neureka.get().backend().getFunction().minusAssign().call( this, other );
    }

    /**
     * @param other The scalar value which should be subtracted from the values of this tensor.
     * @return This tensor after the minus-assign inline operation was applied.
     */
    default Tsr<V> minusAssign( V other ) {
        LogUtil.nullArgCheck(other, "other", this.getItemType(), "Cannot subtract-assign 'null' from a tensor!");
        return minusAssign(
                of( this.getDataType().getItemTypeClass() )
                        .withShape(this.getNDConf().shape())
                        .all(other)
        );
    }

    /**
     * @return A clone of this tensor where the signs of all elements are flipped.
     */
    default Tsr<V> negative() { return Neureka.get().backend().getAutogradFunction().neg().call( this ); }

    /**
     *  Creates and returns a new {@link Tsr} instance which is a transposed twin of this instance.<br>
     *  This is a shorter alternative to the functionally identical {@link #getT()} method.
     *
     * @return A new transposed tensor with the same underlying {@link Data} as this tensor.
     */
    default Tsr<V> T() {
        if ( this.rank() == 1 ) return this;
        else if ( this.rank() == 2 ) {
            boolean wasIntermediate = this.isIntermediate();
            this.getUnsafe().setIsIntermediate(false);
            Tsr<V> result = Neureka.get().backend().getFunction().transpose2D().call( this );
            this.getUnsafe().setIsIntermediate(wasIntermediate);
            return result;
        }
        StringBuilder operation = new StringBuilder();
        for ( int i = rank() - 1; i >= 0; i-- ) operation.append( i ).append( i == 0 ? "" : ", " );
        operation = new StringBuilder( "[" + operation + "]:(I[ 0 ])" );
        return Function.of( operation.toString(), true ).call( this );
    }

    /**
     *  Calculate the mean value of all values
     *  within this tensor and returns it
     *  in the form of a scalar tensor. <br>
     *  This operation supports autograd.
     *
     * @return A scalar tensor which wraps the mean value of all values of this tensor.
     */
    default Tsr<V> mean() {
        Functions functions = Neureka.get().backend().getAutogradFunction();
        Tsr<V> sum = this.sum();
        Tsr<V> result = functions.div().call( sum, of( this.getItemType(), new int[]{1}, this.size() ) );
        if (sum != this) sum.getUnsafe().delete(); // This is a temporary tensor which is not needed anymore! (not even for back propagation)
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
    default Tsr<V> sum() {
        Functions functions = Neureka.get().backend().getAutogradFunction();
        Tsr<V> sum = functions.sum().call( this );
        if ( sum == null )
            throw new IllegalStateException(
                    "Failed to calculate sum using function! Shape: "+
                    Arrays.toString(this.getNDConf().shape())
                );
        return sum;
    }

    /**
     *  Calculate the min value of all values
     *  within this tensor and returns it
     *  in the form of a scalar tensor. <br>
     *  This operation supports autograd.
     *
     * @return A scalar tensor which wraps the smallest of all values of this tensor.
     */
    default Tsr<V> min() {
        Functions functions = Neureka.get().backend().getAutogradFunction();
        Tsr<V> min = functions.min().call( this );
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
    default Tsr<V> max() {
        Functions functions = Neureka.get().backend().getAutogradFunction();
        Tsr<V> max = functions.max().call( this );
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
    default Tsr<V> convDot( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class);
        Tsr<V> a = this;
        int[][] fitter = NDUtil.makeFit( a.getNDConf().shape(), other.getNDConf().shape() );
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
     *  that it is functionally completely identical to the {@link #matMul(Tsr)} method.
     *
     * @param other The tensor which is the right part of the dot product operation.
     * @return A new tensor which is the dot product of this tensor and the passed one.
     */
    default Tsr<V> dot( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot perform dot operation when second operand is 'null'!");
        if ( this.rank() != 2 && other.rank() != 2 )
            throw new IllegalStateException("Not yet implemented!"); // This is not yet available in the backend!
        return this.matMul( other );
    }

    /**
     *  The {@link #matMul(Tsr)} method will produce the matrix product of
     *  two 2 dimensional arrays, where the left operand is this {@link Tsr}
     *  instance and the right operand is the tensor passed to the method.
     *
     * @param other The right operand of the matrix multiplication.
     * @return The matrix product of this instance as the left and the passed {@link Tsr} instance as right operand.
     */
    default Tsr<V> matMul( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot perform matrix multiplication operation when second operand is 'null'!");
        if ( this.rank() != 2 || other.rank() != 2 )
            throw new IllegalArgumentException(
                    "Cannot perform matrix multiplication for tensors whose ranks are not both 2!\n" +
                    "Encountered ranks: " + this.rank() + ", " + other.rank() + ";"
                );

        return Neureka.get().backend().getAutogradFunction().matMul().call( this, other );
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
    default Tsr<V> dimtrim() { return Neureka.get().backend().getAutogradFunction().dimTrim().call( this ); }

    /**
     *  A method which returns a new {@link Tsr} instance which is a transposed twin of this instance.<br>
     *  This is an alternative to the functionally identical {@link #T()} method.
     *
     * @return A new transposed tensor with the same underlying {@link Data} as this tensor.
     */
    default Tsr<V> getT() { return this.T(); } // Transposed

    /**
     *  This method name translates to the "in" keyword in Groovy!
     *  The same is true for the "contains" method in Kotlin.
     *  Both methods do the exact same thing, however they exist
     *  for better language support.
     *
     * @param other The tensor which will be checked.
     * @return The answer to the following question: Is the data of the provided tensor a subset of the data of this tensor?
     */
    boolean isCase( Tsr<V> other );

    /**
     *  This method name translates to the "in" keyword in Kotlin!
     *  The same is true for the "isCase" method in Groovy.
     *  Both methods do the exact same thing, however they exist
     *  for better language support.
     *
     * @param other The tensor which will be checked.
     * @return The answer to the following question: Is the data of the provided tensor a subset of the data of this tensor?
     */
    default boolean contains( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot perform 'contains' operation when second operand is 'null'!");
        return this.isCase( other );
    }

    /**
     *  This method is synonymous to the {@link #times(Tsr)} method.
     *  Both of which will produce the product of
     *  two tensors with the same rank (or two ranks which can be made compatible with padding ones),
     *  where the left operand is this {@link Tsr}
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
     * @return The product of this instance as the left and the passed {@link Tsr} instance as right operand.
     */
    default Tsr<V> multiply( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot multiply 'null' with a tensor!");
        return Neureka.get().backend().getAutogradFunction().mul().call( this, other );
    }

    /**
     * @param other The value which should be broadcast to all elements of a clone of this tensor.
     * @return A new clone of this tensor where all elements are multiplied by the provided value.
     */
    default Tsr<V> multiply( V other ) {
        LogUtil.nullArgCheck(other, "other", this.getItemType(), "Cannot multiply 'null' with a tensor!");
        return multiply(
                of( this.getDataType().getItemTypeClass() )
                        .withShape( this.getNDConf().shape() )
                        .all( other )
        );
    }

    /**
     *  This is a functionally identical synonym to the {@link #multiply(Tsr)} method.
     *  Both of which will produce the product of
     *  two tensors with the same rank (or two ranks which can be made compatible with padding ones),
     *  where the left operand is this {@link Tsr}
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
     * @return The product of this instance as the left and the passed {@link Tsr} instance as right operand.
     */
    default Tsr<V> times( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot multiply 'null' with a tensor!");
        return multiply( other );
    }

    /**
     * @param other The value which should be broadcast to all elements of a clone of this tensor.
     * @return A new clone of this tensor where all elements are multiplied by the provided value.
     */
    default Tsr<V> times( V other ) {
        LogUtil.nullArgCheck(other, "other", getItemType(), "Cannot multiply 'null' with a tensor!");
        return multiply( other );
    }
    /**
     * @param other The tensor whose elements ought to be multiplied and assigned to elements in this tensor.
     * @return This instance where each value element was multiplied by the corresponding element in the provided tensor.
     */
    default Tsr<V> timesAssign( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot multiply-assign 'null' to a tensor!");
        return Neureka.get().backend().getFunction().mulAssign().call( this, other );
    }

    /**
     * @param other The value which ought to be multiplied and assigned to each element in this tensor.
     * @return This instance where each value element was multiplied by the provided element.
     */
    default Tsr<V> timesAssign( V other ) {
        LogUtil.nullArgCheck(other, "other", this.getItemType(), "Cannot multiply-assign 'null' to a tensor!");
        return this.timesAssign( of( getItemType(), getNDConf().shape(), other ) );
    }
    /**
     * @param value The value which should be broadcast to all elements of a clone of this tensor.
     * @return A new clone of this tensor where all elements are multiplied by the provided value.
     */
    default Tsr<V> multiply( double value ) { return multiply( of( getItemType(), getNDConf().shape(), value ) ); }

    /**
     *  This method will produce the quotient of
     *  two tensors with the same rank (or two ranks which can be made compatible with padding ones),
     *  where the left operand is this {@link Tsr}
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
     * @return The quotient of this instance as the left and the passed {@link Tsr} instance as right operand.
     */
    default Tsr<V> div( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot divide a tensor by 'null' (In any sense of the word)!");
        return Neureka.get().backend().getAutogradFunction().div().call( this, other );
    }
    default Tsr<V> div( V value ) { return div( of( getItemType(), getNDConf().shape(), value ) ); }

    default Tsr<V> divAssign( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot divide-assign a tensor by 'null' (In any sense of the word)!");
        return Neureka.get().backend().getFunction().divAssign().call( this, other );
    }

    /**
     *  Produces the modulus of
     *  two tensors with the same rank (or two ranks which can be made compatible with padding ones),
     *  where the left operand is this {@link Tsr}
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
     * @return The modulus of this instance as the left and the passed {@link Tsr} instance as right operand.
     */
    default Tsr<V> mod( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot perform tensor modulo 'null'!");
        return Neureka.get().backend().getAutogradFunction().mod().call( this, other );
    }

    default Tsr<V> mod( int other ) { return mod(of(getItemType(), getNDConf().shape(), other)); }

    /**
     *  This method is synonymous to the {@link #mod(int)} method.
     */
    default Tsr<V> rem( int other ) { return this.mod(other); }

    default Tsr<V> modAssign( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot perform tensor modulo 'null'!");
        return Neureka.get().backend().getFunction().modAssign().call( this, other );
    }

    /**
     *  This will produce the power of
     *  two tensors with the same rank (or two ranks which can be made compatible with padding ones),
     *  where the left operand is this {@link Tsr}
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
     * @return The power of this instance as the left and the passed {@link Tsr} instance as right operand.
     */
    default Tsr<V> power( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot raise a tensor to the power of 'null'!");
        return Neureka.get().backend().getAutogradFunction().pow().call( this, other );
    }

    /**
     *  Raises all items of this tensor to the power of the provided value.
     *  The returned tensor is a new instance which will have the same shape as this tensor.
     * 
     * @param value The value which should be used to raise all items of this tensor to the power of.
     * @return A new clone of this tensor where all items are raised to the power of the provided value.
     */
    default Tsr<V> power( V value ) {
        LogUtil.nullArgCheck(value, "value", getItemType(), "Cannot raise a tensor to the power of 'null'!");
        return power( of( this.itemType(), this.shape(), value ) );
    }
    
    /**
     *  This method is a functionally identical synonym to the {@link #power(Tsr)} method.
     */
    default Tsr<V> xor( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot raise a tensor to the power of 'null'!");
        return Neureka.get().backend().getAutogradFunction().pow().call( this, other );
    }

    /**
     *  This method is a functionally identical synonym to the {@link #power(Tsr)} method.
     */
    default Tsr<V> xor( double value ) { return xor( of( this.itemType(), this.shape(), value ) ); }

    
    /*==================================================================================================================
    |
    |       §(9) : SLICING, INDEXING & INJECTING :
    |   -----------------------------------------------------
    |       ...for more context see package 'ndim.config'...
    */

    /** {@inheritDoc} */
    @Override AxisOrGetTsr<V> slice();

    /** {@inheritDoc} */
    @Override
    Tsr<V> getAt( int... indices );

    /** {@inheritDoc} */
    @Override
    default Tsr<V> getAt( Number i ) {
        return getAt( Collections.singletonList( getNDConf().indicesOfIndex( (i).intValue() ) ).toArray() );
    }

    /** {@inheritDoc} */
    @Override
    default Tsr<V> get( int... indices ) { return getAt( indices ); }

    /** {@inheritDoc} */
    @Override
    default Tsr<V> getAt( Object... args ) {
        List<Object> argsList = Arrays.asList( args );
        return getAt( argsList );
    }

    /** {@inheritDoc} */
    @Override
    default Tsr<V> get( Object... args ) { return getAt( args ); }

    /** {@inheritDoc} */
    @Override
    default Tsr<V> getAt( int i ) { return getAt( indicesOfIndex(i) ); }

    /** {@inheritDoc} */
    @Override
    default Tsr<V> get( int i ) { return getAt( i ); }

    /** {@inheritDoc} */
    @Override
    default Tsr<V> get( Number i ) { return getAt( i ); }

    /** {@inheritDoc} */
    @Override
    default Tsr<V> get( Object key ) { return getAt( key ); }

    /** {@inheritDoc} */
    @Override
    Tsr<V> getAt( Map<?,Integer> rangToStrides );

    /** {@inheritDoc} */
    @Override Tsr<V> getAt( List<?> key );

    /** {@inheritDoc} */
    @Override Tsr<V> putAt( Map<?,Integer> key, Nda<V> value );

    @Override Tsr<V> putAt( int[] indices, V value );

    /** {@inheritDoc} */
    @Override default Tsr<V> set( int[] indices, V value ) { return putAt( indices, value ); }

    /** {@inheritDoc} */
    @Override default Tsr<V> putAt( int index, V value ) { return putAt( indicesOfIndex(index), value ); }

    /** {@inheritDoc} */
    @Override default Tsr<V> set( int index, V value ) { return putAt( index, value ); }

    /** {@inheritDoc} */
    @Override Tsr<V> putAt( List<?> key, Nda<V> value );

    /** {@inheritDoc} */
    @Override default Tsr<V> putAt( List<?> indices, V value ) {
        if ( indices.stream().allMatch( i -> i instanceof Number ) )
            return setItemAt( indexOfIndices(indices.stream().mapToInt( i -> ((Number)i).intValue() ).toArray()), value );
        else
            return this.putAt( indices, of( this.getItemType(), shape(), value ) );
    }

    /** {@inheritDoc} */
    @Override Tsr<V> setItemAt( int i, V o );

    /** {@inheritDoc} */
    @Override Tsr<V> setItems( Object value );

    
    /*==================================================================================================================
    |
    |       §(10) : Mapping :
    |   -----------------------------------------------------
    |       ...transformation and modification...
    */

    /** {@inheritDoc} */
    @Override default <T> Tsr<T> mapTo(
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
                .borrow( (Tsr<Object>) this )
                .in( () -> {
                    Object data = getUnsafe().getData().getRef();
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
                            int[] sourceData = (int[]) getUnsafe().getData().getRef();
                            access = (i -> (Number) mapper.apply((V) Integer.valueOf(sourceData[i])));
                        } else if (this.getItemType() == Double.class) {
                            double[] sourceData = (double[]) getUnsafe().getData().getRef();
                            access = (i -> (Number) mapper.apply((V) Double.valueOf(sourceData[i])));
                        } else if (this.getItemType() == Float.class) {
                            float[] sourceData = (float[]) getUnsafe().getData().getRef();
                            access = (i -> (Number) mapper.apply((V) Float.valueOf(sourceData[i])));
                        } else if (this.getItemType() == Short.class) {
                            short[] sourceData = (short[]) getUnsafe().getData().getRef();
                            access = (i -> (Number) mapper.apply((V) Short.valueOf(sourceData[i])));
                        } else if (this.getItemType() == Byte.class) {
                            byte[] sourceData = (byte[]) getUnsafe().getData().getRef();
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
                            int[] sourceData = (int[]) getUnsafe().getData().getRef();
                            access = (i -> mapper.apply((V) Integer.valueOf(sourceData[i])));
                        } else if ( this.getItemType() == Double.class ) {
                            double[] sourceData = (double[]) getUnsafe().getData().getRef();
                            access = (i -> mapper.apply((V) Double.valueOf(sourceData[i])));
                        } else if ( this.getItemType() == Float.class ) {
                            float[] sourceData = (float[]) getUnsafe().getData().getRef();
                            access = (i -> mapper.apply((V) Float.valueOf(sourceData[i])));
                        } else if ( this.getItemType() == Short.class ) {
                            short[] sourceData = (short[]) getUnsafe().getData().getRef();
                            access = (i -> mapper.apply((V) Short.valueOf(sourceData[i])));
                        } else if ( this.getItemType() == Byte.class ) {
                            byte[] sourceData = (byte[]) getUnsafe().getData().getRef();
                            access = (i -> mapper.apply((V) Byte.valueOf(sourceData[i])));
                        } else
                            throw new IllegalArgumentException(failMessage);

                        newData = map.toObjectArray(access);
                    }
                    return Tsr.of( typeClass, this.getNDConf().shape(), newData );
                });
    }

    /** {@inheritDoc} */
    @Override default Tsr<V> map( java.util.function.Function<V,V> mapper ) {
        return mapTo( this.getItemType(), mapper );
    }

    /**
     *  Turns this tensor into a {@link BufferedImage} based on the provided
     *  {@link Tsr.ImageType} formatting choice.
     *
     * @param type The type of format used to create the buffered image.
     * @return A {@link BufferedImage} populated with the contents of this tensor.
     */
    BufferedImage asImage( Tsr.ImageType type );

    /**
     *  This method takes the provided {@link Tsr} instance and adds its
     *  contents to the contents of the {@link Tsr} which is set as gradient of this very {@link Tsr}.
     *
     * @param error The error gradient which ought to be added to the gradient of this tensor.
     * @return This very tensor instance to enable method chaining.
     */
    Tsr<V> addToGradient( Tsr<V> error );

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
    @Override Tsr<V> deepCopy();

    /** {@inheritDoc} */
    @Override default Tsr<V> shallowCopy() {
        if ( this.isEmpty() || this.isUndefined() ) return this; // Maybe throw an exception here...
        return slice().detached();
    }

    /**
     * @return A shallow copy of this tensor with autograd support.
     */
    default Tsr<V> shallowClone() {
        if ( this.isEmpty() || this.isUndefined() ) return this; // Maybe throw an exception here...
        return slice().get();
    }

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

    /**
     *  Tensors should be considered immutable, however sometimes it
     *  is important to mutate their state for performance reasons.
     *  This interface exposes several methods for mutating the state of this tensor.
     *  The usage of methods exposed by this API is generally discouraged
     *  because the exposed state can easily lead to broken tensors and exceptions...<br>
     *  <br>
     */
    interface Unsafe<T>
    {
        /**
         *  This method sets the NDConfiguration of this NDArray.
         *  Therefore, it should not be used lightly as it can cause major internal inconsistencies.
         *
         * @param configuration The new NDConfiguration instance which ought to be set.
         * @return The final instance type of this class which enables method chaining.
         */
        Tsr<T> setNDConf( NDConfiguration configuration );

        /**
         *  This method is an inline operation which changes the underlying data of this tensor.
         *  It converts the data types of the elements of this tensor to the specified type!<br>
         *  <br>
         *  <b>WARNING : The usage of this method is discouraged for the following reasons: </b><br>
         *  <br>
         *  1. Inline operations are inherently error-prone for most use cases. <br>
         *  2. This inline operation in particular has no safety net,
         *     meaning that there is no implementation of version mismatch detection
         *     like there is for those operations present in the standard operation backend...
         *     No exceptions will be thrown during backpropagation! <br>
         *  3. This method has not yet been implemented to also handle instances which
         *     are slices of parent tensors!
         *     Therefore, there might be unexpected performance penalties or side effects
         *     associated with this method.<br>
         *     <br>
         *
         * @param typeClass The target type class for elements of this tensor.
         * @param <V> The type parameter for the returned tensor.
         * @return The same tensor instance whose data has been converted to hold a different type.
         */
        <V> Tsr<V> toType( Class<V> typeClass );

        /**
         *  Use this to do a runtime checked upcast of the type parameter of the tensor.
         *  This is unsafe because it is in conflict with the {@link #itemType()}
         *  method.
         *
         * @param superType The class of the super type of the tensor's value type.
         * @return A tensor whose type parameter is upcast.
         * @param <U> The super type parameter of the value type of the tensor.
         */
        <U/*super T*/> Tsr<U> upcast(Class<U> superType );

        /**
         *  This method enables modifying the data-type configuration of this {@link AbstractNda}.
         *  Warning! The method should not be used unless absolutely necessary.
         *  This is because it can cause unpredictable inconsistencies between the
         *  underlying {@link DataType} instance of this {@link AbstractNda} and the actual type of the actual
         *  data it is wrapping (or it is referencing on a {@link neureka.devices.Device}).<br>
         *  <br>
         * @param dataType The new {@link DataType} which ought to be set.
         * @return The tensor with the new data type set.
         */
        <V> Tsr<V> setDataType( DataType<V> dataType );

        /**
         *  This method allows you to modify the data-layout of this {@link AbstractNda}.
         *  Warning! The method should not be used unless absolutely necessary.
         *  This is because it can cause unpredictable side effects especially for certain
         *  operations expecting a particular data layout (like for example matrix multiplication).
         *  <br>
         *
         * @param layout The layout of the data array (row or column major).
         * @return The final instance type of this class which enables method chaining.
         */
        Tsr<T> toLayout( NDConfiguration.Layout layout );

        /**
         *  This method is responsible for incrementing
         *  the "_version" field variable which represents the version of the data of this tensor.
         *  Meaning :
         *  Every time the underlying data (_value) changes this version ought to increment alongside.
         *  The method is called during the execution procedure.
         *
         * @param call The context object containing all relevant information that defines a call for tensor execution.
         * @return This very tensor instance. (factory pattern)
         */
        Tsr<T> incrementVersion( ExecutionCall<?> call );

        /**
         *  Intermediate tensors are internal non-user tensors which may be eligible
         *  for deletion when further consumed by a {@link Function}.
         *  For the casual user of Neureka, this flag should always be false!
         *
         * @param isIntermediate The truth value determining if this tensor is not a user tensor but an internal
         *                       tensor which may be eligible for deletion by {@link Function}s consuming it.
         * @return The tensor to which this unsafe API belongs.
         */
        Tsr<T> setIsIntermediate( boolean isIntermediate );

        /**
         *  Although tensors will be garbage collected when they are not strongly referenced,
         *  there is also the option to manually free up the tensor and its associated data in a native environment.
         *  This is especially useful when tensors are stored on a device like the {@link neureka.devices.opencl.OpenCLDevice}.
         *  In that case calling this method will free the memory reserved for this tensor on the device.
         *  This manual memory freeing through this method can be faster than waiting for
         *  the garbage collector to kick in at a latr point in time... <br>
         *  <br>
         *
         * @return The tensor wo which this unsafe API belongs to allow for method chaining.
         */
        Tsr<T> delete();

        Data<T> getData();

        <A> A getDataAs( Class<A> arrayTypeClass );

        /**
         *  A tensor ought to have some way to selectively modify its underlying data array.
         *  This method simply overrides an element within this data array sitting at position "i".
         * @param i The index of the data array entry which ought to be addressed.
         * @param o The object which ought to be placed at the requested position.
         * @return This very tensor in order to enable method chaining.
         */
        Tsr<T> setDataAt( int i, T o );

        Tsr<T> setData(Data data );

        /**
         *  Use this to access the underlying writable data of this tensor if
         *  you want to modify it.
         *  This method will ensure that you receive an instance of whatever array type you provide
         *  or throw descriptive exceptions to make sure that any unwanted behaviour does not
         *  spread further in the backend.
         *
         * @param arrayTypeClass The expected array type underlying the tensor.
         * @param <A> The type parameter of the provided type class.
         * @return The underlying data array of this tensor.
         */
        default <A> A getDataForWriting( Class<A> arrayTypeClass ) {
            LogUtil.nullArgCheck( arrayTypeClass, "arrayTypeClass", Class.class, "Array type must not be null!" );
            if ( !arrayTypeClass.isArray() )
                throw new IllegalArgumentException("Provided type is not an array type.");
            Object data = Unsafe.this.getData().getRef();
            if ( data == null )
                throw new IllegalStateException("Could not find writable tensor data for this tensor (Maybe this tensor is stored on a device?).");

            if ( !arrayTypeClass.isAssignableFrom(data.getClass()) )
                throw new IllegalStateException("The data of this tensor does not match the expect type! Expected '"+arrayTypeClass+"' but got '"+data.getClass()+"'.");

            return (A) data;
        }

        /**
         *  <b>This method detaches this tensor from its underlying computation-graph
         *  or simply does nothing if no graph is present.</b> <br>
         *  Nodes within a computation graph are instances of the "{@link GraphNode}" class which are also
         *  simple components of the tensors they represent in the graph. <br>
         *  Therefore, "detaching" this tensor from the graph simply means removing its {@link GraphNode} component.
         *
         * @return This very instance in order to allows for a more streamline usage of this method.
         */
        Tsr<T> detach();

    }

}
