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
     \/_____/_|         A long yet shallow class.

    This is the the core work-horse class of Neureka. The 'Tsr' class!
    It is a three-letter abbreviation of the word "Tensor"!

------------------------------------------------------------------------------------------------------------------------

   'Any fool can write code that a computer can understand.
    Good programmers write code that humans can understand.'
    – Martin Fowler

    Use the following as search keys :)

    $(1) : CONSTRUCTION
        $(1.0) : GENERIC CONSTRUCTION
        $(1.1) : SHAPE LIST BASED CONSTRUCTION
        $(1.2) : SHAPE ARRAY BASED CONSTRUCTION
        §(1.3) : LAMBDA BASED CONSTRUCTION
        §(1.4) : FUNCTION BASED CONSTRUCTION

    §(2) : FLAGS
        §(2.0) : GRADIENT REQUIREMENT
        §(2.1) : SOURCE LOCATION (DEVICE)
        §(2.2) : VIRTUAL / ACTUAL
        §(2.3) : GRADIENT APPLY REQUIREMENT
        §(2.4) : DELETION

    §(3) : COMPONENT SYSTEM
        §(3.0) : SETTING / REJECTING
        §(3.1) : REMOVING / REJECTING
        §(3.2) : UPDATING

    §(4) : PROPERTIES
        $(4.0) : HIGH LEVEL PROPERTIES
        §(4.1) : COMPONENT PROPERTIES
        §(4.2) : INNER PROPERTIES

    §(5) : OBJECT STATE MODIFICATION

    §(6) : ND-ITERATOR LOGIC

    §(7) : COMPONENT SPECIFIC
        §(7.0) : AUTO-GRAD
        §(7.1) : FRAMING

    §(8) : (OVERLOADABLE) OPERATORS & OPERATIONS
        §(8.0) : OPERATORS
        §(8.1) : OPERATIONS

    §(9) : SLICING, INDEXING & INJECTING
        §(9.0) : SLICING
        §(9.1) : INJECTING

    §(10) : MAPPING
*/

package neureka;

import neureka.autograd.GraphNode;
import neureka.autograd.JITProp;
import neureka.backend.api.ExecutionCall;
import neureka.backend.standard.memory.MemUtil;
import neureka.backend.standard.operations.other.Reshape;
import neureka.calculus.Function;
import neureka.calculus.Functions;
import neureka.common.composition.AbstractComponentOwner;
import neureka.common.composition.Component;
import neureka.common.utility.DataConverter;
import neureka.common.utility.ListReader;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;
import neureka.dtype.DataType;
import neureka.dtype.custom.F32;
import neureka.dtype.custom.F64;
import neureka.fluent.building.TensorBuilder;
import neureka.fluent.building.states.IterByOrIterFromOrAll;
import neureka.fluent.building.states.WithShapeOrScalarOrVector;
import neureka.fluent.building.states.WithShapeOrScalarOrVectorOnDevice;
import neureka.fluent.slicing.SliceBuilder;
import neureka.fluent.slicing.SmartSlicer;
import neureka.framing.NDFrame;
import neureka.framing.Relation;
import neureka.framing.fluent.AxisFrame;
import neureka.ndim.AbstractTensor;
import neureka.ndim.Initializer;
import neureka.ndim.config.AbstractNDC;
import neureka.ndim.config.NDConfiguration;
import neureka.ndim.iterators.NDIterator;
import neureka.optimization.Optimizer;
import neureka.view.TsrAsString;
import neureka.view.TsrStringSettings;
import org.jetbrains.annotations.NotNull;
import org.slf4j.LoggerFactory;

import java.awt.*;
import java.awt.image.*;
import java.util.*;
import java.util.List;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


/**
 *  This class name {@link Tsr} is a 3 letter abbreviation of the word "tensor", a mathematical concept.
 *  A tensor is a type of multidimensional data-structure with certain transformation properties.
 *  Technically however, it is mostly a simple container / data-structure which can house data indexed by N dimensions.
 *  Therefore it is often also described as a nd-array.
 *  Elements of a tensor are also mostly numeric.<br>
 *  This means that: <br>
 *  <i><b>...a tensor of rank 0 is a scalar, a tensor of rank 1 is a vector and a tensor of rank 2 is a matrix, etc...</b></i>
 *  <br><br>
 *  Consequently, tensors are a perfect fit for applying various operations on them.
 *  Such operations might be simple elementwise operations or more complex linear operations like
 *  the dot-product, matrix- or even tensor multiplications. <br>
 *  <br>
 * @param <V> The type parameter for the individual value items within this tensor.
 */
public class Tsr<V> extends AbstractTensor<Tsr<V>, V> implements Component<Tsr<V>>, Cloneable
{
    static {
        _CPU = CPU.get();
        _LOG = LoggerFactory.getLogger( Tsr.class );
    }

    /**
     *  The default device is an instance of the {@link CPU} class. <br>
     *  This field is a reference to this default device implementation.
     */
    private static final Device<Number> _CPU;

    /**
     *  This field contains multiple flags.
     *  The bits of this integer are used to encode various states which a tensor can have.
     *  These bits are flipped by bitmasks which are defined below.
     */
    private byte _flags = 0;

    /**
     *  The following fields are bit masks used to store true / false values
     *  in a targeted bit inside the {@link #_flags} variable.
     */
    private static final byte RQS_GRADIENT_MASK       = 1;
    private static final byte IS_OUTSOURCED_MASK      = 2;
    private static final byte IS_VIRTUAL_MASK         = 4;
    private static final byte GRADIENT_APPLY_RQD_MASK = 8;
    private static final byte IS_DELETED_MASK         = 16;
    private static final byte IS_INTERMEDIATE_MASK    = 32;

    /**
     *  This integer represents the version of the data (accessible through {@link #getData()})
     *  stored within this tensor.
     *  It gets incremented every time an inline operation occurs!
     *  {@link GraphNode} instances tied to this tensor (as component) store
     *  a reference version which is a copy of this field.
     *  If this version changes, despite there being a GraphNode which might
     *  perform auto-differentiation at some point, then an exception will be thrown for debugging.
     *  <br>
     *  The corresponding getter returns the version of the data (accessible through {@link #getData()})
     *  stored within this tensor.
     */
    private int _version = 0;


    /*==================================================================================================================
    |
    |       §(1) : CONSTRUCTION
    |   ---------------------------
    */
    /*
        -------------------------------------------
            §(1.0) : GENERIC CONSTRUCTION
        --------------------------------------------
    */

    /**
     *  This static factory method creates and return a completely empty and undefined tensor
     *  which is void of any contents and meaning.
     *  The use case for this would be to use the produced {@link Tsr}
     *  instance as a target for an inline operations which fills the instance with an actual value. <br>
     *  An example of this approach would be to call the {@link #putAt(List, Tsr)} method with an empty list as key.
     *  This will be interpreted as an inline copy of the contents of the
     *  second parameter into this {@link Tsr} instance.
     */
    public static Tsr<Object> newInstance() { return new Tsr<>(); }

    /**
     *  This constructor creates a completely empty tensor which is void of any contents and meaning.
     *  The use case for this would be to use the produced {@link Tsr}
     *  instance as a target for an inline operation which fills this instance with an actual value. <br>
     *  An example of this approach would be to call the {@link #putAt(List, Tsr)} method with an empty list as key.
     *  This will be interpreted as an inline copy of the contents of the
     *  second parameter into this {@link Tsr} instance.
     *  This constructor will be called by the {@link Tsr#newInstance()} factory method.
     */
    private Tsr() {}

    /**
     *  This static {@link Tsr} factory method tries to interpret the provided
     *  arguments to create the instance the use might wants.
     *
     * @param args The arguments which ought to be interpreted.
     * @return The result of the interpretation in the form of a {@link Tsr} instance of typ {@link Object}.
     */
    public static <T> Tsr<T> of( Object... args ) { return _of( args ); }

    private static <T> Tsr<T> _of( Object... args )
    {
        if ( args == null || args.length == 0 ) return new Tsr<>();
        if ( args.length == 1 ) {
            Tsr<T> t = new Tsr<>();
            boolean success = t.createConstructionAPI().constructAllFromOne( new int[]{ 1 }, args[ 0 ] );
            if ( !success ) {
                String message = "Cannot create tensor from argument of type '" + args[ 0 ].getClass().getName() + "'!";
                _LOG.error( message );
                throw new IllegalArgumentException( message );
            }
            return t;
        }
        args[ 0 ] = ( args[ 0 ] instanceof ArrayList ) ? ( (List<?>) args[ 0 ] ).toArray() : args[ 0 ];
        args[ 1 ] = ( args[ 1 ] instanceof ArrayList ) ? ( (List<?>) args[ 1 ] ).toArray() : args[ 1 ];
        if ( args[ 0 ] instanceof Object[] ) {
            if ( ( (Object[]) args[ 0 ] )[ 0 ] instanceof Integer || ((Object[])args[ 0 ])[ 0 ] instanceof Double) {
                args[ 0 ] = _intArray( (Object[]) args[ 0 ] );
            }
        }
        //CASES:
        if ( args[ 0 ] instanceof int[] ) {
            if ( args[ 1 ] instanceof Double || args[ 1 ] instanceof Integer ) {
                Tsr<T> t = new Tsr<>();
                args[ 1 ] = ( args[ 1 ] instanceof Integer ) ? ( (Integer) args[ 1 ] ).doubleValue() : args[ 1 ];
                t.createConstructionAPI().constructAllFromOne( (int[]) args[ 0 ], args[ 1 ] );
                return t;
            } else {
                Tsr<T> t = new Tsr<>();
                t._setDataType( DataType.of( args[1].getClass() ) );
                t._constructAndAllocate( (int[]) args[0], true );
                ((Object[])t.getData())[0] = args[1];
                return t;
            }
        }
        /* EXPRESSION BASED CONSTRUCTION: 
            The following allows the creation of tensors based on passing an expression
            alongside input tensors to the constructor.
            An example would be:
            
                Tsr<?> t = Tsr.of( "tanh(", x, ") * 7 ^", y );  
        */
        boolean containsString = false;
        int numberOfTensors = 0;
        ArrayList<Tsr<T>> tsrList = new ArrayList<>();
        for ( Object o : args ) {
            containsString = ( o instanceof String ) || containsString;
            if ( o instanceof Tsr ) {
                tsrList.add( (Tsr<T>) o );
                numberOfTensors++;
            }
        }
        boolean doAD = true;
        Tsr<T>[] tensors = new Tsr[ numberOfTensors ];
        StringBuilder f = new StringBuilder();
        int ti = 0;
        for ( Object o : args ) {
            if ( tsrList.contains( o ) ) {
                tensors[ ti ] = ( (Tsr<T>) o );
                f.append( "I[" ).append( ti ).append( "]" );
                ti++;
            }
            else if ( o instanceof  String ) f.append( (String) o );
            else
                _LOG.debug(
                    "Unexpected tensor construction argument of type '"+o.getClass().getSimpleName()+"'"
                );
        }
        if ( tensors.length == 0 || tensors[0] == null) return new Tsr<>();
        return Function.of( f.toString(), doAD ).call( tensors );
    }


    /*
        -------------------------------------------
            §(1.1) : SHAPE LIST BASED CONSTRUCTION
        --------------------------------------------
    */

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
    public static <T> Tsr<T> of( List<Integer> shape, T value ) {
        if ( value == null ) throw new IllegalArgumentException("Provided value is null!");
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
    public static Tsr<Double> of( List<? extends Number> shape, String seed ) {
        int[] shp = new int[ shape.size() ];
        for ( int i = 0; i < shp.length; i++ ) shp[ i ] = shape.get( i ).intValue();
        return of( Double.class, shp, seed );
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
    public static <V> Tsr<V> of( List<? extends Number> shape, List<V> value ) {
        Class<V> typeClass = (Class<V>) Object.class;
        if ( value.size() > 0 ) typeClass = (Class<V>) value.get(0).getClass();
        return Tsr.of(
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
    public static <V> Tsr<V> of( int[] shape, List<V> value ) {
        Class<V> typeClass = (Class<V>) Object.class;
        if ( value.size() > 0 ) typeClass = (Class<V>) value.get(0).getClass();
        return Tsr.of(
                    DataType.of(typeClass),
                    shape,
                    value
                );
    }

    /**
     *  This factory method will turn a list of values or nested lists of values into a {@link Tsr}
     *  instance with the corresponding rank and shape.
     *
     * @param conf A list of either values or nested lists which are themselves either or.
     * @return A new {@link Tsr} instance whose shape and data is based on the provided list structure.
     */
    public static Tsr<Object> of( List<Object> conf ) {
        return of( (Class<Object>) null, conf );
    }

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
    public static <T> Tsr<T> of( Class<T> targetType, List<Object> conf ) {
        boolean isDoubleMatrix = 
                            conf.stream()
                                .allMatch( e ->
                                    e instanceof List
                                            &&
                                    ((List<Object>) e).stream().allMatch( v -> v instanceof Double )
                                );

        if ( isDoubleMatrix )
            return new Tsr<>( conf );

        ListReader.Result result = ListReader.read( conf, o -> ( o instanceof Number ? ((Number)o).doubleValue() : o ) );
        Class resultType;
        Object[] resultData;
        int[] shape = result.getShape().stream().mapToInt(i -> i).toArray();
        if ( targetType == null ) {
            resultType = result.getType();
            resultData = result.getData().toArray();
        } else {
            DataConverter converter = DataConverter.instance();
            resultType = targetType;
            resultData = result.getData().parallelStream().map( v -> converter.convert(v, targetType) ).toArray();
        }
        return Tsr.of( DataType.of(resultType), shape, resultData );
    }


    /**
     * @param axesSizes A list of numbers which will be interpreted as shape array.
     * @return A tensor for storing numbers.
     */
    public static Tsr<Number> ofShape( List<? extends Number> axesSizes ) {
        return ofShape( axesSizes.toArray( new Number[0] ) );
    }

    /**
     * @param axesSizes An array of numbers which will be interpreted as shape array.
     * @return A tensor for storing numbers.
     */
    @SafeVarargs
    public static <T extends Number> Tsr<Number> ofShape( T... axesSizes ) {
        int[] shape = Arrays.stream( axesSizes ).mapToInt( Number::intValue ).toArray();
        return Tsr.ofShape( shape );
    }

    /**
     *  See {@link #of(List)}.
     */
    private Tsr( List<Object> conf ) {
        createConstructionAPI().constructFor( conf.stream().map(e -> (List<Object>) e ).collect( Collectors.toList() ) );
    }

    /*
        -------------------------------------------
            §(1.2) : SHAPE ARRAY BASED CONSTRUCTION
        --------------------------------------------
    */

    /**
     *  This is the entry point to the fluent tensor builder API for building
     *  {@link Tsr} instances in a readable and type safe fashion.
     *  The returned {@link WithShapeOrScalarOrVector} is the next step in the
     *  fluent {@link Tsr} builder API which will lead to the creation
     *  of a tensor storing values defined by the provided type class.
     *
     * @return The next step of the {@link Tsr} builder API which exposes methods for defining shapes.
     */
    public static <V> WithShapeOrScalarOrVectorOnDevice<V> of( Class<V> typeClass ) { return new TensorBuilder<>( typeClass ); }

    /**
     *  This is a simple convenience method which is simply calling the {@link Tsr#of(Class)}
     *  method like so: {@code Tsr.of(Double.class)}.
     *  The returned {@link WithShapeOrScalarOrVector} is the next step in the
     *  fluent {@link Tsr} builder API which in this case will lead to the creation
     *  of a tensor storing doubles.
     *
     * @return The next step of the {@link Tsr} builder API which exposes methods for defining shapes.
     */
    public static WithShapeOrScalarOrVectorOnDevice<Double> ofDoubles() { return of(Double.class); }

    /**
     *  This is a simple convenience method which is simply calling the {@link Tsr#of(Class)}
     *  method like so: {@code Tsr.of(Float.class)}.
     *  The returned {@link WithShapeOrScalarOrVector} is the next step in the
     *  fluent {@link Tsr} builder API which in this case will lead to the creation
     *  of a tensor storing floats.
     *
     * @return The next step of the {@link Tsr} builder API which exposes methods for defining shapes.
     */
    public static WithShapeOrScalarOrVectorOnDevice<Float> ofFloats() { return of(Float.class); }

    /**
     *  This is a simple convenience method which is simply calling the {@link Tsr#of(Class)}
     *  method like so: {@code Tsr.of(Integer.class)}.
     *  The returned {@link WithShapeOrScalarOrVector} is the next step in the
     *  fluent {@link Tsr} builder API which in this case will lead to the creation
     *  of a tensor storing integers.
     *
     * @return The next step of the {@link Tsr} builder API which exposes methods for defining shapes.
     */
    public static WithShapeOrScalarOrVectorOnDevice<Integer> ofInts() { return of(Integer.class); }

    /**
     *  This is a simple convenience method which is simply calling the {@link Tsr#of(Class)}
     *  method like so: {@code Tsr.of(Short.class)}.
     *  The returned {@link WithShapeOrScalarOrVector} is the next step in the
     *  fluent {@link Tsr} builder API which in this case will lead to the creation
     *  of a tensor storing shorts.
     *
     * @return The next step of the {@link Tsr} builder API which exposes methods for defining shapes.
     */
    public static WithShapeOrScalarOrVectorOnDevice<Short> ofShorts() { return of(Short.class); }

    /**
     *  This is a simple convenience method which is simply calling the {@link Tsr#of(Class)}
     *  method like so: {@code Tsr.of(Byte.class)}.
     *  The returned {@link WithShapeOrScalarOrVector} is the next step in the
     *  fluent {@link Tsr} builder API which in this case will lead to the creation
     *  of a tensor storing bytes.
     *
     * @return The next step of the {@link Tsr} builder API which exposes methods for defining shapes.
     */
    public static WithShapeOrScalarOrVectorOnDevice<Byte> ofBytes() { return of(Byte.class); }

    /**
     * @param value The scalar value which ought to be represented as tensor.
     * @return A scalar double tensor.
     */
    public static Tsr<Double> of( double value ) { return new Tsr<>(value); }

    private Tsr( double value ) { createConstructionAPI().constructAllFromOne( new int[]{ 1 }, value ); }

    /**
     *  Constructs a vector of floats based on the provided array.
     *
     * @param value The array of floats from which a 1D tensor ought to be constructed.
     * @return A vector / 1D tensor of floats.
     */
    public static Tsr<Float> of( float... value ) { return new Tsr<>( value ); }

    private Tsr( float[] value ) { createConstructionAPI().constructForFloats( new int[]{ value.length }, value ); }

    /**
     *  Constructs a vector of doubles based on the provided array.
     *
     * @param value The array of doubles from which a 1D tensor ought to be constructed.
     * @return A vector / 1D tensor of doubles.
     */
    public static Tsr<Double> of( double... value ) { return new Tsr<>( value ); }

    private Tsr( double[] value ) { createConstructionAPI().constructForDoubles( new int[]{ value.length }, value ); }

    /**
     *  Constructs a vector of bytes based on the provided array.
     *
     * @param value The array of bytes from which a 1D tensor ought to be constructed.
     * @return A vector / 1D tensor of bytes.
     */
    public static Tsr<Byte> of( byte... value ) { return new Tsr<>( value ); }

    private Tsr( byte[] value ) { createConstructionAPI().constructForBytes( new int[]{ value.length }, value ); }

    /**
     *  Constructs a vector of ints based on the provided array.
     *
     * @param value The array of ints from which a 1D tensor ought to be constructed.
     * @return A vector / 1D tensor of ints.
     */
    public static Tsr<Integer> of( int... value ) { return new Tsr<>( new int[]{ value.length }, value ); }

    private Tsr( int[] shape, int[] value ) { createConstructionAPI().constructForInts( shape, value ); }

    /**
     *  Constructs a vector of longs based on the provided array.
     *
     * @param value The array of longs from which a 1D tensor ought to be constructed.
     * @return A vector / 1D tensor of longs.
     */
    public static Tsr<Long> of( long... value ) { return new Tsr<>( new int[]{ value.length }, value ); }

    private Tsr( int[] shape, long[] value ) { createConstructionAPI().constructForLongs( shape, value ); }

    /**
     *  Constructs a vector of shorts based on the provided array.
     *
     * @param value The array of shorts from which a 1D tensor ought to be constructed.
     * @return A vector / 1D tensor of shorts.
     */
    public static Tsr<Short> of( short... value ) { return new Tsr<>( new int[]{ value.length }, value ); }

    private Tsr( int[] shape, short[] value ) { createConstructionAPI().constructForShorts( shape, value ); }

    public static <V> Tsr<V> of( Class<V> valueType, int[] shape, String seed ) { return new Tsr<>( valueType, shape, seed ); }

    /**
     *  Constructs a vector of booleans based on the provided array.
     *
     * @param value The array of booleans from which a 1D tensor ought to be constructed.
     * @return A vector / 1D tensor of shorts.
     */
    public static Tsr<Short> of( boolean... value ) { return new Tsr<>( new int[]{ value.length }, value ); }

    private Tsr( int[] shape, boolean[] value ) { createConstructionAPI().constructForBooleans( shape, value ); }

    /**
     *  See {@link #of(Class, int[], String)}
     *  ...and {@link #of(List, String)}
     */
    private Tsr( Class<V> valueType, int[] shape, String seed ) {
        createConstructionAPI().constructSeeded( valueType, shape, seed );
    }

    public static Tsr<Number> ofShape( int[] shape ) { return new Tsr<>( shape ); }

    private Tsr( int[] shape ) { _constructAndAllocate( shape, true ); }

    public static Tsr<Double> of( int[] shape, double value ) { return new Tsr<>( shape, value ); }

    private Tsr( int[] shape, double value ) { createConstructionAPI().constructAllFromOne( shape, value ); }

    public static Tsr<Double> of( int[] shape, double[] value ) { return new Tsr<>( shape, value ); }

    private Tsr( int[] shape, double[] value ) { createConstructionAPI().constructForDoubles( shape, value ); }

    public static <V> Tsr<V> of( DataType<V> type, int[] shape ) { return new Tsr<>( shape, type ); }


    private Tsr( int[] shape, DataType<?> type )
    {
        _setDataType( DataType.of( type.getTypeClass() ) );
        _constructAndAllocate( shape, true );
    }

    public static <V> Tsr<V> of( Class<V> typeClass, int[] shape, Object data ) { return of( DataType.of(typeClass), shape, data ); }

    public static <V> Tsr<V> of( Class<V> typeClass, List<Integer> shape, Object data ) { return of( DataType.of(typeClass), shape.stream().mapToInt(i -> i).toArray(), data ); }

    public static <V> Tsr<V> of( Class<V> typeClass, List<Integer> shape, List<V> data ) {
        return Tsr.of(
                    DataType.of( typeClass ),
                    shape.stream().mapToInt( e -> e ).toArray(),
                    data
                );
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
     * @param <V>
     * @return A new {@link Tsr} instance of the specified type, shape and containing the provided data.
     */
    public static <V> Tsr<V> of( DataType<V> dataType, int[] shape, Object data ) { return new Tsr<>( shape, dataType, data ); }

    private Tsr( int[] shape, DataType<?> dataType, Object data ) { createConstructionAPI().tryConstructing( shape, dataType, data ); }

    public static <V> Tsr<V> of( DataType<V> dataType, List<Integer> shape,  List<V> data ) {
        return Tsr.of(
                dataType,
                shape.stream().mapToInt( i -> i ).toArray(),
                data.toArray()
        );
    }

    // Inner construction layer:

    private void _constructAndAllocate(int[] shape, boolean virtual )
    {
        createConstructionAPI().configureFromNewShape( shape, virtual, true );
    }

    private static int[] _intArray( Object[] arg ) {
        int length = arg.length;
        int[] array = new int[ length ];
        for ( int i = 0; i < length; i++ ) {
            if ( arg[ i ] instanceof Double ) array[ i ] = ( (Double) arg[ i ] ).intValue();
            else array[ i ] = (Integer) arg[ i ];
        }
        return array;
    }

    /*
        -------------------------------------------
            §(1.3) : LAMBDA BASED CONSTRUCTION
        --------------------------------------------
    */

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
     * @param initializer The lambda Object which ought to fill this tensor with the appropriate data.
     * @param <T> The type parameter for the actual data array items.
     */
    public static <T> Tsr<T> of( DataType<T> type, List<Integer> shape, Initializer<T> initializer ) {
        return Tsr.of(
                    type,
                    shape.stream().mapToInt( e -> e ).toArray(),
                    initializer
                );
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
     * @param initializer The lambda Object which ought to fill this tensor with the appropriate data.
     * @param <T> The type parameter for the actual data array items.
     */
    public static <T> Tsr<T> of( DataType<T> type, int[] shape, Initializer<T> initializer ) {
        return new Tsr<>( shape, type, initializer );
    }

    /**
     *  see {@link #of(DataType, int[], Initializer)}
     *
     * @param shape The shape of this new tensor ought to have.
     * @param type The data type this tensor ought to have.
     * @param initializer The lambda Object which ought to fill this tensor with the appropriate data.
     * @param <T> The type parameter for the actual data array items.
     */
    private <T> Tsr( int[] shape, DataType<T> type, Initializer<T> initializer )
    {
        _constructFromInitializer( shape, type, initializer );
    }

    /**
     * @param shape The shape of that this new tensor ought to have.
     * @param type The data type that this tensor ought to have.
     * @param initializer The lambda Object which ought to fill this tensor with the appropriate data.
     * @param <T> The type parameter for the actual data array items.
     */
    private <T> void _constructFromInitializer(int[] shape, DataType<T> type, Initializer<T> initializer )
    {
        _setDataType( type );
        _constructAndAllocate( shape, false );
        _initData( initializer );
    }


    /*
        -------------------------------------------
            §(1.4) : FUNCTION BASED CONSTRUCTION
        --------------------------------------------
     */

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
     *      <li><i> 'Tsr a = Tsr.of( "sin( I[0] ) / I[1]", 12f, -6.34f )'</i></li>
     * </ul>
     *
     * @param expression A String which will be used for parsing a Function AST.
     * @param inputs An array of inputs which can be tensors or numeric types.
     */
    public static <V extends Number> Tsr<V> of( String expression, V... inputs ) {
        return Tsr.of( expression, Arrays.asList(inputs) );
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
     *      <li><i> 'Tsr a = Tsr.of( "sin( I[0] ) / I[1]", List.of(b, c) )'</i></li>
     * </ul>
     *
     * @param expression A String which will be used for parsing a Function AST.
     * @param inputs A list of inputs which can be tensors or numeric types.
     */
    public static <V> Tsr<V> of( String expression, List<? extends Object> inputs ) {
        if ( inputs.stream().allMatch( e -> e instanceof Tsr ) )
            return Function.of( expression, true ).call( inputs.stream().toArray( Tsr[]::new ) );
        else {
            return Function.of( expression, true ).call(  inputs.stream().map(args -> _of(args)).toArray(Tsr[]::new) );
        }
    }

    /**
     *  This method takes a tensor and a String expression describing
     *  operations which ought to be applied to said tensor.
     *  This expression will be parsed to a {@link Function} instance expecting one input,
     *  namely : "I[0]" <br>
     *  An example would be the following :
     * <ul>
     *      <li><i> 'Tsr a = Tsr.of( "sin( I[0] ) * 2", b )'</i></li>
     * </ul>
     *
     *  Which takes the tensor 'b' and applies the function "f(x) = sin(x) * 2"
     *  elementwise to produce a new tensor 'a'! <br>
     *  <br>
     *
     * @param tensor A tensor which serves as input to the Function instance parsed from the given expression.
     * @param expression The expression describing operations applied to the provided tensor.
     */
    public static <V> Tsr<V> of( String expression, Tsr<V> tensor ) {
        return  Function.of( expression, true ).call( tensor );
    }

    /**
     *  This method takes an array of tensors and a String expression describing
     *  operations which ought to be applied to the tensors in said array.
     *  This expression will be parsed to a {@link Function} instance expecting as many inputs
     *  as there are array entries, namely : "I[0]", "I[1]", "I[2]", ... <br>
     *  An example would be the following :
     * <ul>
     *      <li><i> 'Tsr a = Tsr.of( "sin( I[0] ) / I[1]", b, c )'</i></li>
     * </ul>
     *
     *  Which takes the tensor 'b' and 'c' and applies the function "f(x,y) = sin(x) / y"
     *  elementwise to produce a new tensor 'a'! <br>
     *
     * @param expression The expression describing operations applied to the provided tensors.
     * @param tensors An array of tensors used as inputs to the Function instance parsed from the provided expression.
     */
    @SafeVarargs
    public static <V> Tsr<V> of( String expression, Tsr<V>... tensors ) {
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
     *      <li><i> 'Tsr a = Tsr.of( "sin( I[0] ) / I[1]", true, b, c )'</i></li>
     * </ul>
     *  Which takes the tensor 'b' and 'c' and applies the function "f(x,y) = sin(x) / y"
     *  elementwise to produce a new tensor 'a'!
     *  Additionally there is a helpful flag which allows one to specify if the
     *  parsed {@link Function} instance emerging from the provided expression
     *  should also allow the tracking of computations via a computation graph ({@link GraphNode} instances).
     *  This history tracking then enables auto-differentiation. <br>
     *
     * @param expression The expression describing operations applied to the provided tensors.
     * @param doAD A flag which when set to true commands the creation of a computation graph during operation execution.
     * @param tensors An array of tensors used as inputs to the Function instance parsed from the provided expression.
     *
     */
    @SafeVarargs
    public static <V> Tsr<V> of( String expression, boolean doAD, Tsr<V>... tensors ) {
        return Function.of( expression, doAD ).call( tensors );
    }

    /*==================================================================================================================
    |
    |       §(2) : FLAGS
    |   ----------------------
    */
    /*
        --------------------------------------------
            §(2.0) : GRADIENT REQUIREMENT  :
        --------------------------------------------
    */

    /**
     *  Settings this flag via this setter will indirectly trigger the activation of
     *  the autograd / auto-differentiation system of this library!
     *  If the flag is set to 'true' and the tensor is used for computation then
     *  it will also receive gradients when the {@link #backward()} method is being called
     *  on any descendant tensor within the computation graph.
     *
     * @param rqsGradient The truth value determining if this tensor ought to receive gradients via
     *                     the built in automatic backpropagation system.
     * @return This very {@link Tsr} instance in order to enable method chaining.
     */
    public Tsr<V> setRqsGradient( boolean rqsGradient ) {
        if ( rqsGradient() != rqsGradient && !rqsGradient ) this.remove( Tsr.class );
        _setRqsGradient( rqsGradient );
        return this;
    }

    /**
     *  This flag will indirectly trigger the activation of the autograd / auto-differentiation system of this library!
     *  If the flag is set to 'true' and the tensor is used for computation then
     *  it will also receive gradients when the {@link #backward()} method is being called
     *  on any descendant tensor within the computation graph.
     *
     * @return The truth value determining if this tensor ought to receive gradients via
     *         the built in automatic backpropagation system.
     */
    public boolean rqsGradient() { return ( _flags & RQS_GRADIENT_MASK ) == RQS_GRADIENT_MASK; }

    protected void _setRqsGradient( boolean rqsGradient ) {
        if ( rqsGradient() != rqsGradient ) {
            if ( rqsGradient ) _flags += RQS_GRADIENT_MASK;
            else               _flags -= RQS_GRADIENT_MASK;
        }
    }

    /**
     *  Intermediate tensors are internal non-user tensors which may be eligible
     *  for deletion when further consumed by a {@link Function}.
     *  For the casual user of Neureka, this flag should always be false!
     *
     * @return The truth value determining if this tensor is not a user tensor but an internal
     *         tensor which may be eligible for deletion by {@link Function}s consuming it.
     */
    public boolean isIntermediate() { return ( _flags & IS_INTERMEDIATE_MASK ) == IS_INTERMEDIATE_MASK; }

    /**
     *  Intermediate tensors are internal non-user tensors which may be eligible
     *  for deletion when further consumed by a {@link Function}.
     *  For the casual user of Neureka, this flag should always be false!
     *
     * @param isIntermediate The truth value determining if this tensor is not a user tensor but an internal
     *                       tensor which may be eligible for deletion by {@link Function}s consuming it.
     */
    protected void _setIsIntermediate( boolean isIntermediate ) {
        if ( isIntermediate() != isIntermediate ) {
            if ( isIntermediate ) _flags += IS_INTERMEDIATE_MASK;
            else                  _flags -= IS_INTERMEDIATE_MASK;
        }
    }

    /*
    ---------------------------------------------
        §(2.1) : SOURCE LOCATION (DEVICE)  :
    ---------------------------------------------
    */

    /**
     *  This method informs this tensor if it's data is supposed to be kept in RAM
     *  or if it has already been migrated somewhere else.
     *  In the latter case, the tensor will nullify the reference to it's
     *  underlying data array to make it elegable for garbage collection.
     *  Otherwise, if {@code isOutsourced} is set to true, the method might
     *  allocate a new data array if none is present.
     *
     * @param isOutsourced The truth value which determines if this tensor should live in RAM or somewhere else.
     * @return This very instance to allow for method chaining.
     */
    public Tsr<V> setIsOutsourced( boolean isOutsourced ) {
        _setIsOutsourced( isOutsourced );
        if ( isOutsourced )
            _setData( null );
        else if (
            !forComponent(
                Device.class,
                device -> {
                    try {
                        if ( device.has( this ) ) device.restore( this );
                    } catch ( Exception exception ) {
                        _LOG.error(
                            "Tensor could not be restored from device component when trying to migrate it back to RAM.",
                            exception
                        );
                        throw exception;
                    }
                    this.remove( Device.class );
                    forComponent(
                        Tsr.class,
                        gradient ->
                            ( (Tsr<V>) gradient ).forComponent(
                                Device.class,
                                gradDevice -> {
                                    try {
                                        if ( gradDevice.has( gradient ) ) gradDevice.restore( gradient );
                                    }
                                    catch ( Exception exception ) {
                                        _LOG.error(
                                                "Gradient could not be restored from device component when trying to migrate it back to RAM.",
                                                exception
                                        );
                                        throw exception;
                                    }
                                    gradient.remove( Device.class );
                                })
                    );
                }
            ) && getData() == null
        ) {
            _setIsVirtual( true );
            _allocate( 1 ); // Only a single value representing the rest.
        }
        return this;
    }

    /**
     *  Outsourced means that the tensor is stored on a {@link Device} implementation instance.
     *
     * @return The truth value determining if the data of this tensor is not actually stored inside of it
     *         in the form of of a traditional primitive JVM array!
     */
    public boolean isOutsourced() { return ( _flags & IS_OUTSOURCED_MASK ) == IS_OUTSOURCED_MASK; }

    protected void _setIsOutsourced( boolean isOutsourced ) {
        if ( isOutsourced() != isOutsourced ) {
            if ( isOutsourced ) _flags += IS_OUTSOURCED_MASK;
            else                _flags -= IS_OUTSOURCED_MASK;
        }
    }

    /*
    --------------------------------------------
        §(2.2) : VIRTUAL / ACTUAL  :
    --------------------------------------------
    */

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
     *
     * @param isVirtual The truth value determining if this tensor ought to be virtualized.
     * @return This very tensor to enable method chaining.
     */
    @Override
    public Tsr<V> setIsVirtual( boolean isVirtual ) {

        assert getNDConf() != null;

        if ( isVirtual() != isVirtual ) {
            // Currently, we avoid offloading the virtualization by restoring outsourced tensors into RAM...
            Device<V> device = this.get( Device.class );
            try {
                if ( device != null ) device.restore( this );
            } catch ( Exception exception ) {
                _LOG.error(
                        "Tensor could not be restored from device component when changing flag 'isVirtual' to " + isVirtual + ".",
                        exception
                );
                throw exception;
            }
            if ( isVirtual ) {
                if ( getData() != null ) _virtualize();
            }
            else _actualize();
            // Virtual and actual tensors require a different mapping from a given index to the underlying data..
            // Therefore, we need to re-initialize the NDConfiguration object:
            createConstructionAPI().configureFromNewShape( getNDConf().shape(), isVirtual, getData() == null );
            if ( isVirtual ) {
                Relation<V> relation = get( Relation.class );
                if ( relation!=null )
                    relation.foreachChild( c -> {
                                c._setData( getData());
                                c.setIsVirtual( true );
                            });
            } else {
                Tsr<?> parentTensor = ( this.isSlice() ) ? get(Relation.class).getParent() : null;
                if ( parentTensor != null ) parentTensor.get( Relation.class ).remove( this );
            }

            try {
                if ( device != null ) device.store( this );
            } catch ( Exception exception ) {
                String message =
                        "Tensor could not be migrated back to host device after changing flag 'isVirtual' to "+isVirtual+".";
                _LOG.error(
                        message,
                        exception
                );
                throw new IllegalStateException( message );
            }
        }
        else if ( isVirtual && getData() == null ) _allocate( 1 ); //> Only a single value representing the rest.
        return this;
    }

    /**
     *  A tensor is virtual if the size of the underlying data is not actually equal to
     *  the number of elements which the tensor claims to store, aka its size.
     *  This is for example the case when initializing a tensor filled with a single
     *  value continuously. In that case the tensor will flag itself as virtual and only allocate the
     *  underlying data array to hold a single item even though the tensor might actually hold
     *  many more items.
     *  The reasons for this feature is that it greatly improves performance in certain cases.
     *  In essence this feature is a form of lazy loading.
     *  <br><br>
     * @return The truth value determining if this tensor is virtual (and therefore not "actual").
     */
    @Override
    public boolean isVirtual() { return ( _flags & IS_VIRTUAL_MASK ) == IS_VIRTUAL_MASK; }

    /**
     *  This method is the inner counterpart to the public "{@link Tsr#setIsVirtual}" method.
     *  It actually performs the bit flipping by applying the corresponding bit mask. <br>
     *  <br>
     * @param isVirtual The truth value which ought to be applied.
     */
    @Override
    protected void _setIsVirtual( boolean isVirtual ) {
        if ( isVirtual() != isVirtual ) {
            if ( isVirtual ) _flags += IS_VIRTUAL_MASK;
            else             _flags -= IS_VIRTUAL_MASK;
        }
    }

    /*
    --------------------------------------------
        §(2.3) : GRADIENT APPLY REQUIREMENT  :
    --------------------------------------------
    */

    /**
     *  This flag works alongside two autograd features which can be enables inside the library settings.
     *  They will come into effect when flipping their feature flags, <br>
     *  namely: <i>'isApplyingGradientWhenRequested'</i> and <i>'isApplyingGradientWhenTensorIsUsed'</i><br>
     *  As the first flag name suggests gradients will be applied to their tensors when it is set to true,
     *  however this will only happened when the second flag is set to true as well, because otherwise gradients
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
    public Tsr<V> setGradientApplyRequested(boolean applyRequested ) {
        if ( gradientApplyRequested() != applyRequested ) {
            if ( applyRequested ) {
                if (
                        Neureka.get().settings().autograd().isApplyingGradientWhenRequested() &&
                                !Neureka.get().settings().autograd().isApplyingGradientWhenTensorIsUsed()
                )
                    this.applyGradient();
                else
                    _flags += GRADIENT_APPLY_RQD_MASK;
            }
            else _flags -= GRADIENT_APPLY_RQD_MASK;
        }
        return this;
    }

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
    public boolean gradientApplyRequested() { return ( _flags & GRADIENT_APPLY_RQD_MASK ) == GRADIENT_APPLY_RQD_MASK; }

    /*
    --------------------------------------------
        §(2.4) : DELETION  :
    --------------------------------------------
    */

    /**
     *  This will check if the {@link #_delete()} method was previously called on this tensor.
     *  This means that any references inside the tensor will be null
     *  as well as that the tensor data was freed on every device,
     *  meaning that what was previously referenced was most likely garbage collected...
     *
     * @return The truth value which determines if {@link #_delete()} was called on this tensor,
     *         making it in essence an empty shell void of any references to data.
     */
    public boolean isDeleted() { return ( _flags & IS_DELETED_MASK ) == IS_DELETED_MASK; }

    /**
     *  Although tensors will be garbage collected when they are not strongly referenced,
     *  there is also the option to manually free up the tensor and its associated data.
     *  This is especially useful when tensors are stored on a device like the OpenCLDevice.
     *  In that case calling the "{@link Tsr#_delete()}" method will free the memory reserved for this tensor.
     *  This manual memory freeing through this method can be faster than waiting for
     *  the garbage collector to kick in... <br>
     *  <br>
     *
     * @return This very tensor instance to allow for method chaining.
     */
    private Tsr<V> _delete()
    {
        if ( isDeleted() ) return this;
        forComponent( GraphNode.class, n -> {
            if ( n.isUsedAsDerivative() ) {
                String message = "Cannot delete a tensor which is used as derivative by the AD computation graph!";
                _LOG.error( message );
                throw new IllegalStateException( message );
            }
        });
        forComponent( Device.class, device -> device.free( this ) );
        _setData( null );
        _setNDConf( null );
        _flags = 0;
        forComponent( Tsr.class, t -> t.getUnsafe().delete() );
        _deleteComponents();
        _flags += IS_DELETED_MASK;

        return this;
    }

    /*==================================================================================================================
    |
    |       §(3) : COMPONENT SYSTEM
    |   --------------------------------
    */
    /*
    --------------------------------------------
        §(3.0) : SETTING / REJECTING  :
    --------------------------------------------
    */

    /**
     * This method is executed when a new Component is added to the tensor.
     * The public add method is implemented in the super class
     * '{@link AbstractComponentOwner}' from which this class inherits.
     * In this super class the component logic is implemented.
     *
     * @param newComponent A component used to access features. ({@link GraphNode}, {@link NDFrame}, {@link Relation}, int[], ...)
     * @return The unchanged object or maybe in future versions: null (component rejected)
     */
    @Override
    protected < T extends Component<Tsr<V>> > T _setOrReject( T newComponent )
    {
        return newComponent;
    }

    /*
    --------------------------------------------
        §(3.1) : REMOVING / REJECTING  :
    --------------------------------------------
    */
    /**
     * This method is executed when a component is being removed from the tensor.
     * The public remove method is implemented in the super class
     * '{@link AbstractComponentOwner}' from which this class inherits.
     * In this super class the component logic is implemented.
     *
     * @param newComponent A component used to access features. ({@link GraphNode}, {@link NDFrame}, {@link Relation}, int[], ...)
     * @return The unchanged object or when rejected: null (component rejected)
     */
    @Override
    protected <T extends Component<Tsr<V>>> T _removeOrReject(T newComponent )
    {
        if ( newComponent instanceof Device ) {
            Device<V> device = (Device<V>) newComponent;
            /*
                The following seems like a redundant check, however often times a tensor
                will be removed from a Device implementation inside the "restore" method
                when the tensor has already been removed from the device...
                With out the condition below a stack overflow would occur!
             */
            if ( device.has( this ) ) {
                try {
                    device.restore( this );
                } catch ( Exception exception ) {
                    _LOG.error(
                            "Removing device from tensor / tensor from device failed.\n" +
                            "Restoring tensor from device threw exception.\n",
                            exception
                    );
                    throw exception;
                }
            }
        }
        return newComponent;
    }

    /*
    ----------------------------
        §(3.2) : UPDATING  :
    ----------------------------
    */
    /**
     *  Important : Components of type {@link Tsr} are simply gradients!
     *  Currently this method is used only to catch illegal arguments which
     *  is for example the case when trying to attach a gradient with a different shape...
     *  (Otherwise the gradient tensor "does not mind" an owner change...)
     */
    @Override
    public boolean update( OwnerChangeRequest<Tsr<V>> changeRequest ) {
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
        changeRequest.executeChange();
        // If the change request type is set to "REPLACED" then
        // this is means that this tensor is a gradient that is being
        // transferred to another tensor to serve as gradient...
        // No update task needs to occur. (This might change in the future...)
        return true;
    }

    /**
     * This method taked a {@link Device} and tries to migrate the contents of this {@link Tsr}
     * instance to that {@link Device}!
     *
     * @param device The {@link Device} which should host this {@link Tsr} as well as be added to its components list.
     * @return This very class to enable method chaining.
     */
    public Tsr<V> to( Device<?> device ){ super._set( device ); return this; }

    /*==================================================================================================================
    |
    |       §(4) : PROPERTIES :
    |   ---------------------------------------
    */
    /*
    --------------------------------------------
        §(4.0) : HIGH LEVEL PROPERTIES  :
    --------------------------------------------
    */

    /**
     *  A tensor is empty if there is neither data referenced within the tensor directly
     *  or within any given device to which the tensor might belong.
     *
     * @return The truth value determining if this tensor has data.
     */
    public boolean isEmpty() { return getData() == null && !this.isOutsourced(); }

    /**
     *  A tensor is "undefined" if it has either no {@link NDConfiguration} implementation instance
     *  or this instance does not have a shape set for this {@link Tsr} which is needed for
     *  a tensor to also have a rank and dimensionality...
     *
     * @return The truth value determining if this tensor has an {@link NDConfiguration} stored internally.
     */
    public boolean isUndefined() { return getNDConf() == null || getNDConf().shape() == null; }

    /**
     *  If this tensor is a slice of a parent tensor then this method will yield true.
     *  Slices can be created by calling the variations of the "{@link Tsr#getAt}" method.
     *
     * @return The truth value determining if this tensor is a slice of another tensor.
     */
    public boolean isSlice() {
        Relation<V> child = get( Relation.class );
        return ( child != null && child.hasParent() );
    }

    /**
     *  This method returns the number of slices which have been
     *  created from this very tensor.
     *  It does so by accessing the {@link Relation} component if present
     *  which internally keeps track of slices via weak references.
     *
     * @return The number of slices derived from this tensor.
     */
    public int sliceCount() {
        Relation<V> child = get( Relation.class );
        return ( child != null ) ? child.childCount() : 0;
    }

    /**
     *  If slices have been derived from this tensor then it is a "slice parent".
     *  This is what this method will determine, in which case, it will return true.
     *
     * @return The truth value determining if slices have been derived from this tensor.
     */
    public boolean isSliceParent() {
        Relation<V> parent = get( Relation.class );
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
    public boolean belongsToGraph() { return this.has( GraphNode.class ); }

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
    public boolean isLeave() { return (!this.has( GraphNode.class )) || this.get( GraphNode.class ).isLeave(); }

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
    public boolean isBranch() { return !this.isLeave(); }

    /**
     *  Tensors can be components of other tensors which makes the
     *  implicitly their gradients.
     *
     * @return The truth value determining if this tensor has another tensor attached to it (which is its gradient).
     */
    public boolean hasGradient() { return this.has( Tsr.class ); }

    /*
        ----------------------------------------------
            §(4.1) : COMPONENT BASED PROPERTIES :
        ----------------------------------------------
     */

    /**
     * @return The gradient of this tensor which is internally stored as component.
     */
    public Tsr<V> getGradient() { return this.get( Tsr.class ); }

    /**
     * @return The device on which this tensor is stored or {@link CPU} if it is not outsourced.
     */
    public Device<V> getDevice() {
        if ( this.isOutsourced() ) return this.get( Device.class );
        return (Device<V>) _CPU;
    }

    /**
     * @return The graph node of the computation graph to which this tensor belongs or null if not part of a graph.
     */
    public GraphNode<V> getGraphNode() { _guardGet("graph node"); return get( GraphNode.class ); }

    /**
     * @return An instance of the {@link NDFrame} component if present.
     */
    public NDFrame<V> frame() { _guardGet("graph node"); return get( NDFrame.class ); }


    /*
        ---------------------------------------
            §(4.2) : INNER PROPERTIES :
        ---------------------------------------
     */


    /*==================================================================================================================
    |
    |       §(5) : OBJECT STATE MODIFICATION :
    |   ------------------------------------------
    */

    private void _toLayout(NDConfiguration.Layout layout ) {

        if ( layout == this.getNDConf().getLayout() ) return;

        Tsr<V> transposed = this.T().clone().detach();
        if ( !this.isVirtual() )
            IntStream.range(0,transposed.size())
                    .parallel()
                    .forEach( i -> this.setDataAt( i, transposed.getDataAt( i ) ) );

        NDConfiguration old = this.getNDConf();
        this._setNDConf(
            AbstractNDC.construct(
                    old.shape(),
                    layout.newTranslationFor(old.shape()),
                    old.translation(),
                    old.spread(),
                    old.offset(),
                    layout
            )
        );
    }

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
    private Tsr<V> _incrementVersionBecauseOf( ExecutionCall<?> call ) {
        if ( Neureka.get().settings().autograd().isPreventingInlineOperations() ) {
            _version++;
            GraphNode<?> node = get( GraphNode.class );
            if ( node != null && node.getPayloadReferenceVersion() != _version ) {
                if ( node.usesAD() || node.isUsedAsDerivative() ) {
                    String error = "Inline operation occurred on tensor which is part of a computation graph node with autograd support!\n" +
                                   "The following OperationType caused an internal version mismatch: '"+call.getOperation().getFunction()+"'";
                    _LOG.error( error );
                    throw new IllegalStateException( error );
                }
            }
        }
        return this;
    }

    /**
     *  In essence tensors are merely fancy wrapper for some form of array of any type... 
     *  This wrapper usually stays the same of a given data array.
     *  However, sometimes a tensor changes its identity, or rather the underlying
     *  data changes the wrapping tensor instance. <br>
     *  <br>
     * @param tensor The tensor whose identity should be stolen.
     * @return This very tensor instance in order to enable method chaining.
     */
    protected Tsr<V> _become( Tsr<V> tensor )
    {
        if ( tensor == null ) return this;
        _setDataType( tensor.getDataType() );
        _setData( tensor.getData() );
        _setNDConf( tensor.getNDConf() );
        _flags = tensor._flags;
        _transferFrom( tensor );
        tensor._setData( null );
        tensor._setDataType( null );
        tensor._setNDConf( null );
        tensor._flags = 0;
        return this;
    }



    /*==================================================================================================================
    |
    |       §(6) : ND-ITERATOR LOGIC :
    |   ---------------------------------------
    */

    /**
     * This method returns an iterator over the elements of this tensor. <br>
     *
     * @return An iterator over elements of type ValType.
     */
    @NotNull
    @Override
    public Iterator<V> iterator()
    {
        NDIterator _ndi = NDIterator.of( this );
        return new Iterator<V>()
        {
            private int _count = 0;
            private final int _size = size();

            @Override
            public boolean hasNext() { return _count != _size; }

            @Override
            public V next() {
                V value = getDataAt( _ndi.i() );
                _ndi.increment();
                _count ++;
                return value;
            }
        };
    }


    /*==================================================================================================================
    |
    |       §(7) : COMPONENT SPECIFIC :
    |   ---------------------------------------
    */
    /*
        -------------------------------
            §(7.0) : AUTO-GRAD :
        -------------------------------
        ... for more context see package 'autograd' ...
     */

    /**
     *  Tensors which are used or produced by the autograd system will have a {@link GraphNode} component attached to them.
     *  This is because autograd requires recording a computation graph for back-prop traversal.
     *  If this tensor is part of a computation graph then this method
     *  will traverse an error backward in the recorded history towards tensors which require
     *  the accumulation of gradients.
     *
     * @param error A tensor which is back-propagated to gradients. Must match the size og this tensor.
     * @return The tensor on which this method was called. (factory pattern)
     */
    public Tsr<V> backward( Tsr<V> error ) {
        if ( this.isOutsourced() ) {
            error = error.clone();
            error = error.to(this.getDevice());
        }
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
     * @return The tensor on which this method was called. (factory pattern)
     */
    public Tsr<V> backward( double value ) {
        backward( Tsr.of( this.getValueClass(), getNDConf().shape(), value ) );
        return this;
    }

    /**
     *  Tensors which are used or produced by the autograd system will have a {@link GraphNode} component attached to them.
     *  This is because autograd requires recording a computation graph for back-prop traversal.
     *  If this tensor is part of a computation graph then this method
     *  will traverse an error backward in the recorded history towards tensors which require
     *  the accumulation of gradients. <br>
     *  <br>
     *  This method assumes that the user wants to back-propagate
     *  an error of "1" having the same shape as
     *  this tensor.
     *
     * @return The tensor on which this method was called. (factory pattern)
     */
    public Tsr<V> backward()
    {
        backward( 1 ); // By default we back-propagate a base factor of 1.
        return this;
    }

    /**
     *  If this tensor owns a gradient tensor as component, then it can be applied by this method. <br>
     *  "Applying" a gradient to a tensor simply means adding the values inside the gradient element-wise
     *  to the owning host tensor via an inline operation. <br>
     */
    public void applyGradient()
    {
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
                    // Therefore we need to turn off the inline operation safety net:
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
     *  <b>This method detaches this tensor from its underlying computation-graph
     *  or simply does nothing if no graph is present.</b> <br>
     *  Nodes within a computation graph are instances of the "{@link GraphNode}" class which are also
     *  simple components of the tensors they represent in the graph. <br>
     *  Therefore, "detaching" this tensor from the graph simply means removing its {@link GraphNode} component.
     *
     * @return This very instance in order to allows for a more streamline usage of this method.
     */
    public Tsr<V> detach() { this.remove( GraphNode.class ); return this; }

    /*
        ----------------------------
            §(7.1) : FRAMING :
        ----------------------------
        ... for more context see package 'framing'...
     */

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
    public Tsr<V> label( String[][] labels )
    {
        _label( null, labels );
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
    public Tsr<V> label( String tensorName, String[][] labels )
    {
        _label( tensorName, labels );
        return this;
    }

    /**
     *  This private method is used by public {@link Tsr#label} methods as a single source of
     *  responsibility for performing the actual labeling based on the user input...
     *
     * @param tensorName The name of this tensor which will be stored in an {@link NDFrame} component.
     * @param labels The label / alias information which will also be stored in an {@link NDFrame} component.
     */
    private void _label( String tensorName, String[][] labels )
    {
        NDFrame<V> frame = get( NDFrame.class );
        if ( frame == null ) {
            frame = new NDFrame( this.rank(), tensorName );
            set(frame);
        }
        assert labels.length <= this.rank();
        for( int i = 0; i < labels.length; i++ ) {
            if ( labels[ i ] != null ) {
                AxisFrame<Integer, V> atAxis = frame.atAxis( i );
                for ( int ii = 0; ii < labels[ i ].length; ii++ ) {
                    if ( labels[ i ][ ii ] != null )
                        atAxis.atIndexAlias( labels[ i ][ ii ] ).setIndex( ii );
                }
            }
        }
    }

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
    public Tsr<V> label( List<List<Object>> labels )
    {
        NDFrame<V> frame = get( NDFrame.class );
        if ( frame == null ) set( new NDFrame( labels, null ) );
        return this;
    }

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
    public Tsr<V> label( String tensorName, List<List<Object>> labels )
    {
        NDFrame<V> frame = get( NDFrame.class );
        if ( frame == null ) set( new NDFrame<>( labels, tensorName ) );
        return this;
    }

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
    public Tsr<V> label( Map<Object, List<Object>> labels )
    {
        this.set( new NDFrame<>( labels, this, null ) );
        return this;
    }

    public Tsr<V> label( String tensorName, Map<Object, List<Object>> labels )
    {
        this.set( new NDFrame<>( labels, this, tensorName ) );
        return this;
    }

    /*==================================================================================================================
    |
    |       §(8) : (OVERLOADABLE) OPERATORS & OPERATIONS :
    |   -----------------------------------------------------
    |       ...for more context see package 'calculus'...
    |*/
    /*
        -----------------------------
            §(8.0) : OPERATORS :
        -----------------------------
     */

    /**
     *  The {@link #plus(Tsr)} method will produce the sum of
     *  two arrays with the same rank (or two ranks which can be made compatible with padding ones),
     *  where the left operand is this {@link Tsr}
     *  instance and the right operand is the tensor passed to the method.
     *  If the shapes of both of the involved tensors is identical then
     *  the result will be a regular elementwise addition.
     *  Otherwise the method will also be able to perform broadcasting, however only if
     *  for every pair of shape dimension the following is true:
     *  Either the dimensions have the same size or one of them has size 1. <br>
     *  Here is an example of 2 matching shapes: (1, 4, 1) and (3, 4, 1)       <br>
     *  And here is an example of a mismatch: (2, 4, 1) and (3, 4, 1)         <br>
     *
     * @param other The right operand of the addition.
     * @return The sum of this instance as the left and the passed {@link Tsr} instance as right operand.
     */
    public Tsr<V> plus( Tsr<V> other ) {
        return Neureka.get().backend().getAutogradFunction().plus().call( this, other );
    }

    public Tsr<V> plusAssign( Tsr<V> other ) {
        return Neureka.get().backend().getFunction().plusAssign().call( this, other );
    }

    /**
     *  The {@link #plus(double)} method will create a new {@link Tsr}
     *  with the provided double scalar added to all elements of this {@link Tsr}.
     *
     *  The shapes of this tensor is irrelevant as the provided value will
     *  simply be broadcastet to any possible shape.
     *
     * @param value The right operand of the addition.
     * @return The sum between this instance as the left and the passed double as right operand.
     */
    public Tsr<V> plus( double value ) { return plus( _of( this.shape(), value ) ); }

    /**
     *  The {@link #minus(Tsr)} method will perform subtraction on
     *  two arrays with the same rank (or two ranks which can be made compatible with padding ones),
     *  where the left operand is this {@link Tsr}
     *  instance and the right operand is the tensor passed to the method.
     *  If the shapes of both of the involved tensors is identical then
     *  the result will be a regular elementwise subtraction.
     *  Otherwise the method will also be able to perform broadcasting, however only if
     *  for every pair of shape dimension the following is true:
     *  Either the dimensions have the same size or one of them has size 1. <br>
     *  Here is an example of 2 matching shapes: (1, 4, 1) and (3, 4, 1)       <br>
     *  And here is an example of a mismatch: (2, 4, 1) and (3, 4, 1)         <br>
     *
     * @param other The right operand of the subtraction.
     * @return The difference between this instance as the left and the passed {@link Tsr} instance as right operand.
     */
    public Tsr<V> minus( Tsr<V> other ) {
        return Neureka.get().backend().getAutogradFunction().minus().call( this, other );
    }

    public Tsr<V> minus( V other ) {
        return minus(
                 Tsr.of((Class<V>)this.getDataType().getTypeClass())
                             .withShape(this.getNDConf().shape())
                             .all(other)
        );
    }

    public Tsr<V> minusAssign( Tsr<V> other ) {
        return Neureka.get().backend().getFunction().minusAssign().call( this, other );
    }

    public Tsr<V> minusAssign( V other ) {
        return minusAssign(
                Tsr.of((Class<V>)this.getDataType().getTypeClass())
                        .withShape(this.getNDConf().shape())
                        .all(other)
        );
    }

    public Tsr<V> negative() {
        return Neureka.get().backend().getAutogradFunction().neg().call( this );
    }

    /**
     *  The {@link #multiply(Tsr)} method is synonymous with the {@link #times(Tsr)} method.
     *  Both of which will produce the product of
     *  two arrays with the same rank (or two ranks which can be made compatible with padding ones),
     *  where the left operand is this {@link Tsr}
     *  instance and the right operand is the tensor passed to the method.
     *  If the shapes of both of the involved tensors is identical then
     *  the result will be a regular elementwise product.
     *  Otherwise the method will also be able to perform broadcasting, however only if
     *  for every pair of shape dimension the following is true:
     *  Either the dimensions have the same size or one of them has size 1. <br>
     *  Here is an example of 2 matching shapes: (1, 4, 1) and (3, 4, 1)       <br>
     *  And here is an example of a mismatch: (2, 4, 1) and (3, 4, 1)         <br>
     *
     * @param other The right operand of the multiplication.
     * @return The product of this instance as the left and the passed {@link Tsr} instance as right operand.
     */
    public Tsr<V> multiply( Tsr<V> other ) {
        return Neureka.get().backend().getAutogradFunction().mul().call( this, other );
    }

    public Tsr<V> multiply( V other ) {
        return multiply(
                           Tsr.of( (Class<V>) this.getDataType().getTypeClass() )
                               .withShape( this.getNDConf().shape() )
                               .all( other )
                        );
    }

    /**
     *  The {@link #times(Tsr)} method is synonymous to the {@link #multiply(Tsr)}.
     *  Both of which will produce the product of
     *  two arrays with the same rank (or two ranks which can be made compatible with padding ones),
     *  where the left operand is this {@link Tsr}
     *  instance and the right operand is the tensor passed to the method.
     *  If the shapes of both of the involved tensors is identical then
     *  the result will be a regular elementwise product.
     *  Otherwise the method will also be able to perform broadcasting, however only if
     *  for every pair of shape dimension the following is true:
     *  Either the dimensions have the same size or one of them has size 1. <br>
     *  Here is an example of 2 matching shapes: (1, 4, 1) and (3, 4, 1)       <br>
     *  And here is an example of a mismatch: (2, 4, 1) and (3, 4, 1)         <br>
     *
     * @param other The right operand of the multiplication.
     * @return The product of this instance as the left and the passed {@link Tsr} instance as right operand.
     */
    public Tsr<V> times( Tsr<V> other ) { return multiply( other ); }

    public Tsr<V> times( V other ) { return multiply( other ); }

    public Tsr<V> timesAssign( Tsr<V> other ) {
        return Neureka.get().backend().getFunction().mulAssign().call( this, other );
    }

    public Tsr<V> timesAssign( V other ) {
        return this.timesAssign( Tsr.of( this.getValueClass(), this.shape(), other ) );
    }


    public Tsr<V> multiply( double value ) { return multiply( _of( this.shape(), value ) ); }

    /**
     *  The {@link #div(Tsr)} method will produce the quotient of
     *  two arrays with the same rank (or two ranks which can be made compatible with padding ones),
     *  where the left operand is this {@link Tsr}
     *  instance and the right operand is the tensor passed to the method.
     *  If the shapes of both of the involved tensors is identical then
     *  the result will be a regular elementwise division.
     *  Otherwise the method will also be able to perform broadcasting, however only if
     *  for every pair of shape dimension the following is true:
     *  Either the dimensions have the same size or one of them has size 1. <br>
     *  Here is an example of 2 matching shapes: (1, 4, 1) and (3, 4, 1)       <br>
     *  And here is an example of a mismatch: (2, 4, 1) and (3, 4, 1)         <br>
     *
     * @param other The right operand of the division.
     * @return The quotient of this instance as the left and the passed {@link Tsr} instance as right operand.
     */
    public Tsr<V> div( Tsr<V> other ) {
        return Neureka.get().backend().getAutogradFunction().div().call( this, other );
    }

    public Tsr<V> div( double value ) {
        return div( _of( this.shape(), value ) );
    }

    public Tsr<V> divAssign( Tsr<V> other ) {
        return Neureka.get().backend().getFunction().divAssign().call( this, other );
    }

    /**
     *  The {@link #mod(Tsr)} method will produce the modulus of
     *  two arrays with the same rank (or two ranks which can be made compatible with padding ones),
     *  where the left operand is this {@link Tsr}
     *  instance and the right operand is the tensor passed to the method.
     *  If the shapes of both of the involved tensors is identical then
     *  the result will be a regular elementwise modulo operation.
     *  Otherwise the method will also be able to perform broadcasting, however only if
     *  for every pair of shape dimension the following is true:
     *  Either the dimensions have the same size or one of them has size 1. <br>
     *  Here is an example of 2 matching shapes: (1, 4, 1) and (3, 4, 1)       <br>
     *  And here is an example of a mismatch: (2, 4, 1) and (3, 4, 1)         <br>
     *
     * @param other The right operand of the modulo operation.
     * @return The modulus of this instance as the left and the passed {@link Tsr} instance as right operand.
     */
    public Tsr<V> mod( Tsr<V> other ) {
        return Neureka.get().backend().getAutogradFunction().mod().call( this, other );
    }

    public Tsr<V> mod( int other ) {
        return mod((Tsr<V>) Tsr.of(this.getNDConf().shape(), other));
    }

    public Tsr<V> rem( int other ) {
        return mod((Tsr<V>) Tsr.of(this.getNDConf().shape(), other));
    }

    public Tsr<V> modAssign( Tsr<V> other ) {
        return Neureka.get().backend().getFunction().modAssign().call( this, other );
    }

    /**
     *  The {@link #power(Tsr)} (Tsr)} method will produce the power of
     *  two arrays with the same rank (or two ranks which can be made compatible with padding ones),
     *  where the left operand is this {@link Tsr}
     *  instance and the right operand is the tensor passed to the method.
     *  If the shapes of both of the involved tensors is identical then
     *  the result will be a regular elementwise exponentiation.
     *  Otherwise the method will also be able to perform broadcasting, however only if
     *  for every pair of shape dimension the following is true:
     *  Either the dimensions have the same size or one of them has size 1. <br>
     *  Here is an example of 2 matching shapes: (1, 4, 1) and (3, 4, 1)       <br>
     *  And here is an example of a mismatch: (2, 4, 1) and (3, 4, 1)         <br>
     *
     * @param other The right operand, also known as exponent, of the exponentiation.
     * @return The power of this instance as the left and the passed {@link Tsr} instance as right operand.
     */
    public Tsr<V> power( Tsr<V> other ) {
        return Neureka.get().backend().getAutogradFunction().pow().call( this, other );
    }

    public Tsr<V> power( double value ) {
        return power( _of( this.shape(), value ) );
    }

    /**
     *  This method is synonymous to the {@link #power(Tsr)} method.
     */
    public Tsr<V> xor( Tsr<V> other ) {
        return Neureka.get().backend().getAutogradFunction().pow().call( this, other );
    }

    public Tsr<V> xor( double value ) {
        return xor( _of( this.shape(), value ) );
    }

    /*
        -----------------------------
            §(8.1) : OPERATIONS :
        -----------------------------
     */

    /**
     *  A method which returns a new {@link Tsr} instance which is a transposed twin of this instance.
     *
     * @return A new transposed tensor with the same underlying data as this tensor.
     */
    public Tsr<V> T() // Transposed!
    {
        if ( this.rank() == 1 ) return this;
        else if ( this.rank() == 2 ) {
            boolean wasIntermediate = this.isIntermediate();
            this.getUnsafe().setIsIntermediate(false);
            Tsr<V> result = Neureka.get().backend().getFunction().transpose2D().call(this);
            this.getUnsafe().setIsIntermediate(wasIntermediate);
            return result;
        }
        StringBuilder operation = new StringBuilder();
        for ( int i = rank() - 1; i >= 0; i-- ) operation.append( i ).append( i == 0 ? "" : ", " );
        operation = new StringBuilder( "[" + operation + "]:(I[ 0 ])" );
        return Function.of( operation.toString(), true ).call( this );
    }

    /**
     *  A method which returns a new {@link Tsr} instance which is a transposed twin of this instance.
     *  It is and alias method to the {@link #T()} method...
     *
     * @return A new transposed tensor with the same underlying data as this tensor.
     */
    public Tsr<V> getT() { // Transposed
        return this.T();
    }

    /**
     *  This method performs various operations by calling {@link Function} instances
     *  in order to ultimately calculate the mean value of all values
     *  of this very tensor!
     *  This scalar tensor is then returned.
     *
     * @return A scalar tensor which is the mean value of all values of this very tensor.
     */
    public Tsr<V> mean() {
        Functions functions = Neureka.get().backend().getAutogradFunction();
        Tsr<V> ones = (Tsr<V>) Tsr.of( this.getValueClass(), this.getNDConf().shape(), (Object) 1 );
        Tsr<V> sum = functions.conv().call( this, ones );
        if ( !ones.has(GraphNode.class) || !ones.getGraphNode().isUsedAsDerivative() )
            ones.getUnsafe().delete();
        if ( sum == null )
            throw new IllegalStateException(
                    "Failed to calculate sum using convolution! Shapes: "+
                    Arrays.toString(this.getNDConf().shape())+"x"+Arrays.toString(ones.getNDConf().shape())
            );
        Tsr<V> result = functions.div().call( sum, Tsr.of( this.getValueClass(), new int[]{1}, this.size() ) );
        sum.getUnsafe().delete();
        return result;
    }

    /**
     *  This method performs a convolutional based dot product between the last dimension of this tensor
     *  and the first dimension of the passed tensor.
     *
     * @param b The tensor which is the right part of the dot product operation.
     * @return A new tensor which is the dot product of this tensor and the passed one.
     */
    public Tsr<V> convDot(Tsr<V> b ) {
        Tsr<V> a = this;
        int[][] fitter = AbstractTensor.Utility.makeFit( a.getNDConf().shape(), b.getNDConf().shape() );
        boolean doReshape = false;
        for ( int i = 0; i < fitter[ 0 ].length && !doReshape; i++ ) if ( fitter[ 0 ][ i ] != i ) doReshape = true;
        for ( int i = 0; i < fitter[ 1 ].length && !doReshape; i++ ) if ( fitter[ 1 ][ i ] != i ) doReshape = true;
        if ( doReshape ) {
            a = Function.of( AbstractTensor.Utility.shapeString( fitter[ 0 ] ) + ":(I[ 0 ])" ).call( a );
            b = Function.of( AbstractTensor.Utility.shapeString( fitter[ 1 ] ) + ":(I[ 0 ])" ).call( b );
        }
        return Neureka.get()
                        .backend()
                        .getAutogradFunction()
                        .conv()
                        .call( a, b )
                        .dimtrim();
    }

    /**
     *  This method performs a dot product between the last dimension of this tensor
     *  and the first dimension of the passed tensor.
     *  However, currently this method can only handle matrices which means
     *  that it is functionally completely identical to the {@link #matMul(Tsr)} method.
     *
     * @param b The tensor which is the right part of the dot product operation.
     * @return A new tensor which is the dot product of this tensor and the passed one.
     */
    public Tsr<V> dot( Tsr<V> b ) {
        if ( this.rank() != 2 && b.rank() != 2 )
            throw new IllegalStateException("Not yet implemented!"); // This is not yet available in the backend!
        return this.matMul( b );
    }

    /**
     *  The {@link #matMul(Tsr)} method will produce the matrix product of
     *  two 2 dimensional arrays, where the left operand is this {@link Tsr}
     *  instaance and the right operand is the tensor passed to the method.
     *
     * @param b The right operand of the matrix multiplication.
     * @return The matrix product of this instance as the left and the passed {@link Tsr} instance as right operand.
     */
    public Tsr<V> matMul( Tsr<V> b ) {
        if ( this.rank() != 2 || b.rank() != 2 ) {
            String message = "Cannot perform matrix multiplication for tensors whose ranks are not both 2!\n" +
                             "Encountered ranks: " + this.rank() + ", " + b.rank() + ";";
            _LOG.error( message );
            throw new IllegalArgumentException( message );
        }
        return Neureka.get().backend().getAutogradFunction().matMul().call( this, b );
    }

    /**
     *  This method creates a new tensor sharing the same data and whose shape is trimmed.
     *  A trimmed shape is simply a shape without preceding and trailing ones. <br>
     *  For example the shape (1x4x1x2x1) would be trimmed to (4x1x2).
     *  The underlying operation does not perform a removal of redundant ones all together.
     *  Only ones at the start and the beginning will be removed.
     *  A scalar tensor will not be affected by this operation.
     *
     * @return A tensor with the same underlying data but possibly trimmed shape without preceding or trailing ones.
     */
    public Tsr<V> dimtrim() { return Neureka.get().backend().getAutogradFunction().dimTrim().call( this ); }

    /**
     *  This method name translates to the "in" keyword in Groovy!
     *  The same is true for the "contains" method in Kotlin.
     *  Both methods do the exact same thing, however they exist
     *  for better language support.
     *
     * @param t The tensor which will be checked.
     * @return The answer to the following question: Is the data of the provided tensor a subset of the data of this tensor?
     */
    public boolean isCase( Tsr<V> t ) {
        boolean[] found = { false };
        this.forComponent( Relation.class, r -> r.foreachChild( c -> {
                if ( c.equals( t ) ) found[ 0 ] = true;
            }));
        return found[ 0 ];
    }

    /**
     *  This method name translates to the "in" keyword in Kotlin!
     *  The same is true for the "isCase" method in Groovy.
     *  Both methods do the exact same thing, however they exist
     *  for better language support.
     *
     * @param t The tensor which will be checked.
     * @return The answer to the following question: Is the data of the provided tensor a subset of the data of this tensor?
     */
    public boolean contains( Tsr<V> t ) { return isCase( t ); }


    /*==================================================================================================================
    |
    |       §(9) : SLICING, INDEXING & INJECTING :
    |   -----------------------------------------------------
    |       ...for more context see package 'ndim.config'...
    */
    /*
        -----------------------------
            §(9.0) : SLICING :
        -----------------------------
     */

    /**
     *  The following method enables access to specific scalar elements within the tensor.
     *  The method name also translates to the subscript operator in Groovy.
     *
     * @param indices The index array of the element which should be returned.
     * @return An element located at the provided index.
     */
    @Override
    public Tsr<V> getAt( int... indices ) { return getAt( Arrays.stream( indices ).boxed().toArray() ); }

    /**
     *  The following method enables the creation of tensor slices which access
     *  the same underlying data (possibly from a different view).
     *  The method name also translates to the subscript operator in Groovy.
     *
     * @param args An arbitrary number of arguments which can be used for slicing.
     * @return A slice tensor created based on the passed keys.
     */
    @Override
    public Tsr<V> getAt( Object... args ) {
        List<Object> argsList = Arrays.asList( args );
        return getAt( argsList );
    }

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
     * @param rankToStrides A map where the keys define where axes should be sliced and values which define the strides for the specific axis.
     * @return A tensor slice with an offset based on the provided map keys and
     *         strides based on the provided map values.
     */
    @Override
    public Tsr<V> getAt( Map<?,Integer> rankToStrides )
    {
        if ( rankToStrides == null ) return this;
        // ...not a simple slice... Advanced:
        return SmartSlicer.slice(
                        new Object[]{rankToStrides},
                        this,
                        this::_sliceOf
                    );
    }

    public Tsr<V> shallowCopy()
    {
        if ( this.isEmpty() || this.isUndefined() ) return this;
        List<List<Integer>> ranges = new ArrayList<>();
        for ( int e : this.shape() ) {
            List<Integer> rangeAsList = new ArrayList<>();
            for ( int i = 0; i < e; i++ ) rangeAsList.add( i );
            ranges.add( rangeAsList);
        }
        return getAt( ranges.toArray() );
    }

    /**
     *  This method enables tensor slicing!
     *  It takes a key of various types and configures a slice
     *  tensor which shares the same underlying data as the original tensor.
     *
     * @param key This object might be a wide range of objects including maps, lists or arrays...
     * @return A slice tensor or scalar value.
     */
    @Override
    public Tsr<V> getAt( Object key ) {
        if ( key == null ) return this;
        if ( key instanceof Object[] && ((Object[]) key).length == 0 ) key = new ArrayList<>();
        if ( key instanceof List && ( (List<?>) key ).isEmpty() ) {
            /*
                An empty List instance is being interpreted as
                the request to create an identical slice, meaning that the
                resulting tensor views the same data as its parent while not
                being the same instance. (In a sense, its a shallow copy!)
             */
            return shallowCopy();
        }

        key = ( key instanceof List ? ((List<?>) key).toArray() : key );

        if ( key instanceof Object[] ) {
            boolean allInt = true;
            for ( Object o : (Object[]) key ) allInt = allInt && o instanceof Integer;
            if ( allInt && ( (Object[]) key ).length == rank() ) {
                int[] newOffset = _intArray((Object[]) key);
                for ( int i = 0; i < this.rank(); i++ )
                    newOffset[ i ] = ( newOffset[ i ] < 0 ) ? getNDConf().shape( i ) + newOffset[ i ] : newOffset[ i ];
                for ( int i = 0; i < this.rank(); i++ )
                    ((Object[])key)[ i ] = newOffset[ i ];
                allInt = false;
            }
            boolean hasScale = false;
            for ( Object o : (Object[]) key ) hasScale = hasScale || o instanceof Map;
            return SmartSlicer.slice(
                    ( allInt ? new Object[]{ _intArray( (Object[]) key ) } : (Object[]) key ),
                    this,
                    this::_sliceOf
            );
        } else {
            String message = "Cannot create tensor slice from key of type '" + key.getClass().getName() + "'!";
            _LOG.error( message );
            throw new IllegalArgumentException( message );
        }
    }

    /**
     *  If this tensor stores value types then this method will
     *  essentially produce a deep copy of this tensor.
     *  If the stored elements are reference types on the other hand,
     *  then the resulting clone may not be treated as a deep copy,
     *  especially if elements are mutable objects.
     *
     * @return A deep copy of this tensor.
     */
    @Override
    public Tsr<V> clone() {
        Function cloner = Neureka.get().backend().getFunction().idy();
        boolean thisIsIntermediate = this.isIntermediate();
        _setIsIntermediate( false );
        Tsr<V> clone = cloner.call( Tsr.of( this.getValueClass(), this.shape(), 0.0 ), this ).to( this.getDevice() );
        clone.getUnsafe().setIsIntermediate( thisIsIntermediate );
        _setIsIntermediate( thisIsIntermediate );
        return clone;
    }

    /**
     *  This method returns a {@link SliceBuilder} instance exposing a simple builder API
     *  which enables the configuration of a slice of the current tensor via method chaining.    <br>
     *  The following code snippet slices a 3-dimensional tensor into a tensor of shape (2x1x3)  <br>
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
    public SliceBuilder<V> slice() { return new SliceBuilder<>( this, this::_sliceOf ); }

    /**
     *  This method is where the creation of a slice occurs.
     *  When creating a slice via the {@link SliceBuilder} or simply by passing ranges in the form of
     *  arrays, lists or maps to a {@link Tsr#getAt}(...) method, then this method will be called eventually.
     *  The creation of a slice always requires information about the shape of the new slice
     *  its position within the original tensor and also the strides / steps.
     *
     * @param newShape The of the slice which ought to be created.
     * @param newOffset The position of the new slice within this tensor.
     * @param newSpread The spread / steps / strides of the slice within this tensor.
     * @return The newly created slice.
     */
    private Tsr<V> _sliceOf( int[] newShape, int[] newOffset, int[] newSpread )
    {
        this.setIsVirtual( false );
        Tsr<V> subset = new Tsr<>();
        subset._setDataType( this.getDataType() );
        subset._setData( this.getData() );
        int[] newTranslation = getNDConf().translation();
        int[] newIndicesMap = this.getNDConf().getLayout().newTranslationFor( newShape );

        for ( int i = 0; i < this.rank(); i++ )
            newSpread[ i ] = ( newSpread[i] == 0 ) ? 1 : newSpread[ i ];

        for ( int i = 0; i < newOffset.length; i++ )
            newOffset[ i ] = newOffset[ i ] + getNDConf().offset( i ); // Offset is being inherited!

        Tsr<?> rootTensor   = ( this.isSlice() ? get( Relation.class ).findRootTensor() : this );
        Tsr<?> parentTensor = ( this.isSlice() ? get( Relation.class ).getParent()      : this );
        /*
            The following code check the validity of the slice shape ranges with
            respect to the 'parentTensor' of this new slice.
         */
        if ( parentTensor.rank() != newShape.length || rootTensor != parentTensor ) {
            // TODO! This requires some more thought about how to check this!
            // THIS CASE HAS NOT YET BEEN THOUGHT TROUGH!
            _LOG.warn(
                    "Exceptional slice request detected. " +
                    "This type of tensor cannot yet be sliced. " +
                    "Please copy this tensor before slicing."
            );
        } else {
            /*
                1. We know that inside this else branch 'this' tensor is a first order slice!
                (So it is not a slice of a slice... reason : 'rootTensor == parentTensor' )

                2. There is however uncertainty about the 'true shape' of this parent tensor!
                Meaning : It might have been reshaped and could therefore be distorted with
                respect to the slice that is currently being prepared!
                -> This means we have to take this possible reshaping into account!
                Like so:

                The following uses an int array also called 'reshapeRelation'.
                This is simply the 'reshape array' which has been recorded inside the 'Relation' component
                by the 'Reshape' operation! ( Hopefully! :) ... custom shape operations need to consider this as well! )

                The following would occur when : "new Tsr<>(...).T().getAt(...);"
                Transposing a tensor performs an inline reshaping of an identical
                slice of the original tensor! Then again slicing this tensor
                via the 'getAt(...)' method leads us to a situation where
                the following variable is NOT NULL! :
             */
            int[] reshaped = ( this.isSlice() ) ? parentTensor.get( Relation.class ).getReshapeRelationFor( this ) : null;
            reshaped = ( reshaped != null ) ? Reshape.invert( reshaped ) : null;
            for ( int i = 0; i < parentTensor.rank(); i++ ) {
                int ii = ( reshaped != null ) ? reshaped[ i ] : i;
                int top = newOffset[ i ] + newShape[ i ];
                if ( top > parentTensor.shape( ii ) ) {
                    String message =
                            "Cannot create slice because ranges are out of the bounds of the targeted tensor.\n" +
                                    "At index '" + i + "' : offset '" + newOffset[ i ] + "' + shape '" + newShape[ i ] + "' = '" + top + "',\n" +
                                    "which is larger than the target shape '" + parentTensor.shape( ii ) + "' at the same index!";
                    Exception exception = new IllegalArgumentException( message );
                    _LOG.error( message, exception );
                    throw new IllegalArgumentException( exception );
                }
            }
        }

        subset._setNDConf(
                AbstractNDC.construct(
                        newShape,
                        newTranslation,
                        newIndicesMap,
                        newSpread,
                        newOffset,
                        NDConfiguration.Layout.ROW_MAJOR
                )
        );

        if ( this.isOutsourced() ) {
            Device<V> device = this.get( Device.class );
            device.store( subset, this );
            subset.setIsOutsourced( true );
        }
        if ( this.isVirtual() ) subset.setIsVirtual( true );
        subset.set( new Relation().addParent( this ) );
        Relation<V> parent = get( Relation.class );
        parent = ( parent != null ) ? parent : new Relation<>();
        parent.addChild( subset );
        this.set( parent );
        return subset;
    }


    /*
        -----------------------------
            §(9.1) : INJECTING :
        -----------------------------
     */

    /**
     *  This method enables injecting slices of tensor to be assigned into this tensor!
     *  It takes a key of various types which is used to configure a slice
     *  tensor sharing the same underlying data as the original tensor.
     *  This slice is then used to assign the second argument to it, namely
     *  the "value" argument.
     *
     * @param key This object is a list defining a targeted index or range of indices...
     * @return A slice tensor or scalar value.
     */
    @Override
    public Tsr<V> putAt( List<?> key, Tsr<V> value ) {
        _putAtCheckFor( value );
        Tsr<V> slice = ( key == null ) ? this : getAt( key );
        return _putAt( slice, value );
    }

    @Override
    public Tsr<V> putAt( int[] indices, V value ) {
        if ( indices == null )
            throw new IllegalArgumentException( "Provided indices are null!" );
        if ( indices.length > this.rank() ) {
            int[] correct = new int[rank()];
            System.arraycopy( indices, 0, correct, 0, indices.length );
            indices = correct;
        }
        Tsr<V> source = Tsr.of( this.getValueClass(), shape(), value );
        Tsr<V> slice = getAt( Arrays.stream( indices ).mapToObj( i -> i ).collect(Collectors.toList()) );
        Neureka.get().backend().getFunction().idy().call(slice, source);
        return this;
    }

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
    @Override
    public Tsr<V> putAt( Map<?,Integer> key, Tsr<V> value ) {
        _putAtCheckFor( value );
        Tsr<V> slice = ( key == null ) ? this : getAt( key );
        return _putAt( slice, value );
    }

    private void _putAtCheckFor( Tsr<?> value ) {
        if ( value.isEmpty() ) {
            String message = "Provided tensor is empty! Empty tensors cannot be injected.";
            _LOG.error( message );
            throw new IllegalArgumentException( message );
        }
    }

    private Tsr<V> _putAt( Tsr<V> slice, Tsr<V> value )
    {
        boolean valueIsDeviceVisitor = false;
        if ( slice.isOutsourced() && !value.isOutsourced() ) {
            Device<V> device = slice.get( Device.class );
            try {
                device.store( value );
            } catch ( Exception e ) {
                _LOG.error( "Trying to migrate target slice tensor to device failed.", e );
                throw e;
            }
            valueIsDeviceVisitor = true;
        }
        if ( this.isEmpty() && slice.isEmpty() || slice.size() != value.size() ) _become( value ); // TODO: Rethink this a little
        else Function.of( "I[ 0 ] <- I[ 1 ]", false ).call(  slice, value  );
        try {
            if ( valueIsDeviceVisitor ) value.get( Device.class ).restore( value );
        } catch ( Exception exception ) {
            _LOG.error( "Trying to migrate source tensor back to original location failed.", exception );
            throw exception;
        }
        return this;
    }

    /**
     *  A tensor ought to have some way to access its underlying data array.
     *  This method simple returns an element within this data array sitting at position "i".
     * @param i The position of the targeted item within the raw data array of the tensor.
     * @return The found object sitting at the specified index position.
     */
    @Override
    public V getDataAt( int i )
    {
        if ( this.isOutsourced() ) {
            Device<V> device = this.get( Device.class );
            if ( device instanceof OpenCLDevice ) {
                return (V)(Double)( (OpenCLDevice) device ).value64f( (Tsr<Number>) this, i );
            }
        }
        else if ( getData() instanceof float[] )  return (V)(Float)  ( (float[]) getData())[ i ];
        else if ( getData() instanceof double[] ) return (V)(Double) ( (double[]) getData())[ i ];
        else if ( getData() instanceof short[] )  return (V)(Short)  ( (short[]) getData())[ i ];
        else if ( getData() instanceof int[] )    return (V)(Integer)( (int[]) getData())[ i ];
        else if ( getData() instanceof byte[] )   return (V)(Byte)   ( (byte[]) getData())[ i ];
        else if ( getData() instanceof long[] )   return (V)(Long)   ( (long[]) getData())[ i ];
        else if ( getData() instanceof boolean[] )return (V)(Boolean)( (boolean[]) getData())[ i ];
        else if ( getData() instanceof char[] )   return (V)(Character)( (char[]) getData())[ i ];
        else return ( (V[]) getData())[ i ];
        return null;
    }

    /**
     *  A tensor ought to have some way to selectively modify its underlying data array.
     *  This method simply overrides an element within this data array sitting at position "i".
     * @param i The index of the data array entry which ought to be addressed.
     * @param o The object which ought to be placed at the requested position.
     * @return This very tensor in order to enable method chaining.
     */
    @Override
    public Tsr<V> setDataAt( int i, V o ) {
        _guardMod("data object");
        _setDataAt( i, o );
        return this;
    }


    /**
     *  A tensor ought to have some way to selectively modify its underlying value array.
     *  This method simply overrides an element within this value array sitting at position "i".
     * @param i The index of the value array entry which ought to be addressed.
     * @param o The object which ought to be placed at the requested position.
     * @return This very tensor in order to enable method chaining.
     */
    @Override
    public Tsr<V> setValueAt( int i, V o ) {
        _guardMod("data object");
        NDConfiguration ndc = this.getNDConf();
        _setDataAt( ndc.indexOfIndex( i ), o );
        return this;
    }

    private void _setDataAt( int i, V o ) {
        if ( getData() instanceof Object[] ) ( (Object[]) getData() )[ i ] = o;
        else if ( getData() instanceof float[]  ) ( (float[])  getData() )[ i ] = (float)  o;
        else if ( getData() instanceof double[] ) ( (double[]) getData() )[ i ] = (double) o;
        else if ( getData() instanceof int[]    ) ( (int[])    getData() )[ i ] = (int)    o;
        else if ( getData() instanceof long[]   ) ( (long[])   getData() )[ i ] = (long)   o;
        else if ( getData() instanceof short[]  ) ( (short[])  getData() )[ i ] = (short)  o;
        else if ( getData() instanceof byte[]   ) ( (byte[])   getData() )[ i ] = (byte)   o;
        else if ( getData() instanceof boolean[]) ( (boolean[])getData() )[ i ] = (boolean)o;
        else if ( getData() instanceof char[])    ( (char[])   getData() )[ i ] = (char)o;
    }

    /**
     * @param value The primitive double array whose value ought to be used to populate this tensor.
     */
    private void _setValue64( double[] value ) {
        if ( this.isOutsourced() ) this.get( Device.class ).write( this, value );
        else if ( getData() == null ) {
            _setDataType( DataType.of( F64.class ) );
            _setData( value );
        }
        else if ( getData() instanceof float[] )
            for ( int i = 0; i < value.length; i++ ) ( (float[]) getData())[ i ] = (float) value[ i ];
        else if ( getData() instanceof double[] )
            System.arraycopy(value, 0, getData(), 0, value.length);
    }

    /**
     * @param value The primitive float array whose value ought to be used to populate this tensor.
     */
    private void _setValue32( float[] value ) {
        if ( this.isOutsourced() ) this.get( Device.class ).write( this, value );
        else if ( getData() == null ) {
            _setDataType( DataType.of( F32.class ) );
            _setData( value );
        }
        else if ( getData() instanceof float[] )
            System.arraycopy(value, 0, getData(), 0, value.length);
        else if ( getData() instanceof double[] )
            for ( int i = 0; i < value.length; i++ ) ( (double[]) getData())[ i ] = value[ i ];
    }

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
    public Tsr<V> setValue( Object value )
    {
        if ( value instanceof float[] ) _setValue32( (float[]) value );
        else if ( value instanceof  double[] ) _setValue64( (double[]) value );
        else if ( value instanceof Float ) {
            this.setIsVirtual( true );
            if ( getData() instanceof float[] ) ( (float[]) getData())[ 0 ] = (Float) value;
            else ( (double[]) getData())[ 0 ] = ( (Float) value ).doubleValue();
        } else if ( value instanceof Double ) {
            this.setIsVirtual( true );
            if ( getData() instanceof double[] ) ( (double[]) getData())[ 0 ] = (Double) value;
            else ( (float[]) getData() )[ 0 ] = ( (Double) value ).floatValue();
        } else if ( value instanceof Integer ) {
            this.setIsVirtual( true );
            ( (int[]) getData() )[ 0 ] = (Integer) value;
        } else if ( value instanceof Long ) {
            this.setIsVirtual( true );
            ( (long[]) getData() )[ 0 ] = (Long) value;
        } else if ( value instanceof int[] ) {
            _setData( value );
            setIsVirtual( false );
        } else if ( value instanceof short[] ) {
            _setData( value );
            setIsVirtual( false );
        } else if ( value instanceof long[] ) {
            _setData( value );
            setIsVirtual( false );
        } else if ( value instanceof byte[] ) {
            _setData( value );
            setIsVirtual( false );
        } else if ( value instanceof Object[] ) {
            _setData( value );
            setIsVirtual( false );
        } else
            _LOG.warn(
                    "Failed to set value array of type '"+value.getClass().getSimpleName()+"'!"
            );
        return this;
    }

    public Object getValue() { // TODO : Make this what it is supposed to be!!! (returning a copy of the targeted data)
        if ( this.isOutsourced() ) {
            Device<V> device = get( Device.class );
            if ( device != null )
                return device.valueFor( this );
            else
                return getData();
        }
        else if ( !this.isVirtual() ) return getData();
        else return getDataType().actualize( getData(), this.size() );
    }

    /*==================================================================================================================
    |
    |       §(10) : Mapping :
    |   -----------------------------------------------------
    |       ...transformation and modification...
    */

    public <T> Tsr<T> mapTo(
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
                .use( (Tsr<Number>) this )
                .in( () -> {
                    Object data = getData();
                    DataConverter.ForTensor map = new DataConverter.ForTensor( this );
                    if ( data == null ) {
                        if ( this.isOutsourced() )
                            _LOG.error("Encountered an outsourced tensor! Only local tensors stored in RAM can be mapped.");
                        else
                            _LOG.error("Invalid tensor state encountered! Cannot map a tensor without data.");
                    }
                    Object newData = null;
                    String failMessage = "Conversion to type "+typeClass+" not yet supported.";
                    if ( Number.class.isAssignableFrom(typeClass) ) {
                        java.util.function.Function<Integer, Number> access = null;
                        if ( this.getValueClass() == Integer.class ) {
                            int[] sourceData = (int[]) this.getData();
                            access = (i -> (Number) mapper.apply((V) Integer.valueOf(sourceData[i])));
                        } else if (this.getValueClass() == Double.class) {
                            double[] sourceData = (double[]) this.getData();
                            access = (i -> (Number) mapper.apply((V) Double.valueOf(sourceData[i])));
                        } else if (this.getValueClass() == Float.class) {
                            float[] sourceData = (float[]) this.getData();
                            access = (i -> (Number) mapper.apply((V) Float.valueOf(sourceData[i])));
                        } else if (this.getValueClass() == Short.class) {
                            short[] sourceData = (short[]) this.getData();
                            access = (i -> (Number) mapper.apply((V) Short.valueOf(sourceData[i])));
                        } else if (this.getValueClass() == Byte.class) {
                            byte[] sourceData = (byte[]) this.getData();
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
                        if ( this.getValueClass() == Integer.class ) {
                            int[] sourceData = (int[]) this.getData();
                            access = (i -> mapper.apply((V) Integer.valueOf(sourceData[i])));
                        } else if ( this.getValueClass() == Double.class ) {
                            double[] sourceData = (double[]) this.getData();
                            access = (i -> mapper.apply((V) Double.valueOf(sourceData[i])));
                        } else if ( this.getValueClass() == Float.class ) {
                            float[] sourceData = (float[]) this.getData();
                            access = (i -> mapper.apply((V) Float.valueOf(sourceData[i])));
                        } else if ( this.getValueClass() == Short.class ) {
                            short[] sourceData = (short[]) this.getData();
                            access = (i -> mapper.apply((V) Short.valueOf(sourceData[i])));
                        } else if ( this.getValueClass() == Byte.class ) {
                            byte[] sourceData = (byte[]) this.getData();
                            access = (i -> mapper.apply((V) Byte.valueOf(sourceData[i])));
                        } else
                            throw new IllegalArgumentException(failMessage);

                        newData = map.toObjectArray(access);
                    }
                    return Tsr.of( typeClass, this.getNDConf().shape(), newData );
                });
    }

    /**
     *  Turns this tensor into a {@link BufferedImage} based on the provided
     *  {@link ImageType} formatting choice.
     *
     * @param type The type of format used to create the buffered image.
     * @return A {@link BufferedImage} populated with the contents of this tensor.
     */
    public BufferedImage asImage( ImageType type )
    {
        if ( type.bufferType == BufferedImage.TYPE_3BYTE_BGR )
        {
            BufferedImage buffi = new BufferedImage( shape(1), shape(0), BufferedImage.TYPE_3BYTE_BGR );
            byte[] data = DataConverter.instance().convert(getData(), byte[].class);
            writeImgData(new DataBufferByte(data, data.length), buffi);
            return buffi;
        }
        throw new IllegalArgumentException("Image type '"+type+"' not supported.");
    }

    private static void writeImgData( DataBuffer data, BufferedImage target ) {
        target.setData(
            Raster.createRaster( target.getSampleModel(), data, new Point() )
        );
    }

    /**
     *  This method takes the provided {@link Tsr} instance and adds its
     *  contents to the contents of the {@link Tsr} which is set as gradient of this very {@link Tsr}.
     *
     * @param error The error gradient which ought to be added to the gradient of this tensor.
     * @return This very tensor instance to enable method chaining.
     */
    public Tsr<V> addToGradient( Tsr<V> error ) {
        _guardSet("gradient");
        if (
                !forComponent(
                    Tsr.class,
                        gradient ->
                        this.set(
                            MemUtil.keep( gradient, error, () ->
                                Neureka.get()
                                        .backend()
                                        .getFunction()
                                        .plusAssign()
                                        .call(gradient, error)
                            )
                        )
                )
        ) set( error ).forComponent( Device.class, device -> {
            try {
                device.store( error ) ;
            } catch ( Exception exception ) {
                _LOG.error( "Failed trying to store a given error to a device for gradient accumulation.", exception );
                throw exception;
            }
        });
        return this;
    }

    /**
     *  This method constitutes a pure operation producing a new tensor instance
     *  which is a deep copy of this original tensor and contains data whose
     *  elements have been converted to a new data type, namely :<br>
     *  The type specified by the argument <br>
     *  <br>
     *  The method does not change this tensor, which is why the operation is pure.
     *  Important to note is that the method will return instances of the specified
     *  type but merely another tensor containing elements of that type...
     *  The name of this method for example translates to the "as" operator
     *  found in Groovy, so the following code : <i>" myTensor as Double "</i> <br>
     *  would not return a Double instance!<br>
     *  <br>
     *
     * @param typeClass The class which is the target of the underlying type conversion...
     * @param <T> The value type of the tensor that will be returned.
     * @return A new tensor which hosting the supplied type.
     */
    public <T> Tsr<T> asType( Class<T> typeClass )
    {
        if ( typeClass == Tsr.class ) return (Tsr<T>) this.slice().get();
        DataType<?> newDT = DataType.of( typeClass );
        Object newData;
        if ( this.isOutsourced() ) {
            Device<V> device = get( Device.class );
            device.restore( this );
            newData = _convertedDataOfType( typeClass );
            device.store( this );
        }
        else newData = _convertedDataOfType( typeClass );
        return new Tsr<>( this.getNDConf().shape(), newDT, newData );
    }

    /**
     *  This method is an inline operation which changes the underlying data of this tensor.
     *  It converts the data types of the elements of this tensor to the specified type!<br>
     *  <br>
     *  <b>WARNING : The use of this method is discouraged for the following reasons: </b><br>
     *  <br>
     *  1. Inline operations are inherently error-prone for most use cases. <br>
     *  2. This inline operation in particular has no safety net,
     *     meaning that there is no implementation of version mismatch detection
     *     like there is for those operations present in the standard operation backend...
     *     No exceptions will be thrown during backpropagation! <br>
     *  3. This method has not yet been implemented to also handle instances which
     *     are slices of parent tensors!
     *     Therefore there might be unexpected performance penalties or side effects
     *     associated with this method.<br>
     *     <br>
     *
     * @param typeClass The target type class for elements of this tensor.
     * @param <T> The type parameter for the returned tensor.
     * @return The same tensor instance whose data has been converted to hold a different type.
     */
    private <T> Tsr<T> _toType(Class<T> typeClass )
    {
        if ( this.isOutsourced() ) {
            _setDataType( DataType.of( typeClass ) );
            return (Tsr<T>) this;
        }
        else {
            Object newData = _convertedDataOfType( typeClass );
            _setData( null );
            _setDataType( DataType.of( typeClass ) );
            _setData( newData );
        }
        forComponent( Tsr.class, gradient -> gradient._toType( typeClass ) );
        return (Tsr<T>) this;
    }

    public <A> A getValueAs( Class<A> arrayTypeClass ) {
        if ( arrayTypeClass == double[].class ) return (A) _value64();
        if ( arrayTypeClass == float[].class  ) return (A) _value32();
        if ( this.isVirtual() )
            return DataConverter.instance().convert(
                        getDataType().actualize( this.getData(), this.size() ),
                        arrayTypeClass
            );
        return (A) getData();
    }

    public <A> A getDataAs( Class<A> arrayTypeClass ) {
        if ( this.isOutsourced() ) {
            return getValueAs( arrayTypeClass );
        }
        return DataConverter.instance().convert( getData(), arrayTypeClass );
    }

    private double[] _value64() {
        Device<V> found = this.get( Device.class );
        if ( getData() == null && this.isOutsourced() && found != null ) {
            if ( found instanceof OpenCLDevice )
                return ( (OpenCLDevice) found).value64f( (Tsr<Number>) this );
            else return null;
        }
        double[] newValue = DataConverter.instance().convert( getData(), double[].class );

        if ( this.isVirtual() && newValue != null && this.size() > 1 ) {

           double[] value = new double[ this.size() ];
           Arrays.fill( value, newValue[ 0 ] );
           return value;
        }
        return newValue;
    }

    private float[] _value32() {
        Device<V> found = this.get( Device.class );
        if ( getData() == null && this.isOutsourced() && found != null ) {
            if ( found instanceof OpenCLDevice )
                return ( (OpenCLDevice) found ).value32f( (Tsr<Number>) this);
        }
        float[] newValue = DataConverter.instance().convert( getData(), float[].class );
        if ( this.isVirtual() && newValue != null ) {
            newValue = new float[ this.size() ];
            Arrays.fill( newValue, newValue[ 0 ] );
        }
        return newValue;
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    public String toString( String mode ) {
        return _toString( mode );
    }

    public String toString( TsrStringSettings config ) {
        if ( this.isDeleted() ) return "deleted";
        else if ( this.isEmpty() ) return "empty";
        else if ( this.isUndefined() ) return "undefined";
        return TsrAsString.representing( this ).withConfig( config ).toString();
    }

    /**
     *  This allows you to provide a lambda to configure how this tensor should be
     *  converted to {@link String} instances.
     *  The provided {@link Consumer} will receive a {@link TsrStringSettings} instance
     *  which allows you to change various settings with the help of method chaining.
     *
     * @param config A consumer of the {@link TsrStringSettings} ready to be configured.
     * @return The {@link String} representation of this tensor.
     */
    public String toString( Consumer<TsrStringSettings> config ) {
        if ( this.isDeleted() ) return "deleted";
        TsrStringSettings defaults = Neureka.get().settings().view().getTensorSettings().clone();
        config.accept(defaults);
        return TsrAsString.representing( this ).withConfig( defaults ).toString();
    }


    protected String _toString( String config )
    {
        if ( this.isDeleted() ) return "deleted";
        else if ( this.isEmpty() ) return "empty";
        else if ( this.isUndefined() ) return "undefined";
        return TsrAsString.representing( this ).withConfig( config ).toString();
    }

    @Override
    public String toString()
    {
        if ( this.isDeleted() ) return "deleted";
        else if ( this.isEmpty() ) return "empty";
        else if ( this.isUndefined() ) return "undefined";
        return TsrAsString.representing( this ).byDefaults().toString();
    }

    /**
     *  The version number is tracking how often this tensor has been mutated.
     *  This is especially useful for checking the correcting of autp-grad!
     */
    public int getVersion() { return _version; }

    /**
     *  Use this factory method to instantiate a new tensor with the same data type, shape
     *  and memory location ({@link Device} instance) as the provided template tensor.
     *
     * @param template The template tensor whose type, shape and location should be taken to construct a new tensor.
     * @param <V> The type parameter defining the value type of the provided as well as returned tensor.
     * @return A new {@link Tsr} instance with the same data type, shape and memory location as the provided template.
     */
    public static <V> IterByOrIterFromOrAll<V> like( Tsr<V> template ) {
        return Tsr.of( (Class<V>) template.getDataType().getJVMTypeClass() )
                    .on( template.getDevice() )
                    .withShape( template.getNDConf().shape() );
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
    public static <V> Tsr<V> ofRandom( Class<V> valueTypeClass, int... shape ) {
        long seed = 8701252152903546L;
        return Tsr.of( valueTypeClass )
                    .withShape( shape )
                    .andSeed( seed );
    }

    /**
     *  This method exposes an API for mutating the state of this tensor.
     *  The usage of methods exposed by this API is generally discouraged
     *  because the exposed state can easily lead to broken tensors and exceptions...<br>
     *  <br>
     *  Use this in performance critical situations only.
     */
    @Override
    public Unsafe<V> getUnsafe() {
        _guardGet("mutate");
        return new Unsafe<V>() {
            @Override
            public Tsr<V> setNDConf(NDConfiguration configuration ) { Tsr.this._setNDConf( configuration ); return Tsr.this; }
            @Override
            public <V> Tsr<V> toType( Class<V> typeClass ) { return Tsr.this._toType( typeClass ); }
            @Override
            public <V> Tsr<V> setDataType( DataType<V> dataType ) { return (Tsr<V>) Tsr.this._setDataType(dataType); }
            @Override
            public Tsr<V> toLayout(NDConfiguration.Layout layout) { Tsr.this._toLayout( layout ); return Tsr.this; }
            @Override
            public Tsr<V> incrementVersion(ExecutionCall<?> call ) {
                _incrementVersionBecauseOf( call );
                return Tsr.this;
            }
            @Override
            public Tsr<V> setIsIntermediate( boolean isIntermediate ) {
                _setIsIntermediate( isIntermediate );
                return Tsr.this;
            }
            @Override
            public Tsr<V> delete() {
                return Tsr.this._delete();
            }
        };
    }


    public enum ImageType {
        RGB_1INT(1),
        ARGB_1INT(2),
        ARGB_PRE_1INT(3),
        BGR_1INT(4),
        BGR_3BYTE(5),
        ABGR_4BYTE(6),
        ABGR_PRE_4BYTE(7),
        RGB_565_USHORT(8),
        RGB_555_USHORT(9),
        GRAY_BYTE(0),
        GRAY_USHORT(1);

        public final int bufferType;

        ImageType(int bufferType) {
            this.bufferType = bufferType;
        }
    }


}
