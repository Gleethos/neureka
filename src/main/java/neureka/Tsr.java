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
    It is a three-letter abbreviation of the word "tensor"!

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
*/

package neureka;

import groovy.lang.IntRange;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.experimental.Accessors;
import neureka.backend.api.ExecutionCall;
import neureka.backend.standard.operations.other.Reshape;
import neureka.devices.opencl.OpenCLDevice;
import neureka.dtype.DataType;
import neureka.dtype.custom.*;
import neureka.ndim.AbstractNDArray;
import neureka.devices.host.HostCPU;
import neureka.devices.Device;
import neureka.framing.IndexAlias;
import neureka.framing.Relation;
import neureka.calculus.Function;
import neureka.calculus.assembly.FunctionBuilder;
import neureka.autograd.GraphNode;
import neureka.autograd.JITProp;
import neureka.ndim.config.AbstractNDC;
import neureka.ndim.config.NDConfiguration;
import neureka.ndim.iterators.NDIterator;
import neureka.ndim.config.types.virtual.VirtualNDConfiguration;
import neureka.optimization.Optimizer;
import neureka.utility.DataConverter;
import neureka.utility.TsrAsString;
import org.jetbrains.annotations.NotNull;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Array;
import java.math.BigDecimal;
import java.util.*;
import java.util.stream.Collectors;


/**
 *  This class name "Tsr" is a 3 letter abbreviation of the term "tensor", a mathematical concept.
 *  A tensor is a type of multidimensional data-structure with certain transformation properties.
 *  Technically however, it is mostly a simple container / data-structure which can house data indexed by N dimensions.
 *  Therefore it is often also described as an nd-array.
 *  Elements of a tensor are also mostly numeric.
 *  This means that: <br>
 *  ...a tensor of rank 0 is a scalar, a tensor of rank 1 is a vector and a tensor of rank 2 is a matrix, etc...
 *  <br><br>
 *  Consequently, tensors are a perfect fit for applying various operations on them.
 *  Such operations might be simple elementwise operations or more complex linear operations like
 *  the dot-product, matrix- or even tensor multiplications.
 *
 * @param <ValueType>
 */
@Accessors( prefix = {"_"} )
public class Tsr<ValueType> extends AbstractNDArray<Tsr<ValueType>, ValueType> implements Component<Tsr<ValueType>>
{
    static {
        _CPU = HostCPU.instance();
        _LOG = LoggerFactory.getLogger( Tsr.class );
    }

    /**
     *  Default device (host cpu)
     */
    private static final Device<Number> _CPU;

    /**
     *  This field contains multiple flags.
     *  The bits of this integer are used to encode various states which a tensor can have.
     *  These bits are flipped by bitmasks which are defined below.
     */
    private int _flags = 0;

    /**
     *  This is a bit mask used to store true / false values
     *  in a targeted bit inside the "_flags" variable.
     */
    private static final int RQS_GRADIENT_MASK = 1;
    private static final int IS_OUTSOURCED_MASK = 2;
    private static final int IS_VIRTUAL_MASK = 4;
    private static final int GRADIENT_APPLY_RQD_MASK = 8;

    /**
     *  The version of the data ( _data ) stored within this tensor.
     *  It gets incremented every time an inline operation occurs!
     *  GraphNode instances tied to this tensor (as component) store
     *  a reference version which is a copy of this field.
     *  If this version changes, despite there being a GraphNode which might
     *  perform auto-differentiation at some point, then an exception will be thrown for debugging.
     *  <br>
     *  The getter returns the version of the data (_data) stored within this tensor.
     */
    @Getter
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

    public Tsr() {}

    public Tsr( Object arg ) {
        _construct( new Object[]{ arg } );
    }

    public Tsr( Object arg1, Object arg2, Object arg3 ) {
        _construct( new Object[]{ arg1, arg2, arg3 } );
    }

    public Tsr( Object arg1, Object arg2, Object arg3, String arg4 ) {
        _construct( new Object[]{ arg1, arg2, arg3, arg4 } );
    }

    public Tsr( Object arg1, Object arg2, Object arg3, Object arg4, Object arg5 ) {
        _construct( new Object[]{ arg1, arg2, arg3, arg4, arg5 } );
    }

    public Tsr( Object arg1, Object arg2, Object arg3, Object arg4, Object arg5, Object arg6 ) {
        _construct( new Object[]{ arg1, arg2, arg3, arg4, arg5, arg6 } );
    }

    public Tsr ( Object arg1, Object arg2, Object arg3, Object arg4, Object arg5, Object arg6, Object arg7 ) {
        _construct( new Object[]{arg1, arg2, arg3, arg4, arg5, arg6, arg7} );
    }

    public Tsr( Object arg1, Object arg2, Object arg3, Object arg4, Object arg5, Object arg6, Object arg7, Object arg8) {
        _construct( new Object[]{ arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8 } );
    }

    public Tsr( Object arg1, Object arg2, Object arg3, Object arg4, Object arg5, Object arg6, Object arg7, Object arg8, Object arg9 ) {
        _construct( new Object[]{ arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9 } );
    }

    public Tsr( Object arg1, Object arg2, Object arg3, Object arg4, Object arg5, Object arg6, Object arg7, Object arg8, Object arg9, Object arg10 ) {
        _construct( new Object[]{ arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10 } );
    }

    public Tsr( Object[] args )
    {
        _construct( args );
    }

    private void _construct( Object[] args )
    {
        if ( args == null || args.length == 0 ) return;
        if ( args.length == 1 ) {
            if ( args[ 0 ] instanceof Object[] ) {
                _construct( (Object[]) args[ 0 ] );
                return;
            } else if ( args[ 0 ] instanceof BigDecimal ) {
                _construct( new int[]{ 1 }, ( (BigDecimal) args[ 0 ] ).doubleValue());
                return;
            } else if ( args[ 0 ] instanceof Integer ) {
                _construct( new int[]{ 1 }, ( (Integer) args[ 0 ] ).doubleValue() );
                return;
            } else {
                String message = "Cannot create tensor from argument of type '" + args[ 0 ].getClass().getName() + "'!";
                _LOG.error( message );
                throw new IllegalArgumentException( message );
            }
        }
        args[ 0 ] = ( args[ 0 ] instanceof ArrayList ) ? ( (ArrayList) args[ 0 ] ).toArray() : args[ 0 ];
        args[ 1 ] = ( args[ 1 ] instanceof ArrayList ) ? ( (ArrayList) args[ 1 ] ).toArray() : args[ 1 ];
        if ( args[ 0 ] instanceof Object[] ) {
            if ( ( (Object[]) args[ 0 ] )[ 0 ] instanceof Integer || ((Object[])args[ 0 ])[ 0 ] instanceof Double) {
                args[ 0 ] = _intArray( (Object[]) args[ 0 ] );
            }
        }
        if ( args[ 1 ] instanceof Object[] ) {
            if ( ((Object[]) args[ 1 ] )[ 0 ] instanceof Integer ) args[ 1 ] = _doubleArray( (Object[]) args[ 1 ] );
            else if ( ( ( Object[] ) args[ 1 ] )[ 0 ] instanceof BigDecimal ) args[ 1 ] = _doubleArray( (Object[]) args[ 1 ] );
        }
        //CASES:
        if ( args[ 0 ] instanceof int[] ) {
            if ( args[ 1 ] instanceof Double || args[ 1 ] instanceof Integer ) {
                args[ 1 ] = ( args[ 1 ] instanceof Integer ) ? ( (Integer) args[ 1 ] ).doubleValue() : args[ 1 ];
                _construct( (int[]) args[ 0 ], (Double) args[ 1 ] );
                return;
            } else if ( args[ 1 ] instanceof double[] ) {
                _constructForDoubles( (int[]) args[ 0 ], (double[]) args[ 1 ] );
                return;
            } else {
                this.setDataType( DataType.of( args[1].getClass() ) );
                _construct( (int[]) args[0], true, true );
                ((Object[])getData())[0] = args[1];
                return;
            }
        }

        // EXPRESSION:
        boolean containsString = false;
        int numberOfTensors = 0;
        ArrayList<Tsr<ValueType>> tsrList = new ArrayList<>();
        for ( Object o : args ) {
            containsString = ( o instanceof String ) || containsString;
            if ( o instanceof Tsr ) {
                tsrList.add( (Tsr<ValueType>) o );
                numberOfTensors++;
            }
        }
        boolean doAD = true;
        Tsr<?>[] tsrs = new Tsr[ numberOfTensors ];
        StringBuilder f = new StringBuilder();
        int ti = 0;
        for ( Object o : args ) {
            if ( tsrList.contains( o ) ) {
                tsrs[ ti ] = ( (Tsr<?>) o );
                f.append( "I[" ).append( ti ).append( "]" );
                ti++;
            } else if ( o instanceof  String ) f.append( (String) o );
            else if ( o instanceof  Boolean ) doAD = (Boolean) o;
        }
        _construct( tsrs, f.toString(), doAD );
    }


    /*
        -------------------------------------------
            §(1.1) : SHAPE LIST BASED CONSTRUCTION
        --------------------------------------------
    */

    public Tsr( List<Integer> arg1, Object arg2 ) {
        _construct( new Object[]{ arg1, arg2 } );
    }

    public Tsr( List<?> arg1, String arg2 )
    {
        java.util.function.Function<Class<?>, Boolean> isType = c -> arg1.stream().allMatch( e -> e.getClass() == c );

        if ( isType.apply( Integer.class ) ) {
            List<Integer> shape = (List<Integer>) arg1;
            int[] shp = new int[ shape.size() ];
            for ( int i=0; i < shp.length; i++ ) shp[ i ] = shape.get( i );
            _construct( shp, arg2 );
        } else if ( isType.apply( Tsr.class ) ) {
            _construct( arg1.toArray( new Tsr[ 0 ] ), arg2, true );
        } else {
            _construct(
                    ( (List<Object>) arg1 ).stream().map( Tsr::new ).toArray( Tsr[]::new ),
                    arg2,
                    true
            );
        }
    }

    public Tsr( List<Integer> shape, List<ValueType> range )
    {
        // Nested Groovy list should be unpacked:
        if ( range.size() == 1 && range.get( 0 ) instanceof IntRange ) range = (List<ValueType>) range.get( 0 );
        _constructForRange(
                shape.stream().mapToInt( e -> e ).toArray(),
                DataType.of( F64.class ),
                (ValueType[]) range.toArray()
        );
    }

    public Tsr( int[] shape, List<ValueType> range )
    {
        // Nested Groovy list should be unpacked:
        if ( range.size() == 1 && range.get( 0 ) instanceof IntRange ) range = (List<ValueType>) range.get( 0 );
        _constructForRange(
                shape,
                DataType.of( F64.class ),
                (ValueType[]) range.toArray()
        );
    }

    private void _constructForRange( int[] shp, DataType<?> dataType, ValueType[] range ) {
        if ( range.length != 0 && !( range[ 0 ] instanceof Number ) ) {
            Class<?> givenClass = range[ 0 ].getClass();
            @SuppressWarnings("unchecked")
            final ValueType[] value = (ValueType[]) Array.newInstance(
                    givenClass,
                    NDConfiguration.Utility.szeOfShp( shp )
            );
            for ( int i = 0; i < value.length; i++ ) value[ i ] = range[ i % range.length ];
            setDataType( DataType.of( givenClass ) );
            _setData( value );
            _construct( shp, value );
        } else {
            setDataType( dataType );
            if ( dataType.getTypeClass() == F64.class )
                _constructForDoubles(
                        shp,
                        DataConverter.Utility.objectsToDoubles( range, NDConfiguration.Utility.szeOfShp( shp ) )
                );
            else if ( dataType.getTypeClass() == F32.class  )
                _constructForFloats(
                        shp,
                        DataConverter.Utility.objectsToFloats( range, NDConfiguration.Utility.szeOfShp( shp ) )
                );
            else if ( dataType.getTypeClass() == I32.class )
                _constructForInts(
                        shp,
                        DataConverter.Utility.objectsToInts( range, NDConfiguration.Utility.szeOfShp( shp ) )
                );
            else if ( dataType.getTypeClass() == I16.class )
                _constructForShorts(
                        shp,
                        DataConverter.Utility.objectsToShorts( range, NDConfiguration.Utility.szeOfShp( shp ) )
                );
        }
    }


    public Tsr( List<Object> conf ) {
        boolean isMatrix = conf.stream().allMatch( e -> e instanceof List );
        if ( isMatrix ) {
            _construct( conf.stream().map( e -> (List<Object>) e ).collect( Collectors.toList() ) );
            return;
        }
        boolean isNatural = ( conf.size() <= 64 );
        for( Object e : conf ) {
            if ( !isNatural ) break;
            double asNum = ( e instanceof BigDecimal ) ?
                    ( (BigDecimal) e ).doubleValue()
                    : ( e instanceof Double )
                    ? (Double) e
                    : (Integer) e;
            isNatural = asNum % 1 == 0;
        }
        if ( isNatural ) {
            int[] shape = new int[ conf.size() ];
            for ( int i = 0; i < shape.length; i++ ) {
                shape[ i ] = ( conf.get( i ) instanceof BigDecimal )
                        ? ( (BigDecimal) conf.get( i ) ).intValue() :
                        ( conf.get( i ) instanceof Double )
                                ? ( (Double) conf.get( i ) ).intValue()
                                :( (Integer) conf.get( i ) );
            }
            _construct( shape, false, false );
        } else {
            double[] value = new double[ conf.size() ];
            for ( int i = 0; i < value.length; i++ ) {
                value[ i ] = ( conf.get( i ) instanceof BigDecimal )
                        ? ( (BigDecimal) conf.get( i ) ).doubleValue() :
                        ( conf.get( i ) instanceof Double )
                                ? ( (Double) conf.get( i ) ).doubleValue()
                                : ( (Integer) conf.get( i ) );
            }
            _constructForDoubles( new int[]{ conf.size() }, value );
        }

    }


    /**
     *  This method receives a list of lists which represents a matrix of objects.
     *  It parses this matrix into a 2D shape array and a double array.
     *
     * @param matrix A list of list which ought to resemble a matrix.
     */
    private void _construct( List<List<Object>> matrix ) {
        boolean isNumeric = matrix.stream().allMatch( e -> e.stream().allMatch( ie -> ie instanceof Number ) );
        if ( isNumeric ) {
            int n = matrix.get( 0 ).size();
            boolean isHomogenous = matrix.stream().allMatch( e -> e.size() == n );
            if ( isHomogenous ) {
                int m = matrix.size();
                double[] value = new double[ m * n ];
                int[] shape = new int[]{ m, n };

                for ( int mi = 0; mi < m; mi++ ) {
                    for ( int ni = 0; ni < n; ni++ ) {
                        int i = n * mi + ni;
                        value[ i ] = DataConverter.instance().convert( matrix.get( mi ).get( ni ), Double.class );
                    }
                }
                _constructForDoubles( shape, value );
            } else {
                String message = "Provided nested list(s) do not form a regular matrix.";
                _LOG.error( message );
                throw new IllegalArgumentException( message );
            }
        }
    }

    /*
        -------------------------------------------
            §(1.2) : SHAPE ARRAY BASED CONSTRUCTION
        --------------------------------------------
    */

    public Tsr( double value )
    {
        _construct( new int[]{ 1 }, value );
    }

    public Tsr( float[] value )
    {
        _constructForFloats( new int[]{ value.length }, value );
    }

    public Tsr( int[] shape, String seed )
    {
        _construct( shape, seed );
    }

    public Tsr( int[] shape )
    {
        _construct( shape, true, true );
    }

    public Tsr( int[] shape, double value )
    {
        _construct( shape, value );
    }


    public Tsr( int[] shape, double[] value ) {
        _constructForDoubles( shape, value );
    }


    public Tsr( int[] shape, DataType<?> type )
    {
        setDataType( DataType.of( type.getTypeClass() ) );
        _construct( shape, true, true );
    }

    public Tsr( int[] shape, Class<?> typeClass, Object data )
    {
        setDataType( DataType.of( typeClass ) );
        _configureFromNewShape( shape, false, false );
        setValue( data );
    }

    public <T> Tsr( List<Integer> shape, Class<T> typeClass, List<T> data )
    {
        _constructForRange( shape.stream().mapToInt( e -> e ).toArray(), DataType.of( typeClass ), (ValueType[]) data.toArray());
    }

    public Tsr( int[] shape, DataType<?> dataType, Object data )
    {
        setDataType( dataType );
        _configureFromNewShape( shape, false, false );
        _setData( data );
    }

    public <T> Tsr( List<Integer> shape, DataType<T> dataType, List<T> data )
    {
        setDataType( dataType );
        _configureFromNewShape( shape.stream().mapToInt( e -> e ).toArray(), false, false );
        _setData( data );
    }

    // Inner construction layer:

    private void _construct( int[] shape, String seed )
    {
        _construct( shape, false, false );
        _setData( DataConverter.Utility.seededDoubleArray( (double[]) getData(), seed ) );
    }

    private void _construct( int[] shape, boolean allocate, boolean virtual )
    {
        if ( allocate ) _allocate( ( virtual ) ? 1 : NDConfiguration.Utility.szeOfShp( shape ) );
        if ( virtual ) setIsVirtual( true );
        _configureFromNewShape( shape, virtual, true );
    }

    private void _construct( int[] shape, double value )
    {
        int size = NDConfiguration.Utility.szeOfShp( shape );
        setDataType( DataType.of( F64.class ) );
        _allocate( 1 );
        setIsVirtual( size > 1 );
        _configureFromNewShape( shape, size > 1, true );
        ( (double[]) getData())[ 0 ] = value;
    }

    private void _constructForDoubles(int[] shape, double[] value )
    {
        int size = NDConfiguration.Utility.szeOfShp( shape );
        setDataType( DataType.of( F64.class ) );
        if ( size != value.length ) {
            _allocate( size );
            for ( int i = 0; i < size; i++ ) ( (double[]) getData())[ i ]  = value[ i % value.length ];
        } else _setData( value );
        _configureFromNewShape( shape, false, true );
    }

    private void _constructForFloats(int[] shape, float[] value )
    {
        int size = NDConfiguration.Utility.szeOfShp( shape );
        setDataType( DataType.of( F32.class ) );
        if ( size != value.length ) {
            _allocate( size );
            for ( int i = 0; i < size; i++ ) ( (float[]) getData())[ i ]  = value[ i % value.length ];
        } else _setData( value );
        _configureFromNewShape( shape, false, true );
    }

    private void _constructForInts( int[] shape, int[] value )
    {
        int size = NDConfiguration.Utility.szeOfShp( shape );
        setDataType( DataType.of( I32.class ) );
        if ( size != value.length ) {
            _allocate( size );
            for ( int i = 0; i < size; i++ ) ( (int[]) getData())[ i ]  = value[ i % value.length ];
        } else _setData( value );
        _configureFromNewShape( shape, false, true );
    }

    private void _constructForShorts( int[] shape, short[] value )
    {
        int size = NDConfiguration.Utility.szeOfShp( shape );
        setDataType( DataType.of( I16.class ) );
        if ( size != value.length ) {
            _allocate( size );
            for ( int i = 0; i < size; i++ ) ( (short[]) getData())[ i ]  = value[ i % value.length ];
        } else _setData( value );
        _configureFromNewShape( shape, false, true );
    }

    private void _construct( int[] shape, ValueType[] value ) {
        int size = NDConfiguration.Utility.szeOfShp( shape );
        if ( size != value.length ) {
            Class<?> givenClass = value[ 0 ].getClass();
            @SuppressWarnings("unchecked")
            final ValueType[] newValue = (ValueType[]) Array.newInstance(
                    givenClass,
                    NDConfiguration.Utility.szeOfShp( shape )
            );
            for ( int i = 0; i < newValue.length; i++ ) newValue[ i ] = value[ i % value.length ];
            setDataType( DataType.of( givenClass ) );
            _setData( newValue );
        }
        else _setData( value );
        _configureFromNewShape( shape, false, true );
    }

    private int[] _intArray( Object[] arg ) {
        int length = arg.length;
        int[] array = new int[ length ];
        for ( int i = 0; i < length; i++ ) {
            if ( arg[ i ] instanceof Double ) array[ i ] = ( (Double) arg[ i ] ).intValue();
            else array[ i ] = (Integer) arg[ i ];
        }
        return array;
    }

    private double[] _doubleArray( Object[] arg )
    {
        int length = arg.length;
        double[] array = new double[ length ];
        for ( int i = 0; i < length; i++ ) {
            if ( arg[ i ] instanceof Integer ) array[ i ] = (Integer) arg[ i ];
            else if ( arg[ i ] instanceof Double ) array[ i ] = (Double) arg[ i ];
            else if ( arg[ i ] instanceof BigDecimal ) array[ i ] = ( (BigDecimal) arg[ i ] ).doubleValue();
        }
        return array;
    }

    /*
        -------------------------------------------
            §(1.3) : LAMBDA BASED CONSTRUCTION
        --------------------------------------------
    */

    public interface Initializer<T> {  T init( int i, int[] index );  }

    public <T> Tsr( List<Integer> shape, DataType<T> type, Initializer<T> initializer )
    {
        _constructFromInitializer( shape.stream().mapToInt(e -> e ).toArray(), type, initializer );
    }

    public <T> Tsr( int[] shape, DataType<T> type, Initializer<T> initializer )
    {
        _constructFromInitializer( shape, type, initializer );
    }

    private <T> void _constructFromInitializer(int[] shape, DataType<T> type, Initializer<T> initializer )
    {
        setDataType( type );
        _construct( shape, true, false );
        Object data = getData();
        if ( data instanceof double[] )
            for ( int i=0; i<((double[])data).length; i++ )
                ( (double[]) data )[i] = (double) initializer.init( i, _NDConf.idx_of_i( i )  );
        else if ( data instanceof float[] )
            for ( int i=0; i<((float[])data).length; i++ )
                ( (float[]) data )[i] = (float) initializer.init( i, _NDConf.idx_of_i( i )  );
        else if ( data instanceof int[] )
            for ( int i=0; i<((int[])data).length; i++ )
                ( (int[]) data )[i] = (int) initializer.init( i, _NDConf.idx_of_i( i )  );
        else if ( data instanceof short[] )
            for ( int i=0; i<((short[])data).length; i++ )
                ( (short[]) data )[i] = (short) initializer.init( i, _NDConf.idx_of_i( i )  );
        else if ( data instanceof byte[] )
            for ( int i=0; i<((byte[])data).length; i++ )
                ( (byte[]) data )[i] = (byte) initializer.init( i, _NDConf.idx_of_i( i )  );
        else for ( int i=0; i<((Object[])data).length; i++ )
                ( (Object[]) data )[i] = initializer.init( i, _NDConf.idx_of_i( i )  );

    }


    /*
        -------------------------------------------
            §(1.4) : FUNCTION BASED CONSTRUCTION
        --------------------------------------------
     */


    public Tsr( String expression, List<Object> inputs )
    {
        if ( inputs.stream().allMatch( e -> e instanceof Tsr ) )
            _construct(
                    inputs.stream().toArray( Tsr[]::new ),
                    expression,
                    true
            );
        else
            _construct(
                    inputs.stream().map( Tsr::new ).toArray( Tsr[]::new ),
                    expression,
                    true
            );
    }

    /**
     *  This method takes a tensor and a String expression describing
     *  operations which ought to be applied to said tensor.
     *  This expression will be parsed to a Function instance expecting one input,
     *  namely : "I[0]"
     *  An example would be the following :
     *  'Tsr a = new Tsr( b, "sin( I[0] ) * 2" )'
     *  Which takes the tensor 'b' and applies the function "f(x) = sin(x) * 2"
     *  elementwise to produce a new tensor 'a'!
     *
     * @param tensor A tensor which serves as input to the Function instance parsed from the given expression.
     * @param expression The expression describing operations applied to the provided tensor.
     */
    public Tsr( Tsr<ValueType> tensor, String expression ) {
        if ( tensor == null ) return;
        _construct( new Tsr[]{ tensor }, expression, true );
    }

    /**
     *  This method takes an array of tensors and a String expression describing
     *  operations which ought to be applied to the tensors in said array.
     *  This expression will be parsed to a Function instance expecting as many inputs
     *  as there are array entries, namely : "I[0]", "I[1]", "I[2]", ...
     *  An example would be the following :
     *  'Tsr a = new Tsr( new Tsr[]{ b, c }, "sin( I[0] ) / I[1]" )'
     *  Which takes the tensor 'b' and 'c' and applies the function "f(x,y) = sin(x) / y"
     *  elementwise to produce a new tensor 'a'!
     *
     * @param tensors An array of tensors used as inputs to the Function instance parsed from the provided expression.
     * @param expression The expression describing operations applied to the provided tensors.
     */
    public Tsr( Tsr<ValueType>[] tensors, String expression ) {
        _construct( tensors, expression, true );
    }

    /**
     *  This method takes an array of tensors and a String expression describing
     *  operations which ought to be applied to the tensors in said array.
     *  This expression will be parsed to a Function instance expecting as many inputs
     *  as there are array entries, namely : "I[0]", "I[1]", "I[2]", ...
     *  An example would be the following :
     *  'Tsr a = new Tsr( new Tsr[]{ b, c }, "sin( I[0] ) / I[1]" )'
     *  Which takes the tensor 'b' and 'c' and applies the function "f(x,y) = sin(x) / y"
     *  elementwise to produce a new tensor 'a'!
     *  Additionally there is a helpful flag which allows one to specify if the
     *  parsed Function instance emerging from the provided expression
     *  should also allow the tracking of computations via a computation graph (GraphNode instances).
     *  This history tracking then enables auto-differentiation.
     *
     * @param tensors An array of tensors used as inputs to the Function instance parsed from the provided expression.
     * @param expression The expression describing operations applied to the provided tensors.
     * @param doAD A flag which when set to true commands the creation of a computation graph during operation execution.
     */
    public Tsr( Tsr<ValueType>[] tensors, String expression, boolean doAD )
    {
        _construct( tensors, expression, doAD );
    }

    private void _construct( Tsr[] tensors, String operation, boolean doAD ) {
        if ( tensors == null || tensors.length == 0 || tensors[ 0 ] == null ) return;
        Tsr<ValueType> result = Function.Setup.commit( this, tensors, operation, doAD );
        this._become( result );
    }

    /**
     *  This method is responsible for instantiating and setting the _conf variable.
     *  The core requirement for instantiating NDConfiguration interface implementation s
     *  is a shape array of integers which is being passed to the method...
     *
     * @param newShape An array if integers which are all greater 0 and represent the tensor dimensions.
     */
    protected void _configureFromNewShape( int[] newShape, boolean makeVirtual, boolean autoAllocate )
    {
        int size = NDConfiguration.Utility.szeOfShp( newShape );
        if ( size == 0 ) {
            String shape = Arrays.stream( newShape ).mapToObj( String::valueOf ).collect( Collectors.joining( "x" ) );
            String message = "The provided shape '"+shape+"' must not contain zeros. Dimensions lower than 1 are not possible.";
            _LOG.error( message );
            throw new IllegalArgumentException( message );
        }
        if ( getData() == null && autoAllocate ) _allocate( size );
        int length = _dataLength();
        if ( length >= 0 ) {
            if ( size != length && ( !this.isVirtual() || !makeVirtual) ) {
                String message = "Size of shape does not match stored value64!";
                _LOG.error( message );
                throw new IllegalArgumentException( message );
            }
        }
        if ( makeVirtual ) _NDConf = VirtualNDConfiguration.construct( newShape );
        else {
            int[] newTranslation = NDConfiguration.Utility.newTlnOf( newShape );
            int[] newIdxmap = newTranslation;
            int[] newSpread = new int[ newShape.length ];
            Arrays.fill( newSpread, 1 );
            int[] newOffset = new int[ newShape.length ];
            _NDConf = AbstractNDC.construct( newShape, newTranslation, newIdxmap, newSpread, newOffset );
        }
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

    public Tsr<ValueType> setRqsGradient( boolean rqsGradient ) {
        if ( rqsGradient() != rqsGradient && !rqsGradient ) this.remove( Tsr.class );
        _setRqsGradient( rqsGradient );
        return this;
    }

    public boolean rqsGradient() {
        return ( _flags & RQS_GRADIENT_MASK ) == RQS_GRADIENT_MASK;
    }

    protected void _setRqsGradient( boolean rqsGradient ) {
        if ( rqsGradient() != rqsGradient ) {
            if ( rqsGradient ) _flags += RQS_GRADIENT_MASK;
            else _flags -= RQS_GRADIENT_MASK;
        }
    }

    /*
    ---------------------------------------------
        §(2.1) : SOURCE LOCATION (DEVICE)  :
    ---------------------------------------------
    */

    public Tsr<ValueType> setIsOutsourced( boolean isOutsourced ) {
        _setIsOutsourced( isOutsourced );
        if ( isOutsourced ) {
            _setData( null );
        } else if (
                !forComponent(
                        Device.class,
                        d -> {
                            try {
                                if ( d.has( this ) ) d.restore( this );
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
                                            gradient.forComponent(
                                                    Device.class,
                                                    gd -> {
                                                        try {
                                                            if ( ( (Device) gd ).has( gradient ) ) ( (Device) gd ).restore( gradient );
                                                        } catch ( Exception exception ) {
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
            setIsVirtual( true );
        }
        return this;
    }

    public boolean isOutsourced() {
        return ( _flags & IS_OUTSOURCED_MASK ) == IS_OUTSOURCED_MASK;
    }

    protected void _setIsOutsourced( boolean isOutsourced ) {
        if ( isOutsourced() != isOutsourced ) {
            if ( isOutsourced ) _flags += IS_OUTSOURCED_MASK;
            else _flags -= IS_OUTSOURCED_MASK;
        }
    }

    /*
    --------------------------------------------
        §(2.2) : VIRTUAL / ACTUAL  :
    --------------------------------------------
    */

    public Tsr<ValueType> setIsVirtual( boolean isVirtual ) {
        if ( isVirtual() != isVirtual ) {
            Device device = this.find( Device.class );
            try {
                if ( device != null ) device.restore( this );
            } catch ( Exception exception ) {
                _LOG.error(
                        "Tensor could not be restored from device component when changing flag 'isVirtual' to " + isVirtual + "."
                        , exception
                );
                throw exception;
            }
            if ( isVirtual ) {
                if ( getData() == null ) _allocate( 1 );
                else _virtualize();
                Relation<ValueType> relation = find( Relation.class );
                if ( relation!=null ) relation.foreachChild( c -> c._setData( getData()) );
            } else {
                Tsr<?> parentTensor = (this.isSlice())? find(Relation.class).getParent() : null;
                if ( parentTensor != null ) {
                    parentTensor.find( Relation.class ).remove( this );
                }
                _actualize();
            }
            _setIsVirtual( isVirtual );
            if ( _NDConf != null ) _configureFromNewShape( _NDConf.shape(), isVirtual, true );
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
        } else if ( isVirtual && getData() == null ) _allocate( 1 );//_value = //new double[]{0};
        return this;
    }

    public boolean isVirtual() {
        return (_flags & IS_VIRTUAL_MASK) == IS_VIRTUAL_MASK;
    }

    /**
     *  This method is the inner counterpart to the public "setIsVirtual" method.
     *  It actually performs the bit flipping by applying the corresponding bit mask.
     * @param isVirtual The truth value which ought to be applied.
     */
    protected void _setIsVirtual( boolean isVirtual ) {
        if ( isVirtual() != isVirtual ) {
            if ( isVirtual ) _flags += IS_VIRTUAL_MASK;
            else _flags -= IS_VIRTUAL_MASK;
        }
    }

    /*
    --------------------------------------------
        §(2.3) : GRADIENT APPLY REQUIREMENT  :
    --------------------------------------------
    */

    public Tsr<ValueType> setGradientApplyRqd( boolean applyRequested ) {
        if ( gradientApplyRqd() != applyRequested ) {
            if ( applyRequested ) {
                if (
                        Neureka.instance().settings().autograd().isApplyingGradientWhenRequested() &&
                                !Neureka.instance().settings().autograd().isApplyingGradientWhenTensorIsUsed()
                ) {
                    this.applyGradient();
                } else _flags += GRADIENT_APPLY_RQD_MASK;
            }
            else _flags -= GRADIENT_APPLY_RQD_MASK;
        }
        return this;
    }

    public boolean gradientApplyRqd() {
        return (_flags & GRADIENT_APPLY_RQD_MASK) == GRADIENT_APPLY_RQD_MASK;
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
     * 'AbstractComponentOwner' from which this class inherits.
     * In this super class the component logic is implemented.
     *
     * @param newComponent A component used to access features. (GraphNode, IndexAlias, Relation, int[], ...)
     * @return The unchanged object or maybe in future versions: null (component rejected)
     */
    @Override
    protected < T extends Component<Tsr<ValueType>> > T _setOrReject(T newComponent )
    {
        if ( newComponent.getClass() == HostCPU.class ) return null;
        if ( newComponent instanceof Device && !( (Device) newComponent ).has( this ) )
        {
            if ( this.has( Relation.class ) ) {
                Relation relation = find( Relation.class );
                if ( relation.hasParent() ) { // Root needs to be found ! :
                    Tsr<ValueType> root = relation.findRootTensor();
                    try {
                        ((Device)newComponent).store( root );
                    } catch ( Exception exception ) {
                        _LOG.error( "Could not store tensor on device '" + newComponent.toString() +"'.", exception );
                        throw exception;
                    }
                    root.find( Relation.class ).foreachChild( c -> ((Tsr)c).setIsOutsourced( true ) );
                } else { // This is root ! :
                    relation.foreachChild( c -> ((Tsr<?>)c).setIsOutsourced( true ) );
                    try {
                        ((Device)newComponent).store( this );
                    } catch ( Exception exception ) {
                        _LOG.error( "Could not store tensor on device '" + newComponent.toString() +"'.", exception );
                        throw exception;
                    }
                }
            } else {
                try {
                    ((Device)newComponent).store( this );
                } catch ( Exception exception ) {
                    _LOG.error( "Could not store tensor on device '" + newComponent.toString() +"'.", exception );
                    throw exception;
                }
            }
            if ( ((Device)newComponent).has( this ) ) setIsOutsourced( true );
        } else if ( newComponent instanceof Tsr ) {
            if (
                    ((Tsr)newComponent).shape().hashCode() != this.shape().hashCode() ||
                            Arrays.hashCode(((Tsr)newComponent).getNDConf().shape()) != Arrays.hashCode( _NDConf.shape() )
            ) newComponent = null;
        }
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
     * 'AbstractComponentOwner' from which this class inherits.
     * In this super class the component logic is implemented.
     *
     * @param newComponent A component used to access features. (GraphNode, IndexAlias, Relation, int[], ...)
     * @return The unchanged object or when rejected: null (component rejected)
     */
    @Override
    protected <T extends Component<Tsr<ValueType>>> T _removeOrReject( T newComponent )
    {
        if ( newComponent instanceof Device ) {
            Device<ValueType> device = (Device<ValueType>) newComponent;
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
     *  Important : Components of type Tsr are simply gradients!
     *  This method does not need to have an implementation in this case.
     *  (A gradient tensor "does not mind" an owner change...)
     *
     * @param oldOwner The previous owner type instance.
     * @param newOwner The new owner type instance.
     */
    @Override
    public void update( Tsr<ValueType> oldOwner, Tsr<ValueType> newOwner ) {
        // This is means that this tensor is a gradient that is being
        // transferred to another tensor to serve as gradient...
        // No update task needs to occur. (This might change in the future...)
    }


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

    public boolean isEmpty() {
        return getData() == null && !this.isOutsourced();
    }

    public boolean isUndefined() {
        return _NDConf ==null || _NDConf.shape() == null;
    }

    public boolean isSlice() {
        Relation<ValueType> child = find( Relation.class );
        return ( child != null && child.hasParent() );
    }

    public int sliceCount() {
        Relation<ValueType> child = find( Relation.class );
        return ( child != null ) ? child.childCount() : 0;
    }

    public boolean isSliceParent() {
        Relation<ValueType> parent = find( Relation.class );
        return ( parent != null && parent.hasChildren() );
    }

    public boolean belongsToGraph() {
        return this.has( GraphNode.class );
    }

    public boolean isLeave() {
        return (!this.has( GraphNode.class )) || this.find( GraphNode.class ).isLeave();
    }

    public boolean isBranch() {
        return !this.isLeave();
    }

    public boolean hasGradient() {
        return this.has( Tsr.class );
    }

    /*
        ----------------------------------------------
            §(4.1) : COMPONENT BASED PROPERTIES :
        ----------------------------------------------
     */

    public Tsr<ValueType> getGradient() {
        return this.find( Tsr.class );
    }

    /**
     * @return The device on which this tensor is stored or 'CPU' if it is not outsourced.
     */
    public Device<ValueType> device() {
        if ( this.isOutsourced() ) return this.find( Device.class );
        return (Device<ValueType>) _CPU;
    }

    /**
     *
     * @return The graph node of the computation graph to which this tensor belongs or null if not part of a graph.
     */
    public GraphNode<ValueType> getGraphNode() {
        return find( GraphNode.class );
    }

    /**
     *
     * @return Custom IndexAlias object.
     */
    public IndexAlias<ValueType> index() {
        return find( IndexAlias.class );
    }


    /*
        ---------------------------------------
            §(4.2) : INNER PROPERTIES :
        ---------------------------------------
     */

    private int _dataLength()
    {
        if ( !(getData() instanceof float[]) && !(getData() instanceof double[]) ) {
            if ( getData() instanceof Object[] ) return ((Object[]) getData()).length;
            else return -1;
        } 
        else if ( getData() instanceof double[] ) return ( (double[]) getData()).length;
        else return ( (float[]) getData()).length;
    }

    /*==================================================================================================================
    |
    |       §(5) : OBJECT STATE MODIFICATION :
    |   ------------------------------------------
    */
    /**
     *  This method is responsible for incrementing
     *  the "_version" field variable which represents the version of the data of this tensor.
     *  Meaning :
     *  Every time the underlying data (_value) changes this version ought to increment alongside.
     *  The method is called during the execution procedure.
     *
     * @param call The context object containing all relevent informatin that defines a call for tensor execution.
     * @return This very tensor instance. (factory pattern)
     */
    public Tsr<ValueType> incrementVersionBecauseOf( ExecutionCall call ) {
        if ( Neureka.instance().settings().autograd().isPreventingInlineOperations() ) {
            _version ++;
            GraphNode<?> node = find( GraphNode.class );
            if ( node != null && node.getPayloadReferenceVersion() != this._version ) {
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


    public Tsr<ValueType> delete() 
    {
        forComponent( GraphNode.class, n -> {
            if ( n.isUsedAsDerivative() ) {
                String message = "Cannot delete a tensor which is used as derivative by the AD computation graph!";
                _LOG.error( message );
                throw new IllegalStateException( message );
            }
        });
        forComponent( Device.class, d -> d.free( this ) );
        _flags = -1;
        _setData( null );
        _NDConf = null;
        forComponent( Tsr.class, Tsr::delete );
        _components = null;
        return this;
    }

    protected Tsr<ValueType> _become( Tsr<ValueType> tensor )
    {
        if ( tensor == null ) return this;
        this.setDataType( tensor.getDataType() );
        _setData( tensor.getData() );
        _NDConf = tensor._NDConf;
        _components = Collections.synchronizedList( new ArrayList<>() );
        _flags = tensor._flags;
        if ( tensor._components != null ) { // Inform components about their new owner:
            _components.addAll( tensor._components );
            List<Component<Tsr<ValueType>>> snapshot = new ArrayList<>( tensor._components );
            for ( Component<Tsr<ValueType>> o : snapshot ) o.update( tensor, this );
        }
        tensor._setData( null );
        tensor.setDataType( null );
        tensor._NDConf = null;
        tensor._components = null;
        tensor._flags = -1;
        return this;
    }



    /*==================================================================================================================
    |
    |       §(6) : ND-ITERATOR LOGIC :
    |   ---------------------------------------
    */

    @NotNull
    @Override
    public Iterator<ValueType> iterator()
    {
        NDIterator _ndi = NDIterator.of( this );
        return new Iterator<ValueType>() 
        {
            private int _count = 0;
            private final int _size = size();

            @Override
            public boolean hasNext() {
                return _count != _size;
            }

            @Override
            public ValueType next() {
                Object o = getDataAt( _ndi.i() );
                _ndi.increment();
                _count ++;
                return (ValueType) o;
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
     *
     * @param error A tensor which is back-propagated to gradients. Must match the size og this tensor.
     * @return The tensor on which this method was called. (factory pattern)
     */
    public Tsr<ValueType> backward( Tsr<ValueType> error ) {
        if ( !forComponent( GraphNode.class, node -> node.backward( error ) ) && this.rqsGradient() ) {
            addToGradient( error );
        }
        return this;
    }

    /**
     *  This method turns the given scalar value and
     *  turns it into a matching tensor (same shape)
     *  which will be back-propagated through the
     *  recorded computation graph.
     *
     * @param value A scalar which is back-propagated to gradients. Must match the size og this tensor.
     * @return The tensor on which this method was called. (factory pattern)
     */
    public Tsr<ValueType> backward( double value )
    {
        backward( new Tsr( _NDConf.shape(), value ) );
        return this;
    }

    /**
     *  This method assumes that the user wants to backpropagate
     *  an error of "1" having the same shape as
     *  this tensor.
     *
     * @return The tensor on which this method was called. (factory pattern)
     */
    public Tsr<ValueType> backward()
    {
        backward( 1 );
        return this;
    }

    public void applyGradient()
    {
        forComponent( JITProp.class, JITProp::execute );
        remove( JITProp.class );
        forComponent(
                Tsr.class,
                g -> {
                    forComponent( Optimizer.class, o -> o.optimize( this ) );
                    remove( Tsr.class );
                    boolean inlineSafety = Neureka.instance().settings().autograd().isPreventingInlineOperations();
                    if ( inlineSafety )
                        Neureka.instance().settings().autograd().setIsPreventingInlineOperations( false );
                    // INLINE OPERATION :
                    Function.Detached.PLUS_ASSIGN.call( new Tsr[]{ this, g } );
                    // INLINE END !
                    if ( inlineSafety )
                        Neureka.instance().settings().autograd().setIsPreventingInlineOperations( true );
                }
        );
    }

    public void detach()
    {
        this.remove( GraphNode.class );
    }

    /*
        ----------------------------
            §(7.1) : FRAMING :
        ----------------------------
        ... for more context see package 'framing'...
     */

    /**
     *  This method receives a nested String array which
     *  ought to contain a label for the index of this tensor.
     *  The index for a single element of this tensor would be an array
     *  of numbers as long as the rank where every number is
     *  in the range of the corresponding shape dimension...
     *  Labeling an index means that for every dimension there
     *  must be a label for elements in this range array! <br>
     *  For example the shape (2,3) could be labeled as follows: <br>
     *
     *      dim 0 : ["A", "B"]
     *      dim 1 : ["1", "2", "3"]
     *
     * @param labels A nested String array containing labels for indexes of the tensor dimensions.
     * @return This tensor (method chaining).
     */
    public Tsr<ValueType> label( String[][] labels )
    {
        IndexAlias indexAlias = find( IndexAlias.class );
        if ( indexAlias == null ) {
            indexAlias = new IndexAlias( this.rank() );
            set( indexAlias );
        }
        assert labels.length <= this.rank();
        for( int i = 0; i < labels.length; i++ ) {
            if ( labels[ i ] != null ) {
                for ( int ii = 0; ii < labels[ i ].length; ii++ ) {
                    if ( labels[ i ][ ii ] != null ) indexAlias.set( i, labels[ i ][ ii ], ii );
                }
            }
        }
        return this;
    }

    /**
     *  This method receives a nested String list which
     *  ought to contain a label for the index of this tensor.
     *  The index for a single element of this tensor would be an array
     *  of numbers as long as the rank where every number is
     *  in the range of the corresponding shape dimension...
     *  Labeling an index means that for every dimension there
     *  must be a label for elements in this range array! <br>
     *  For example the shape (2,3) could be labeled as follows: <br>
     *
     *      dim 0 : ["A", "B"]
     *      dim 1 : ["1", "2", "3"]
     *
     * @param labels A nested String list containing labels for indexes of the tensor dimensions.
     * @return This tensor (method chaining).
     */
    public Tsr<ValueType> label( List<List<Object>> labels )
    {
        IndexAlias indexAlias = find( IndexAlias.class );
        if ( indexAlias == null ) set( new IndexAlias( labels ) );
        return this;
    }

    /**
     *  This method provides the ability to
     *  label not only the indices of the shape of this tensor, but also
     *  the dimension of the shape.
     *  The first and only argument of the method expects a map instance
     *  where keys are the objects which ought to act as dimension labels
     *  and the values are lists of labels for the indices of said dimensions.
     *  For example the shape (2,3) could be labeled as follows: <br>
     *  [
     *      "dim 0" : ["A", "B"],
     *      "dim 1" : ["1", "2", "3"]
     *  ]
     * @param labels A map where keys are dimension labels and values are lists of index labels.
     * @return This tensor (method chaining).
     */
    public Tsr<ValueType> label( Map<Object, List<Object>> labels )
    {
        this.set( new IndexAlias<>( labels, this ) );
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

    public Tsr<ValueType> plus( Tsr<ValueType> other ) {
        return Function.PLUS.call( new Tsr[]{ this, other } );
    }

    public Tsr<ValueType> plusAssign( Tsr<ValueType> other ) {
        return Function.Detached.PLUS_ASSIGN.call( new Tsr[]{ this, other } );
    }

    public Tsr<ValueType> plus( Double value ) {
        return plus( new Tsr<>( this.shape(), value ) );
    }

    public Tsr<ValueType> minus( Tsr<ValueType> other ) {
        return Function.MINUS.call( new Tsr[]{ this, other } );
    }

    public Tsr<ValueType> minusAssign( Tsr<ValueType> other ) {
        return Function.Detached.MINUS_ASSIGN.call( new Tsr[]{ this, other } );
    }

    public Tsr<ValueType> negative() {
        return Function.NEG.call( new Tsr[]{ this } );
    }

    public Tsr<ValueType> multiply( Tsr<ValueType> other ) {
        return Function.MUL.call( new Tsr[]{ this, other } );
    }

    public Tsr<ValueType> timesAssign( Tsr<ValueType> other ) {
        return Function.Detached.MUL_ASSIGN.call( new Tsr[]{ this, other } );
    }

    public Tsr<ValueType> multiply( Double value ) {
        return multiply( new Tsr<>( this.shape(), value ) );
    }

    public Tsr<ValueType> div( Tsr<ValueType> other ) {
        return Function.DIV.call( new Tsr[]{ this, other } );
    }

    public Tsr<ValueType> div( Double value ) {
        return div( new Tsr<>( this.shape(), value ) );
    }

    public Tsr<ValueType> divAssign( Tsr<ValueType> other ) {
        return Function.Detached.DIV_ASSIGN.call( new Tsr[]{ this, other } );
    }

    public Tsr<ValueType> mod( Tsr<ValueType> other ) {
        return Function.MOD.call( new Tsr[]{ this, other } );
    }

    public Tsr<ValueType> modAssign( Tsr<ValueType> other ) {
        return Function.Detached.MOD_ASSIGN.call( new Tsr[]{ this, other } );
    }

    public Tsr<ValueType> power( Tsr<ValueType> other ) {
        return Function.POW.call( new Tsr[]{ this, other } );
    }

    public Tsr<ValueType> power( Double value ) {
        return power( new Tsr<>( this.shape(), value ) );
    }

    public Tsr<ValueType> xor( Tsr<ValueType> other ) {
        return Function.POW.call( new Tsr[]{ this, other} );
    }

    public Tsr<ValueType> xor( Double value ) {
        return xor( new Tsr<>( this.shape(), value ) );
    }

    /*
        -----------------------------
            §(8.1) : OPERATIONS :
        -----------------------------
     */

    /**
     * @return A new transposed tensor with the same underlying data as this tensor.
     */
    public Tsr<ValueType> T()  // Transposed!
    {
        StringBuilder operation = new StringBuilder();
        for ( int i = rank() - 1; i >= 0; i-- ) operation.append( i ).append( ( i == 0 ) ? "" : ", " );
        operation = new StringBuilder( "[" + operation + "]:(I[ 0 ])" );
        return new Tsr<>( this, operation.toString() );
    }

    /**
     *  This method performs various operations by calling Function instances
     *  in order to ultimately calculate the mean value of all values
     *  of this very tensor!
     *  This scalar tensor is then returned.
     *
     * @return A scalar tensor which is the mean value of all values of this very tensor.
     */
    public Tsr<ValueType> mean() {
        Tsr<ValueType> ones = new Tsr<>( this.getNDConf().shape(), 1 );
        Tsr<ValueType> sum = Function.X.call( new Tsr[]{ this, ones } );
        return Function.DIV.call( new Tsr[]{ sum, new Tsr( this.size() ) } );
        //TODO :Function.DIV.call(new Tsr[]{sum, new Tsr(this.size())});
    }

    /**
     *  This method performs a dot product between the last dimension of this tensor
     *  and the first dimension of the passed tensor.
     *
     * @param b The tensor which is the right part of the dot product operation.
     * @return A new tensor which is the dot product of this tensor and the passed one.
     */
    public Tsr<ValueType> dot( Tsr<ValueType> b ) {
        Tsr<ValueType> a = this;
        int[][] fitter = AbstractNDArray.Utility.Indexing.makeFit( a.getNDConf().shape(), b.getNDConf().shape() );
        boolean doReshape = false;
        for ( int i = 0; i < fitter[ 0 ].length && !doReshape; i++ ) if ( fitter[ 0 ][ i ] != i ) doReshape = true;
        for ( int i = 0; i < fitter[ 1 ].length && !doReshape; i++ ) if ( fitter[ 1 ][ i ] != i ) doReshape = true;
        if ( doReshape ) {
            a = Function.create( AbstractNDArray.Utility.Stringify.strConf( fitter[ 0 ] ) + ":(I[ 0 ])" ).call( a );
            b = Function.create( AbstractNDArray.Utility.Stringify.strConf( fitter[ 1 ] ) + ":(I[ 0 ])" ).call( b );
        }
        return Function.X.call( new Tsr[]{ a, b } ).dimtrim();
    }

    /**
     *  This method creates a new tensor sharing the same data and whose shape is trimmed.
     *  A trimmed shape is simply a shape without preceding and trailing ones.
     *  For example the shape (1x4x1x2x1) would be trimmed to (4x1x2).
     *  The underlying operation does not perform a removal of redundant ones all together.
     *  Only ones at the start and the beginning will be removed.
     *  A scalar tensor will not be affected by this operation.
     *
     * @return A tensor with the same underlying data but possibly trimmed shape without preceding or trailing ones.
     */
    public Tsr<ValueType> dimtrim() {
        return Function.DIMTRIM.call( this );
    }

    /**
     *  This method name translates to the "in" keyword in Groovy!
     *  The same is true for the "contains" method in Kotlin.
     *  Both methods do the exact same thing, however they exist
     *  for better language support.
     *
     * @param t The tensor which will be checked.
     * @return The answer to the following question: Is the data of the provided tensor a subset of the data of this tensor?
     */
    public boolean isCase( Tsr<ValueType> t ) {
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
    public boolean contains( Tsr<ValueType> t ) {
        return isCase( t );
    }


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
     *  The following method enables access to specific elements within the tensor.
     *  The method name also translates to the subscript operator in Groovy.
     *
     * @param idx The index of the element which should be returned.
     * @return An element located at the provided index.
     */
    public Object getAt( int[] idx ) {
        return getDataAt( getNDConf().i_of_idx( idx ) );
    }

    /**
     *  The following method enables access to scalar values.
     *  The method name also translates to the subscript operator in Groovy.
     *
     * @param idx The index of the element which should be returned.
     * @return A scalar value located at the provided index.
     */
    public double getF64( int[] idx ) {
        return value64( i_of_idx( idx ) );
    }


    /**
     *  The following method enables the creation of tensor slices which access
     *  the same underlying data (possibly from a different view).
     *  The method name also translates to the subscript operator in Groovy.
     *
     * @param i1
     * @param i2
     * @return
     */
    public Object getAt( Object i1, Object i2 ) {
        List<Object> args = Arrays.asList( i1, i2 );
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
    public Tsr<ValueType> getAt( int i ) {
        return getAt( new Object[]{ i, i } );
    }

    /**
     *  The following method returns a raw value item within this tensor
     *  targeted by a scalar index.
     *
     * @param i The scalar index of the value item which should be returned by the method.
     * @return The value item found at the targeted index.
     */
    public ValueType getValueAt( int i ) {
        return (ValueType) getDataAt( _NDConf.i_of_i( i ) );
    }

    /**
     *  This method returns a raw value item within this tensor
     *  targeted by an index array which is expect to hold an index for
     *  every dimension of the shape of this tensor.
     *  So the provided array must have the same length as the
     *  rank of this tensor!
     *
     * @param idx The index array which targets a single value item within this tensor.
     * @return The found raw value item targeted by the provided index array.
     */
    public ValueType getValueAt( int[] idx ) {
        return (ValueType) getDataAt( _NDConf.i_of_idx( idx ) );
    }

    /**
     *  Individual entries for value items in this tensor can be set
     *  via this method.
     *
     * @param i The scalar index targeting a specific value position within this tensor
     *          which ought to be replaced by the one provided by the second parameter
     *          of this method.
     *
     * @param o The item which ought to be placed at the targeted position.
     * @return This very tensor in order to enable method chaining...
     */
    public Tsr<ValueType> setAt( int i, ValueType o ) {
        setDataAt( _NDConf.i_of_i( i ), o );
        return this;
    }

    public Tsr<ValueType> getAt( double i ) {
        return (Tsr<ValueType>) getAt( Arrays.asList( _NDConf.idx_of_i( (int) Math.floor( i ) ) ).toArray() );
    }

    public Tsr<ValueType> getAt( BigDecimal i ) {
        return (Tsr<ValueType>) getAt( Arrays.asList( _NDConf.idx_of_i(( i ).intValue()) ).toArray() );
    }

    public Object getAt( Map<?,?> rangToStrides )
    {
        if ( rangToStrides == null ) return this;
        int[] newOffset = new int[ this.rank() ]; // ...not a simple slice... Advanced:
        int[] newSpread = new int[ this.rank() ];
        int[] newShape = new int[ this.rank() ];
        Object[] ranges = rangToStrides.keySet().toArray();
        _configureSubsetFromRanges( ranges, newOffset, newSpread, newShape, 0 );
        Object[] steps = rangToStrides.values().toArray();
        for ( int i = 0; i < this.rank(); i++ ) {
            newSpread[ i ] = (Integer) steps[ i ];
            newShape[ i ] /= (Integer) steps[ i ];
        }
        return _sliceOf( newShape, newOffset, newSpread );
    }

    public Tsr<ValueType> shallowCopy()
    {
        if ( this.isEmpty() || this.isUndefined() ) return this;
        List<List<Integer>> ranges = new ArrayList<>();
        for ( int e : this.shape() ) {
            List<Integer> rangeAsList = new ArrayList<>();
            for ( int i = 0; i < e; i++ ) rangeAsList.add( i );
            ranges.add( rangeAsList);
        }
        return (Tsr<ValueType>) getAt( ranges.toArray() );
    }

    /**
     *  This method enables tensor slicing!
     *  It takes a key of various types and configures a slice
     *  tensor which shares the same underlying data as the original tensor.
     *
     * @param key This object might be a wide range of objects including maps, lists or arrays...
     * @return A slice tensor or scalar value.
     */
    public Tsr<ValueType> getAt( Object key ) {
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

        int[] newOffset = new int[ this.rank() ];
        int[] newSpread = new int[ this.rank() ];
        int[] newShape = new int[ this.rank() ];
        key = ( key instanceof List ) ? ((List<?>) key).toArray() : key;

        if ( key instanceof Object[] ) {
            boolean allInt = true;
            for ( Object o : (Object[]) key ) allInt = allInt && o instanceof Integer;
            if ( allInt && ((Object[]) key).length == rank() ) {
                newOffset = _intArray((Object[]) key);
                if ( newOffset != null ) {
                    for ( int i = 0; i < this.rank(); i++ )
                        newOffset[i] = ( newOffset[i] < 0 ) ? _NDConf.shape( i ) + newOffset[ i ] : newOffset[ i ];
                    for ( int i = 0; i < this.rank(); i++ )
                        ((Object[])key)[i] = newOffset[i];
                    allInt = false;
                }
            }
            boolean hasScale = false;
            for ( Object o : (Object[]) key ) hasScale = hasScale || o instanceof Map;
            if ( allInt ) _configureSubsetFromRanges(
                    new Object[]{ _intArray( (Object[]) key ) },
                    newOffset, newSpread,
                    newShape,
                    0
            );
            else _configureSubsetFromRanges( (Object[]) key, newOffset, newSpread, newShape, 0 );
        } else {
            String message = "Cannot create tensor slice from key of type '" + key.getClass().getName() + "'!";
            _LOG.error( message );
            throw new IllegalArgumentException( message );
        }
        return _sliceOf( newShape, newOffset, newSpread );
    }

    private Tsr<ValueType> _sliceOf( int[] newShape, int[] newOffset, int[] newSpread )
    {
        this.setIsVirtual( false );
        Tsr<ValueType> subset = new Tsr<>();
        subset.setDataType( this.getDataType() );
        subset._setData( this.getData() );
        int[] newTranslation = this._NDConf.translation();
        int[] newIdxmap = NDConfiguration.Utility.newTlnOf( newShape );

        for ( int i = 0; i < this.rank(); i++ )
            newSpread[ i ] = ( newSpread[i] == 0 ) ? 1 : newSpread[ i ];

        for ( int i = 0; i < newOffset.length; i++ )
            newOffset[ i ] = newOffset[ i ] + getNDConf().offset( i ); // Offset is being inherited!

        Tsr<?> rootTensor = ( this.isSlice() ) ? find( Relation.class ).findRootTensor() : this;
        Tsr<?> parentTensor = ( this.isSlice() ) ? find( Relation.class ).getParent() : this;
        /*
            The following code check the validity of the slice shape ranges with
            respect to the 'parentTensor' of this new slice.
         */
        if ( parentTensor.rank() != newShape.length || rootTensor != parentTensor ) {
            // TODO! This requires some more thought about how to check this!
            // THIS CASE HAS NOT YET BEEN THOUGHT TROUGH!
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
                by the 'Reshape' operation! ( Hopefully! :) )

                The following would occur when : "new Tsr(...).T().gatAt(...);"
                Transposing a tensor performs an inline reshaping of an identical
                slice of the original tensor! Then again slicing this tensor
                via the 'getAt(...)' method leads us to a situation where
                the following variable is NOT NULL! :
             */
            int[] reshaped = ( this.isSlice() ) ? parentTensor.find( Relation.class ).getReshapeRelationFor( this ) : null;
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

        subset._NDConf = AbstractNDC.construct( newShape, newTranslation, newIdxmap, newSpread, newOffset );

        if ( this.isOutsourced() ) {
            Device device = this.find( Device.class );
            device.store( subset, this );
            subset.setIsOutsourced( true );
        }
        if ( this.isVirtual() ) subset.setIsVirtual( true );
        subset.set( new Relation().addParent( this ) );
        Relation parent = find( Relation.class );
        parent = ( parent != null ) ? parent : new Relation();
        parent.addChild( subset );
        this.set( parent );
        return subset;
    }


    /**
     *
     * @param ranges Elements of this array might be multiple things:
     *               - A map whose first entry represents a mapping between range and steps.
     *               - A list from which a first and last entry will be interpreted as range.
     *               - Any other object which might bew found in a 'IndexAlias' component.
     * @param offset Start index for every rank.
     * @param newShape New shape of the new sub-tensor.
     * @param iOffset Rank offset incremented according to recursive calls.
     * @return A new rank index.
     */
    private int _configureSubsetFromRanges(
            Object[] ranges,
            int[] offset,  int[] spread,
            int[] newShape,
            int iOffset
    ) {
        for ( int i = 0; i < ranges.length; i++ ) {
            int first = 0;
            int last = 0;
            if ( ranges[ i ] instanceof int[] ) {
                List<Integer> intList = new ArrayList<>( ( (int[]) ranges[ i ] ).length );
                for ( int ii : (int[]) ranges[ i ] ) intList.add( ii );
                ranges[ i ] = intList;
            } else if ( ranges[ i ] instanceof String[] ) {
                List<String> strList = new ArrayList<>( ( (String[]) ranges[ i ] ).length);
                for ( String ii : (String[]) ranges[ i ] ) strList.add( ii );
                ranges[ i ] = strList;
            }
            if ( !( ranges[ i ] instanceof  List ) ) {
                if ( ranges[ i ] instanceof Map ) {
                    Object[] ks = ( (Map<?,?>) ranges[ i ] ).keySet().toArray();
                    Object[] steps = ( (Map<?,?>) ranges[ i ]).values().toArray();
                    int newI = _configureSubsetFromRanges( ks, offset, spread, newShape, i + iOffset );
                    for ( int ii = 0; ii < steps.length; ii++ ) {
                        spread[ ii + i + iOffset ] = (Integer) steps[ ii ];
                        newShape[ ii + i + iOffset ] /= spread[ ii + i + iOffset ];
                    }
                    i = newI;
                    continue;
                } else if ( ranges[ i ] instanceof Integer ) {
                    first = (Integer) ranges[ i ];
                    last = (Integer) ranges[ i ];
                } else {
                    IndexAlias<?> indexAlias = find( IndexAlias.class );
                    if ( indexAlias != null ) {
                        int position = indexAlias.get( ranges[ i ], i + iOffset );
                        first = position;
                        last = position;
                    } else {
                        String message = "Given indexAlias key at axis " + ( i + iOffset ) + " not found!";
                        _LOG.error( message );
                        throw new IllegalStateException( message );
                    }
                }
            } else {
                ranges[ i ] = ( (List<?>) ranges[ i ] ).toArray();
                ranges[ i ] = ( ( (Object[]) ranges[ i ] )[ 0 ] instanceof List )
                        ? ( (List<?>) ( (Object[]) ranges[ i ] )[ 0 ] ).toArray()
                        : ( (Object[]) ranges[ i ] );
                if (
                        !( ( (Object[]) ( ranges[ i ] ) )[ 0 ] instanceof Integer )
                                || !( ( (Object[]) ranges[ i ] )[ ( (Object[]) ( ranges[ i ] ) ).length - 1 ] instanceof Integer )
                ) {
                    IndexAlias<?> indexAlias = find( IndexAlias.class );
                    if ( !( ( (Object[]) (ranges[ i ]) )[ 0 ] instanceof Integer ) ) {
                        if ( indexAlias != null ) {
                            first = indexAlias.get( ( (Object[]) ranges[ i ])[ 0 ], i + iOffset );
                        }
                    }
                    else first = (Integer) ( (Object[]) ranges[ i ] )[ 0 ];

                    if ( !( ( (Object[]) ranges[ i ] )[ ( (Object[]) ranges[ i ] ).length - 1 ] instanceof Integer )  ) {
                        if ( indexAlias != null ) {
                            last = indexAlias.get(
                                    ( (Object[]) ranges[ i ] )[ ( (Object[]) ranges[ i ] ).length - 1 ],
                                    i + iOffset
                            );
                        }
                    }
                    else last = (Integer) ( (Object[]) ranges[ i ] )[ ( (Object[]) ranges[ i ] ).length - 1 ];

                } else {
                    first = (Integer)( (Object[]) ranges[ i ] )[ 0 ];
                    last = (Integer) ( (Object[]) ranges[ i ] )[ ( (Object[]) ranges[ i ] ).length - 1 ];
                }
            }
            if ( first < 0 && last < 0 && first > last ) {
                int temp = first;
                first = last;
                last = temp;
            }
            first = ( first < 0 ) ? _NDConf.shape( i ) + first : first;
            last = ( last < 0 ) ? _NDConf.shape( i ) + last : last;
            newShape[ i + iOffset ] = ( last - first ) + 1;
            offset[ i + iOffset ] = first;
        }
        return ranges.length + iOffset - 1;
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
    public Tsr<ValueType> putAt( List<?> key, Tsr<ValueType> value ) {
        _putAtCheckFor( value );
        Tsr<ValueType> slice = ( key == null ) ? this : (Tsr) getAt( key );
        return _putAt( slice, value );
    }

    /**
     *  This method enables injecting slices of tensor to be assigned into this tensor!
     *  It takes a key which is used to configure a slice
     *  tensor sharing the same underlying data as the original tensor.
     *  This slice is then used to assign the second argument to it, namely
     *  the "value" argument.
     *
     * @param key This object is a map defining a stride and a targeted index or range of indices...
     * @return A slice tensor or scalar value.
     */
    public Tsr<ValueType> putAt( Map<?,?> key, Tsr<ValueType> value ) {
        _putAtCheckFor( value );
        Tsr<ValueType> slice = ( key == null ) ? this : (Tsr<ValueType>) getAt( key );
        return _putAt( slice, value );
    }

    private void _putAtCheckFor( Tsr value ) {
        if ( value.isEmpty() ) {
            String message = "Provided tensor is empty! Empty tensors cannot be injected.";
            _LOG.error( message );
            throw new IllegalArgumentException( message );
        }
    }

    private Tsr<ValueType> _putAt( Tsr<ValueType> slice, Tsr<ValueType> value )
    {
        boolean valueIsDeviceVisitor = false;
        if ( slice.isOutsourced() && !value.isOutsourced() ) {
            Device<ValueType> device = slice.find( Device.class );
            try {
                device.store( value );
            } catch ( Exception exce ) {
                _LOG.error( "Trying to migrate target slice tensor to device failed.", exce );
                throw exce;
            }
            valueIsDeviceVisitor = true;
        }
        if ( this.isEmpty() && slice.isEmpty() || slice.size() != value.size() ) _become( value ); // TODO: Rethink this a little
        else new Tsr( new Tsr[]{ slice, value }, "I[ 0 ] <- I[ 1 ]", false );
        try {
            if ( valueIsDeviceVisitor ) value.find( Device.class ).restore( value );
        } catch ( Exception exception ) {
            _LOG.error( "Trying to migrate source tensor back to original location failed.", exception );
            throw exception;
        }
        return this;
    }

    /**
     *  This is a static nested utility class
     *  which is used to allow for fast access to
     *  tensors storing doubles.
     *
     */
    public static class IO
    {
        private IO() {}

        public static double getFrom( Tsr<?> t, int i ) {
            if ( t.isEmpty() || t.isUndefined() ) return 0;
            else if ( t.isVirtual() ) return t.value64()[ 0 ];
            return t.value64()[ t.i_of_i( i ) ];
        }

        public static double getFrom( Tsr<?> t, int[] idx ) {
            t.setIsVirtual( false );
            return t.value64()[ t.i_of_idx( idx ) ];
        }

        public static void setInto( Tsr<?> t, int i, double value ) {
            t.setIsVirtual( false );
            t.value64()[ t.i_of_i( i ) ] = value;
        }

        public static void setInto( Tsr<?> t, int[] idx, double value ) {
            t.setIsVirtual( false );
            t.value64()[ t.i_of_idx( idx ) ] = value;
        }

        public static void addInto( Tsr<?> t, int i, double value ) {
            t.setIsVirtual( false );
            t.value64()[ t.i_of_i( i ) ] += value;
        }

        public static void addInto( Tsr<?> t, int[] idx, double value ) {
            t.setIsVirtual( false );
            t.value64()[ t.i_of_idx( idx ) ] += value;
        }

        public static Tsr<?> addInto( Tsr<?> t, Tsr<?> source ) {
            if ( t.isVirtual() && source.isVirtual() ) t.value64()[ 0 ] += source.value64()[ 0 ];
            else FunctionBuilder.build( "I[ 0 ]<-(I[ 0 ]+I[ 1 ])", false ).call( new Tsr[]{ t, source } );
            return source;
        }

        public static void subInto( Tsr<?> t, int i, double value ) {
            t.setIsVirtual( false );
            t.value64()[ t.i_of_i( i ) ] -= value;
        }

        public static void subInto( Tsr<?> t, int[] idx, double value ) {
            t.setIsVirtual( false );
            t.value64()[ t.i_of_idx( idx ) ] -= value;
        }

        public static void subInto( Tsr<?> t, Tsr<?> source ) {
            if ( t.isVirtual() && source.isVirtual() ) {
                t.value64()[ 0 ] -= source.value64()[ 0 ];
            } else {
                if ( t.isVirtual() ) t.setIsVirtual( false );
                int[] index = new int[ t.getNDConf().shape().length ];
                int size = t.size();
                for ( int i = 0; i < size; i++ ) {
                    IO.subInto( t, index, IO.getFrom( source, index ) );
                    NDConfiguration.Utility.increment( index, t.getNDConf().shape() );
                }
            }
        }

        public static void mulInto( Tsr<?> t, int i, double value ) {
            t.setIsVirtual( false );
            t.value64()[ t.i_of_i( i ) ] *= value;
        }

        public static void mulInto( Tsr<?> t, int[] idx, double value ) {
            t.setIsVirtual( false );
            t.value64()[ t.i_of_idx( idx ) ] *= value;
        }

    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    /**
     *  A tensor ought to have some way to access its underlying data array.
     *  This method simple returns an element within this data array sitting at position "i".
     * @param i The position of the targeted item within the raw data array of the tensor.
     * @return The found object sitting at the specified index position.
     */
    @Override
    public Object getDataAt( int i )
    {
        if ( this.isOutsourced() ) {
            Device<ValueType> device = this.find( Device.class );
            if ( device instanceof OpenCLDevice ) {
                return ( (OpenCLDevice) device ).value64f( (Tsr<Number>) this, i );
            }
        }
        else if ( getData() instanceof float[] ) return ( (float[]) getData())[ i ];
        else if ( getData() instanceof double[] ) return ( (double[]) getData())[ i ];
        else if ( getData() instanceof short[] ) return ( (short[]) getData())[ i ];
        else if ( getData() instanceof int[] ) return ( (int[]) getData())[ i ];
        else if ( getData() instanceof byte[] ) return ( (byte[]) getData())[ i ];
        else if ( getData() instanceof long[] ) return ( (long[]) getData())[ i ];
        else return ( (ValueType[]) getData())[ i ];
        return null;
    }

    /**
     *  A tensor ought to have some way to selectively modify its underlying data array.
     *  This method simply returns an element within this data array sitting at position "i".
     * @param i The index of the data array entry which ought to be addressed.
     * @param o The object which ought to be placed at the requested position.
     * @return This very tensor in order to enable method chaining.
     */
    @Override
    public Tsr<ValueType> setDataAt( int i, ValueType o ) {
        if ( getData() instanceof Object[] ) ( (Object[]) getData() )[ i ] = o;
        else if ( getData() instanceof float[]  ) ( (float[])  getData() )[ i ] = (float)  o;
        else if ( getData() instanceof double[] ) ( (double[]) getData() )[ i ] = (double) o;
        else if ( getData() instanceof int[]    ) ( (int[])    getData() )[ i ] = (int)    o;
        else if ( getData() instanceof long[]   ) ( (long[])   getData() )[ i ] = (long)   o;
        else if ( getData() instanceof short[]  ) ( (short[])  getData() )[ i ] = (short)  o;
        else if ( getData() instanceof byte[]   ) ( (byte[])   getData() )[ i ] = (byte)   o;
        return this;
    }

    public Tsr<ValueType> setValue64( double[] value ) {
        if ( this.isOutsourced() ) this.find( Device.class ).overwrite64( this, value );
        else if ( getData() == null ) {
            setDataType( DataType.of( F64.class ) );
            _setData( value );
        }
        else if ( getData() instanceof float[] )
            for ( int i = 0; i < value.length; i++ ) ( (float[]) getData())[ i ] = (float) value[ i ];
        else if ( getData() instanceof double[] )
            for ( int i = 0; i < value.length; i++ ) ( (double[]) getData())[ i ] = value[ i ];
        return this;
    }

    public Tsr<ValueType> setValue32( float[] value ) {
        if ( this.isOutsourced() ) this.find( Device.class ).overwrite32( this, value );
        else if ( getData() == null ) {
            setDataType( DataType.of( F32.class ) );
            _setData( value );
        }
        else if ( getData() instanceof float[] )
            for ( int i = 0; i < value.length; i++ ) ( (float[]) getData())[ i ] = value[ i ];
        else if ( getData() instanceof double[] )
            for ( int i = 0; i < value.length; i++ ) ( (double[]) getData())[ i ] = value[ i ];
        return this;
    }

    public Tsr<ValueType> setValue( Object value ) {
        if ( value instanceof float[] ) this.setValue32( (float[]) value );
        else if ( value instanceof  double[] ) this.setValue64( (double[]) value );
        else if ( value instanceof Float ) {
            this.setIsVirtual( true );
            if ( this.is32() ) ( (float[]) getData())[ 0 ] = (Float) value;
            else ( (double[]) getData())[ 0 ] = ( (Float) value ).doubleValue();
        } else if ( value instanceof Double ) {
            this.setIsVirtual( true );
            if ( this.is64() ) ( (double[]) getData())[ 0 ] = (Double) value;
            else ( (float[]) getData())[ 0 ] = ( (Double) value ).floatValue();
        } else if ( value instanceof int[] ) {
            setDataType( DataType.of(I32.class) );
            _setData( value );
            setIsVirtual( false );
        } else if ( value instanceof short[] ) {
            setDataType( DataType.of(I16.class) );
            _setData( value );
            setIsVirtual( false );
        }
        return this;
    }

    public Object getValue() { // TODO : Make this what it is supposed to be!!!
        if ( this.isOutsourced() ) {
            Device device = find( Device.class );
            if ( device != null ) {
                return device.valueFor( this );
                //return ( this.is32() )
                //        ? device.value32f( this )
                //        : device.value64f( this );
            }
            else return getData();
        }
        else if ( !this.isVirtual() ) return getData();
        else return getDataType().actualize( getData(), this.size() );
    }
    
    public double[] gradient64() {
        Tsr<ValueType> gradient = this.find( Tsr.class );
        if ( gradient == null ) return new double[ 0 ];
        return ( this.is32() )
                ? DataConverter.Utility.floatToDouble( gradient.value32() )
                : gradient.value64();
    }

    public float[] gradient32() {
        Tsr<ValueType> gradient = this.find( Tsr.class );
        if ( gradient == null ) return new float[ 0 ];
        return ( this.is64() )
                ? DataConverter.Utility.doubleToFloat( gradient.value64() )
                : gradient.value32();
    }


    public Tsr<ValueType> addToGradient( Tsr<ValueType> error ) {
        if (
                !forComponent(
                    Tsr.class,
                        gradient ->
                        this.set(
                                Function.Detached.PLUS_ASSIGN.call( new Tsr[]{ gradient, error } )
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
     *  elements have been converted to a new data type, namely :
     *  the type specified by the argument
     *
     *  The method does not change this tensor, which is why the operation is pure.
     *  Important to note is that the method will return instances of the specified
     *  type but merely another tensor containing elements of that type...
     *  The name of this method for example translates to the "as" operator
     *  found in Groovy, so the following code : " myTensor as Double "
     *  would not return a Double instance!
     *
     * @param typeClass The class which is the target of the underlying type conversion...
     * @param <T> The value type of the tensor that will be returned.
     * @return A new tensor which hosting the supplied type.
     */
    public <T> Tsr<T> asType( Class<T> typeClass )
    {
        DataType newDT = DataType.of( typeClass );
        Object newData;
        if ( this.isOutsourced() ) {
            Device device = find( Device.class );
            device.restore( this );
            newData = _convertedDataOfType( typeClass );
            device.store( this );
        }
        else newData = _convertedDataOfType( typeClass );
        return (Tsr<T>) new Tsr( this.getNDConf().shape(), newDT, newData );
    }

    /**
     *  This method is an inline operation which changes the underlying data of this tensor.
     *  It converts the data types of the elements of this tensor to the specified type!
     *  WARNING : The use of this method is discouraged for the following reasons:
     *  1. Inline operations are inherently error prone for most use cases.
     *  2. This inline operation in particular has no safety net,
     *     meaning that there is no implementation of version mismatch detection
     *     like there is for those operations present in the operation backend...
     *     No exceptions will be thrown during backpropagation!
     *  3. This method has not yet been implemented to also handle instances which
     *     are slices of parent tensors!
     *     Therefore there might be unexpected performance penalties or side effects
     *     associated with this method.
     *
     * @param typeClass The target type class for elements of this tensor.
     * @param <T> The type parameter for the returned tensor.
     * @return The same tensor instance whose data has been converted to hold a different type.
     */
    public <T> Tsr<T> toType( Class<T> typeClass )
    {
        if ( this.isOutsourced() ) {
            setDataType( DataType.of( typeClass ) );
            return (Tsr<T>) this;
        }
        else {
            Object newData = _convertedDataOfType( typeClass );
            _setData( null );
            setDataType( DataType.of( typeClass ) );
            _setData( newData );
        }
        forComponent( Tsr.class, gradient -> gradient.toType( typeClass ) );
        return (Tsr<T>) this;
    }

    public double value64( int i ) {
        if ( this.isOutsourced() ) {
            if ( find( Device.class ) instanceof  OpenCLDevice )
                return find( OpenCLDevice.class ).value64f( (Tsr<Number>) this, i );
            else return 0.0;
        }
        if ( this.isVirtual() ) {
            if ( this.is64() ) return ( (double[]) getData() )[ 0 ];
            else return ( (float[]) getData() )[ 0 ];
        } else {
            if ( this.is64() ) return ( (double[]) getData() )[ i ];
            else return ( (float[]) getData() )[ i ];
        }
    }

    public double[] value64() {
        Device found = this.find( Device.class );
        if ( getData() == null && this.isOutsourced() && found != null ) {
            if ( found instanceof OpenCLDevice )
                return ( (OpenCLDevice) found).value64f((Tsr<Number>) this);
            else return null;
        }
        double[] newValue = DataConverter.instance().convert(getData(), double[].class );

        if ( this.isVirtual() && newValue != null && this.size() > 1 ) {

           double[] value = new double[ this.size() ];
           Arrays.fill( value, newValue[ 0 ] );
           return value;
        }
        return newValue;
    }

    public float value32( int i ) {
        if ( this.isOutsourced() ) {
            if ( find( Device.class ) instanceof OpenCLDevice )
                return find( OpenCLDevice.class ).value32f( (Tsr<Number>) this, i );
            else return 0.0f;
        }
        if ( this.isVirtual() ) {
            if ( this.is64() ) return (float) ( (double[]) getData() )[ 0 ];
            else return ( (float[]) getData())[ 0 ];
        } else {
            if ( this.is64() ) return (float) ( (double[]) getData() )[ i ];
            else return ( (float[]) getData() )[ i ];
        }
    }

    public float[] value32() {
        Device<ValueType> found = this.find( Device.class );
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

    //DISPLAY :
    //=========================
    public String toString( String mode ) {
        return _toString( mode, ( mode.contains( "f" ) ) ? "    " : null );
    }

    public String toString( Map<TsrAsString.Should, Object> config, String deep ) {
        return new TsrAsString( this, config ).toString( deep );
    }

    protected String _toString( String mode, String deep )
    {
        return new TsrAsString( this, mode ).toString( deep );
    }

    @Override
    public String toString()
    {
        return new TsrAsString( this ).toString();
    }


    public static void makeFit( Tsr[] tsrs, boolean doesAD )
    {
        int largest = -1;
        int[] shape = null;
        for ( Tsr t : tsrs ) if ( t.rank() > largest ) {
            largest = t.rank();
            shape = t.getNDConf().shape();
        }
        int prefix = 0;
        for ( int s : shape ) if ( s == 1 ) prefix++; else break;
        int postfix = 0;
        for ( int i = shape.length-1; i>=0; i-- ) if ( shape[ i ] == 1 ) postfix++; else break;
        for ( int i = 0; i < tsrs.length; i++ ) {
            if ( tsrs[ i ].rank() != largest ) {
                int[] oldShape = tsrs[ i ].getNDConf().shape();
                int[] newReshape = new int[ largest ];
                int padding = largest - oldShape.length;

                int handle = ( postfix <= prefix ) ? padding : largest - padding;
                for ( int ii = 0; ii < handle; ii++ ) newReshape[ ii ] = ( postfix <= prefix ) ? -1 : ii;
                for ( int ii = handle; ii < largest; ii++ ) newReshape[ ii ] = ( postfix <= prefix ) ? ii - padding : -1;

                Function f = Function.create(
                    AbstractNDArray.Utility.Stringify.strConf( newReshape ) + ":(I[ 0 ])",
                        doesAD
                );
                tsrs[ i ] = f.call( tsrs[ i ] );
            }
        }

    }

    /**
     *  This is a nested static utility class which is used
     *  to create tensor instances.
     */
    @NoArgsConstructor
    public static class Create
    {
        public  static Tsr<?> E( int[] shape ) {
            return new Tsr( shape, 2.7182818284590452353602874713527 );
        }

        public static Tsr<?> newRandom( int[] shape ) {
            return newRandom( shape, 8701252152903546L );
        }

        public static Tsr<?> newRandom( int[] shape, long seed ) {
            int size = NDConfiguration.Utility.szeOfShp( shape );
            return new Tsr<>( shape, DataConverter.Utility.newSeededDoubleArray( seed, size ) );
        }

        public static Tsr<?> newTsrLike( Tsr<?> template, double value ) {
            Tsr<Object> t = (Tsr<Object>) _newEmptyLike( template );
            if ( template.is32() ) t.setValue( (float) value );
            else t.setValue( value );
            try {
                if ( template.isOutsourced() ) ( (Device<Object>) template.find( Device.class ) ).store( t );
            } catch ( Exception exception ) {
                _LOG.error( "Failed storing a newly created tensor from a template tensor to its host device.", exception );
                throw exception;
            }
            return t;
        }

        public static Tsr<?> newTsrLike( Tsr<?> template ) { // The output tensor will not have gradients!
            Tsr t = _newEmptyLike( template );
            if ( template.is32() ) t.setValue32( new float[ template.size() ] );
            else t.setValue64( new double[ template.size() ] );
            try {
                if ( template.isOutsourced() ) ( (Device<Object>) template.find( Device.class ) ).store( t );
            } catch ( Exception exception ) {
                _LOG.error( "Failed storing a newly created tensor from a template tensor to its host device.", exception );
                throw exception;
            }
            return t;
        }

        private static Tsr<?> _newEmptyLike( Tsr<?> template ) {
            Tsr<?> t = new Tsr<>();
            t._configureFromNewShape( template.getNDConf().shape(), false, true );
            return t;
        }

    }

}
