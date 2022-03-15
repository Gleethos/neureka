package neureka.ndim;

import neureka.common.utility.DataConverter;
import neureka.dtype.DataType;
import neureka.dtype.custom.*;
import neureka.ndim.config.AbstractNDC;
import neureka.ndim.config.NDConfiguration;
import neureka.ndim.config.types.views.virtual.VirtualNDConfiguration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 *  The {@link TsrConstructor} is an <b>internal API</b> for receiving a wide range
 *  of different inputs and using them to populate the fields
 *  of freshly instantiated {@link neureka.Tsr} instances.
 *  The existence of this class is a symptom of the fact that there
 *  is a very large API for creating tensors in Neureka.
 *  This means that all the code dealing with verifying and converting
 *  API input (provided by various {@link neureka.Tsr#of} methods)
 *  sits inside this class instead of polluting the already very large
 *  {@link neureka.Tsr} class.
 */
public final class TsrConstructor {

    private static final Logger _LOG = LoggerFactory.getLogger(TsrConstructor.class);

    /**
     *  An interface defining methods for configuring a {@link neureka.Tsr}
     *  in the making...
     */
    public interface API {
        void   setType( DataType<?> type );
        void   setConf( NDConfiguration conf );
        void   setData( Object o );
        void   allocate( int size );
        Object getData();
        void   setIsVirtual(  boolean isVirtual );
    }

    private final API _API;

    public TsrConstructor(API API ) { _API = API; }

    /**
     *  This method is responsible for instantiating and setting the _conf variable.
     *  The core requirement for instantiating {@link NDConfiguration} interface implementation s
     *  is a shape array of integers which is being passed to the method... <br>
     *  <br>
     *
     * @param newShape The shape which should be used to configure a new tensor (and its nd-configuration).
     * @param makeVirtual A flag determining if the tensor should be actual or virtual (not fully allocated).
     * @param autoAllocate Determines if the underlying data array should be allocated or not.
     * @param newShape An array if integers which are all greater 0 and represent the tensor dimensions.
     */
    public void configureFromNewShape( int[] newShape, boolean makeVirtual, boolean autoAllocate )
    {
        NDConfiguration.Layout layout = NDConfiguration.Layout.ROW_MAJOR;
        _API.setIsVirtual( makeVirtual );
        int size = NDConfiguration.Utility.sizeOfShape( newShape );
        if ( size == 0 ) {
            String shape = Arrays.stream( newShape ).mapToObj( String::valueOf ).collect( Collectors.joining( "x" ) );
            String message = "The provided shape '"+shape+"' must not contain zeros. Dimensions lower than 1 are not possible.";
            _LOG.error( message );
            throw new IllegalArgumentException( message );
        }
        if ( _API.getData() == null && autoAllocate ) _API.allocate( makeVirtual ? 1 : size );
        if ( makeVirtual ) _API.setConf( VirtualNDConfiguration.construct( newShape ) );
        else {
            int[] newTranslation = layout.newTranslationFor( newShape );
            int[] newSpread = new int[ newShape.length ];
            Arrays.fill( newSpread, 1 );
            int[] newOffset = new int[ newShape.length ];
            _API.setConf(
                    AbstractNDC.construct(
                            newShape,
                            newTranslation,
                            newTranslation, // indicesMap
                            newSpread,
                            newOffset
                    )
            );
        }
    }

    public void tryConstructing( int[] shape, DataType<?> dataType, Object data ) {
        int size = NDConfiguration.Utility.sizeOfShape( shape );
        if ( data instanceof List<?> ) {
            List<?> range = (List<?>) data;
            data = range.toArray();// TODO: This is probably wrong!
        }
        if ( data instanceof Object[] )
            data = _autoConvertAndOptimizeObjectArray( (Object[]) data, dataType, size );

        boolean isDefinitelyScalarValue = ( dataType == DataType.of( data.getClass() ) );

        if ( data instanceof Number && !isDefinitelyScalarValue ) {
            data = DataConverter.instance().convert( data, dataType.getJVMTypeClass() );
            isDefinitelyScalarValue = true;
        }

        if ( isDefinitelyScalarValue ) // This means that "data" is a single value!
            if ( constructAllFromOne( shape, data ) ) return;

        _API.setType( dataType );
        configureFromNewShape( shape, false, false );
        _API.setData( data );
    }

    private Object _autoConvertAndOptimizeObjectArray( Object[] data, DataType<?> dataType, int size ) {
        if ( Arrays.stream( data ).anyMatch( e -> DataType.of(e.getClass()) != dataType ) ) {
            for ( int i = 0; i < ( data ).length; i++ ) {
                ( data )[i] = DataConverter.instance().convert( ( (Object[]) data )[i], dataType.getJVMTypeClass() );
            }
        }
        return TsrConstructor.optimizeObjectArray(dataType, data, size);
    }

    public boolean constructAllFromOne( int[] shape, Object data ) {
        if ( data instanceof Double    ) { _constructAllF64( shape, (Double)    data ); return true; }
        if ( data instanceof Float     ) { _constructAllF32( shape, (Float)     data ); return true; }
        if ( data instanceof Integer   ) { _constructAllI32( shape, (Integer)   data ); return true; }
        if ( data instanceof Short     ) { _constructAllI16( shape, (Short)     data ); return true; }
        if ( data instanceof Byte      ) { _constructAllI8(  shape, (Byte)      data ); return true; }
        if ( data instanceof Long      ) { _constructAllI64( shape, (Long)      data ); return true; }
        if ( data instanceof Boolean   ) { _constructAllBool( shape, (Boolean)  data ); return true; }
        if ( data instanceof Character ) { _constructAllChar( shape, (Character)data ); return true; }
        if ( Number.class.isAssignableFrom( data.getClass() ) ) {
            _constructAllF64( shape, ((Number)data).doubleValue() ); return true;
        } else if ( !data.getClass().isArray() ) {
            _constructAll( shape, data ); return true;
        }
        return false;
    }

    private void _constructAllF64( int[] shape, double value ) {
        _constructAll( shape, F64.class );
        ( (double[]) _API.getData())[ 0 ] = value;
    }

    private void _constructAllF32( int[] shape, float value ) {
        _constructAll( shape, F32.class );
        ( (float[]) _API.getData())[ 0 ] = value;
    }

    private void _constructAllI32( int[] shape, int value ) {
        _constructAll( shape, I32.class );
        ( (int[]) _API.getData())[ 0 ] = value;
    }

    private void _constructAllI16( int[] shape, short value ) {
        _constructAll( shape, I16.class );
        ( (short[]) _API.getData())[ 0 ] = value;
    }

    private void _constructAllI8( int[] shape, byte value ) {
        _constructAll( shape, I8.class );
        ( (byte[]) _API.getData())[ 0 ] = value;
    }

    private void _constructAllI64( int[] shape, long value ) {
        _constructAll( shape, I64.class );
        ( (long[]) _API.getData())[ 0 ] = value;
    }

    private void _constructAllBool( int[] shape, boolean value ) {
        _constructAll( shape, Boolean.class );
        ( (boolean[]) _API.getData())[ 0 ] = value;
    }

    private void _constructAllChar( int[] shape, char value ) {
        _constructAll( shape, Character.class );
        ( (char[]) _API.getData())[ 0 ] = value;
    }

    private void _constructAll( int[] shape, Object value ) {
        _constructAll( shape, value.getClass() );
        ( (Object[]) _API.getData())[ 0 ] = value;
    }

    private void _constructAll( int[] shape, Class<?> typeClass ) {
        int size = NDConfiguration.Utility.sizeOfShape( shape );
        _API.setType( DataType.of( typeClass ) );
        configureFromNewShape( shape, size > 1, true );
    }

    /**
     *  This method receives a list of lists which represent a matrix of objects.
     *  It parses this matrix into a 2D shape array and a double array.<br>
     *  <br>
     *
     * @param matrix A list of lists which ought to resemble a matrix.
     */
    public void constructFor( List<List<Object>> matrix ) {
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
                constructForDoubles( shape, value );
            } else {
                String message = "Provided nested list(s) do not form a regular matrix.";
                _LOG.error( message );
                throw new IllegalArgumentException( message );
            }
        }
    }

    public void constructForDoubles( int[] shape, double[] value )
    {
        int size = NDConfiguration.Utility.sizeOfShape( shape );
        _API.setType( DataType.of( F64.class ) );
        if ( size != value.length ) {
            _API.allocate( size );
            for ( int i = 0; i < size; i++ ) ( (double[]) _API.getData())[ i ]  = value[ i % value.length ];
        }
        else _API.setData( value );
        configureFromNewShape( shape, false, false );
    }

    public void constructForFloats( int[] shape, float[] value )
    {
        int size = NDConfiguration.Utility.sizeOfShape( shape );
        _API.setType( DataType.of( F32.class ) );
        if ( size != value.length ) {
            _API.allocate( size );
            for ( int i = 0; i < size; i++ ) ( (float[]) _API.getData())[ i ]  = value[ i % value.length ];
        } else _API.setData( value );
        configureFromNewShape( shape, false, false );
    }

    public void constructForInts( int[] shape, int[] value )
    {
        int size = NDConfiguration.Utility.sizeOfShape( shape );
        _API.setType( DataType.of( I32.class ) );
        if ( size != value.length ) {
            _API.allocate( size );
            for ( int i = 0; i < size; i++ ) ( (int[]) _API.getData())[ i ]  = value[ i % value.length ];
        } else _API.setData( value );
        configureFromNewShape( shape, false, false );
    }

    public void constructForShorts( int[] shape, short[] value )
    {
        int size = NDConfiguration.Utility.sizeOfShape( shape );
        _API.setType( DataType.of( I16.class ) );
        if ( size != value.length ) {
            _API.allocate( size );
            for ( int i = 0; i < size; i++ ) ( (short[]) _API.getData())[ i ]  = value[ i % value.length ];
        } else _API.setData( value );
        configureFromNewShape( shape, false, false );
    }

    public void constructForBooleans( int[] shape, boolean[] value )
    {
        int size = NDConfiguration.Utility.sizeOfShape( shape );
        _API.setType( DataType.of( Boolean.class ) );
        if ( size != value.length ) {
            _API.allocate( size );
            for ( int i = 0; i < size; i++ ) ( (boolean[]) _API.getData())[ i ]  = value[ i % value.length ];
        } else _API.setData( value );
        configureFromNewShape( shape, false, false );
    }

    public void constructForBytes( int[] shape, byte[] value )
    {
        int size = NDConfiguration.Utility.sizeOfShape( shape );
        _API.setType( DataType.of( I8.class ) );
        if ( size != value.length ) {
            _API.allocate( size );
            for ( int i = 0; i < size; i++ ) ( (byte[]) _API.getData())[ i ]  = value[ i % value.length ];
        } else _API.setData( value );
        configureFromNewShape( shape, false, false );
    }

    public void constructForLongs( int[] shape, long[] value )
    {
        int size = NDConfiguration.Utility.sizeOfShape( shape );
        _API.setType( DataType.of( I64.class ) );
        if ( size != value.length ) {
            _API.allocate( size );
            for ( int i = 0; i < size; i++ ) ( (long[]) _API.getData())[ i ]  = value[ i % value.length ];
        } else _API.setData( value );
        configureFromNewShape( shape, false, false );
    }

    public <V> void constructSeeded( Class<V> valueType, int[] shape, Object seed ) {
        _API.setType( DataType.of(valueType) );
        int size = NDConfiguration.Utility.sizeOfShape( shape );

        if ( valueType == Double.class )
            _API.setData( DataConverter.Utility.seededDoubleArray( new double[size], seed.toString() ) );
        else if ( valueType == Float.class )
            _API.setData( DataConverter.Utility.seededFloatArray( new float[size], seed.toString() ) );
        else if ( valueType == Integer.class )
            _API.setData( DataConverter.Utility.seededIntArray( new int[size], seed.toString() ) );
        else if ( valueType == Short.class )
            _API.setData( DataConverter.Utility.seededShortArray( new short[size], seed.toString() ) );
        else if ( valueType == Byte.class )
            _API.setData( DataConverter.Utility.seededByteArray( new byte[size], seed.toString() ) );
        else if ( valueType == Long.class )
            _API.setData( DataConverter.Utility.seededLongArray( new long[size], seed.toString() ) );
        else if ( valueType == Boolean.class )
            _API.setData( DataConverter.Utility.seededBooleanArray( new boolean[size], seed.toString() ) );
        else if ( valueType == Character.class )
            _API.setData( DataConverter.Utility.seededCharacterArray( new char[size], seed.toString() ) );
        else
            throw new IllegalArgumentException("Seeding not supported for value type '"+valueType.getSimpleName()+"'!");

        configureFromNewShape( shape, false, false  );
    }

    /**
     *  If possible, turns the provided {@code Object} array into a memory compact array of primitive types.
     *
     * @param dataType The {@link DataType} of the elements in the provided array.
     * @param values The array of values which ought to be optimized into a flat array of primitives.
     * @param size The size of the optimized array of primitives.
     * @return An optimized flat array of primitives.
     */
    public static Object optimizeObjectArray( DataType<?> dataType, Object[] values, int size ) {
        Object data = values;
        IntStream indices = IntStream.iterate( 0, i -> i + 1 ).limit(size);
        if ( size > 1_000 ) indices = indices.parallel();
        indices = indices.map( i -> i % values.length );
        if      ( dataType == DataType.of(Double.class)  ) data = indices.mapToDouble( i -> (Double) values[i] ).toArray();
        else if ( dataType == DataType.of(Integer.class) ) data = indices.map( i -> (Integer) values[i] ).toArray();
        else if ( dataType == DataType.of(Long.class)    ) data = indices.mapToLong( i -> (Long) values[i] ).toArray();
        else if ( dataType == DataType.of(Float.class)   ) {
            float[] floats = new float[size];
            for( int i = 0; i < size; i++ ) floats[ i ] = (Float) values[ i % values.length ];
            data = floats;
        }
        else if ( dataType == DataType.of(Byte.class) ) {
            byte[] bytes = new byte[size];
            for( int i = 0; i < size; i++ ) bytes[ i ] = (Byte) values[ i % values.length ];
            data = bytes;
        }
        else if ( dataType == DataType.of(Short.class) ) {
            short[] shorts = new short[size];
            for( int i = 0; i < size; i++ ) shorts[ i ] = (Short) values[ i % values.length ];
            data = shorts;
        } else if ( values.length != size ) {
            Object[] objects = new Object[size];
            for( int i = 0; i < size; i++ ) objects[ i ] = values[ i % values.length ];
            data = objects;
        }
        return data;
    }


}
