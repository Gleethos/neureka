package neureka.ndim;

import neureka.common.utility.DataConverter;
import neureka.common.utility.LogUtil;
import neureka.dtype.DataType;
import neureka.dtype.custom.*;
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
     * @param ndcProducer A producer of the {@link NDConfiguration} interface implementation.
     * @param makeVirtual A flag determining if the tensor should be actual or virtual (not fully allocated).
     * @param autoAllocate Determines if the underlying data array should be allocated or not.
     */
    public void configureFromNewShape( NDCProducer ndcProducer, boolean makeVirtual, boolean autoAllocate )
    { 
        _API.setIsVirtual( makeVirtual ); 
        if ( _API.getData() == null && autoAllocate ) _API.allocate( makeVirtual ? 1 : ndcProducer.getSize() );
        _API.setConf( ndcProducer.produceNDC( makeVirtual ) );
    }

    public void tryConstructing( NDCProducer ndcProducer, DataType<?> dataType, Object data )
    {
        LogUtil.nullArgCheck( ndcProducer.getShape(), "shape", int[].class );
        LogUtil.nullArgCheck( dataType, "dataType", DataType.class );
        LogUtil.nullArgCheck( data, "data", Object.class );

        int size = ndcProducer.getSize();
        if ( data instanceof List<?> ) {
            List<?> range = (List<?>) data;
            data = range.toArray();// TODO: This is probably wrong!
        }
        if ( data instanceof Object[] )
            data = _autoConvertAndOptimizeObjectArray( (Object[]) data, dataType, size );

        boolean isDefinitelyScalarValue = ( dataType == DataType.of( data.getClass() ) );

        if ( data instanceof Number && !isDefinitelyScalarValue ) {
            data = DataConverter.get().convert( data, dataType.getValueTypeClass() );
            isDefinitelyScalarValue = true;
        }

        if ( isDefinitelyScalarValue ) // This means that "data" is a single value!
            if ( constructAllFromOne( ndcProducer, data ) ) return;

        if (      data instanceof double[]  ) _constructForDoubles(  ndcProducer, (double[]) data );
        else if ( data instanceof float[]   ) _constructForFloats(   ndcProducer, (float[]) data );
        else if ( data instanceof int[]     ) _constructForInts(     ndcProducer, (int[]) data );
        else if ( data instanceof byte[]    ) _constructForBytes(    ndcProducer, (byte[]) data );
        else if ( data instanceof short[]   ) _constructForShorts(   ndcProducer, (short[]) data );
        else if ( data instanceof boolean[] ) _constructForBooleans( ndcProducer, (boolean[]) data );
        else if ( data instanceof long[]    ) _constructForLongs(    ndcProducer, (long[]) data );
        else {
            _API.setType(dataType);
            configureFromNewShape(ndcProducer, false, false);
            _API.setData(data);
        }
    }

    private Object _autoConvertAndOptimizeObjectArray( Object[] data, DataType<?> dataType, int size ) {
        if ( Arrays.stream( data ).anyMatch( e -> DataType.of(e.getClass()) != dataType ) ) {
            for ( int i = 0; i < ( data ).length; i++ ) {
                ( data )[i] = DataConverter.get().convert( ( (Object[]) data )[i], dataType.getValueTypeClass() );
            }
        }
        return _optimizeObjectArray(dataType, data, size);
    }

    public boolean constructAllFromOne( NDCProducer ndcProducer, Object data ) {
        if ( data instanceof Double    ) { _constructAllF64(  ndcProducer, (Double)    data ); return true; }
        if ( data instanceof Float     ) { _constructAllF32(  ndcProducer, (Float)     data ); return true; }
        if ( data instanceof Integer   ) { _constructAllI32(  ndcProducer, (Integer)   data ); return true; }
        if ( data instanceof Short     ) { _constructAllI16(  ndcProducer, (Short)     data ); return true; }
        if ( data instanceof Byte      ) { _constructAllI8(   ndcProducer, (Byte)      data ); return true; }
        if ( data instanceof Long      ) { _constructAllI64(  ndcProducer, (Long)      data ); return true; }
        if ( data instanceof Boolean   ) { _constructAllBool( ndcProducer, (Boolean)   data ); return true; }
        if ( data instanceof Character ) { _constructAllChar( ndcProducer, (Character) data ); return true; }
        if ( Number.class.isAssignableFrom( data.getClass() ) ) {
            _constructAllF64( ndcProducer, ((Number)data).doubleValue() ); return true;
        } else if ( !data.getClass().isArray() ) {
            _constructAll( ndcProducer, data ); return true;
        }
        return false;
    }

    private void _constructAllF64( NDCProducer ndcProducer, double value ) {
        _constructAll( ndcProducer, F64.class );
        ( (double[]) _API.getData())[ 0 ] = value;
    }

    private void _constructAllF32( NDCProducer ndcProducer, float value ) {
        _constructAll( ndcProducer, F32.class );
        ( (float[]) _API.getData())[ 0 ] = value;
    }

    private void _constructAllI32( NDCProducer ndcProducer, int value ) {
        _constructAll( ndcProducer, I32.class );
        ( (int[]) _API.getData())[ 0 ] = value;
    }

    private void _constructAllI16( NDCProducer ndcProducer, short value ) {
        _constructAll( ndcProducer, I16.class );
        ( (short[]) _API.getData())[ 0 ] = value;
    }

    private void _constructAllI8( NDCProducer ndcProducer, byte value ) {
        _constructAll( ndcProducer, I8.class );
        ( (byte[]) _API.getData())[ 0 ] = value;
    }

    private void _constructAllI64( NDCProducer ndcProducer, long value ) {
        _constructAll( ndcProducer, I64.class );
        ( (long[]) _API.getData())[ 0 ] = value;
    }

    private void _constructAllBool( NDCProducer ndcProducer, boolean value ) {
        _constructAll( ndcProducer, Boolean.class );
        ( (boolean[]) _API.getData())[ 0 ] = value;
    }

    private void _constructAllChar( NDCProducer ndcProducer, char value ) {
        _constructAll( ndcProducer, Character.class );
        ( (char[]) _API.getData())[ 0 ] = value;
    }

    private void _constructAll( NDCProducer ndcProducer, Object value ) {
        _constructAll( ndcProducer, value.getClass() );
        ( (Object[]) _API.getData())[ 0 ] = value;
    }

    private void _constructAll( NDCProducer ndcProducer, Class<?> typeClass ) {
        int size = ndcProducer.getSize();
        _API.setType( DataType.of( typeClass ) );
        configureFromNewShape( ndcProducer, size > 1, true );
    }

    private void _constructForDoubles( NDCProducer ndcProducer, double[] value )
    {
        int size = ndcProducer.getSize();
        _API.setType( DataType.of( F64.class ) );
        if ( size != value.length ) {
            _API.allocate( size );
            for ( int i = 0; i < size; i++ ) ( (double[]) _API.getData())[ i ]  = value[ i % value.length ];
        }
        else _API.setData( value );
        configureFromNewShape( ndcProducer, false, false );
    }

    private void _constructForFloats( NDCProducer ndcProducer, float[] value )
    {
        int size = ndcProducer.getSize();
        _API.setType( DataType.of( F32.class ) );
        if ( size != value.length ) {
            _API.allocate( size );
            for ( int i = 0; i < size; i++ ) ( (float[]) _API.getData())[ i ]  = value[ i % value.length ];
        } else _API.setData( value );
        configureFromNewShape( ndcProducer, false, false );
    }

    private void _constructForInts( NDCProducer ndcProducer, int[] value )
    {
        int size = ndcProducer.getSize();
        _API.setType( DataType.of( I32.class ) );
        if ( size != value.length ) {
            _API.allocate( size );
            for ( int i = 0; i < size; i++ ) ( (int[]) _API.getData())[ i ]  = value[ i % value.length ];
        } else _API.setData( value );
        configureFromNewShape( ndcProducer, false, false );
    }

    private void _constructForShorts( NDCProducer ndcProducer, short[] value )
    {
        int size = ndcProducer.getSize();
        _API.setType( DataType.of( I16.class ) );
        if ( size != value.length ) {
            _API.allocate( size );
            for ( int i = 0; i < size; i++ ) ( (short[]) _API.getData())[ i ]  = value[ i % value.length ];
        } else _API.setData( value );
        configureFromNewShape( ndcProducer, false, false );
    }

    private void _constructForBooleans( NDCProducer ndcProducer, boolean[] value )
    {
        int size = ndcProducer.getSize();
        _API.setType( DataType.of( Boolean.class ) );
        if ( size != value.length ) {
            _API.allocate( size );
            for ( int i = 0; i < size; i++ ) ( (boolean[]) _API.getData())[ i ]  = value[ i % value.length ];
        } else _API.setData( value );
        configureFromNewShape( ndcProducer, false, false );
    }

    private void _constructForBytes( NDCProducer ndcProducer, byte[] value )
    {
        int size = ndcProducer.getSize();
        _API.setType( DataType.of( I8.class ) );
        if ( size != value.length ) {
            _API.allocate( size );
            for ( int i = 0; i < size; i++ ) ( (byte[]) _API.getData())[ i ]  = value[ i % value.length ];
        } else _API.setData( value );
        configureFromNewShape( ndcProducer, false, false );
    }

    private void _constructForLongs( NDCProducer ndcProducer, long[] value )
    {
        int size = ndcProducer.getSize();
        _API.setType( DataType.of( I64.class ) );
        if ( size != value.length ) {
            _API.allocate( size );
            for ( int i = 0; i < size; i++ ) ( (long[]) _API.getData())[ i ]  = value[ i % value.length ];
        } else _API.setData( value );
        configureFromNewShape( ndcProducer, false, false );
    }

    public <V> void constructSeeded( Class<V> valueType, NDCProducer ndcProducer, Object seed ) {
        _API.setType( DataType.of(valueType) );
        int size = ndcProducer.getSize();

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

        configureFromNewShape( ndcProducer, false, false  );
    }

    /**
     *  If possible, turns the provided {@code Object} array into a memory compact array of primitive types.
     *
     * @param dataType The {@link DataType} of the elements in the provided array.
     * @param values The array of values which ought to be optimized into a flat array of primitives.
     * @param size The size of the optimized array of primitives.
     * @return An optimized flat array of primitives.
     */
    private static Object _optimizeObjectArray(DataType<?> dataType, Object[] values, int size ) {
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
    
    public interface NDCProducer 
    {
        int getSize();
        int[] getShape();

        NDConfiguration produceNDC(boolean makeVirtual);


        static NDCProducer of( int[] newShape ) {

            int size = NDConfiguration.Utility.sizeOfShape( newShape );
            if ( size == 0 ) {
                String shape = Arrays.stream( newShape ).mapToObj( String::valueOf ).collect( Collectors.joining( "x" ) );
                String message = "The provided shape '"+shape+"' must not contain zeros. Dimensions lower than 1 are not possible.";
                _LOG.error( message );
                throw new IllegalArgumentException( message );
            }
            return new NDCProducer() {
                @Override public int getSize() { return size;}
                @Override public int[] getShape() { return newShape.clone(); }
                @Override public NDConfiguration produceNDC(boolean makeVirtual) {
                    if ( makeVirtual ) return VirtualNDConfiguration.construct( newShape );
                    else {
                        int[] newTranslation = NDConfiguration.Layout.ROW_MAJOR.newTranslationFor( newShape );
                        int[] newSpread = new int[ newShape.length ];
                        Arrays.fill( newSpread, 1 );
                        int[] newOffset = new int[ newShape.length ];
                        return
                                NDConfiguration.of(
                                        newShape,
                                        newTranslation,
                                        newTranslation, // indicesMap
                                        newSpread,
                                        newOffset
                                );
                    }
                }
            };
        }

    }


}
