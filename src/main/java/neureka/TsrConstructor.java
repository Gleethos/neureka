package neureka;

import neureka.backend.main.operations.other.Randomization;
import neureka.common.utility.DataConverter;
import neureka.common.utility.LogUtil;
import neureka.devices.host.CPU;
import neureka.dtype.DataType;
import neureka.dtype.custom.*;
import neureka.ndim.NDConstructor;
import neureka.ndim.config.NDConfiguration;

import java.util.Arrays;
import java.util.List;
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
final class TsrConstructor
{
    /**
     *  An interface defining methods for configuring a {@link neureka.Tsr}
     *  in the making...
     */
    public interface API {
        void   setType( DataType<?> type );
        void   setConf( NDConfiguration conf );
        void   setData( Object o );
        Object getData();
        void   setIsVirtual(  boolean isVirtual );
    }

    private final API _API;

    public TsrConstructor( API API ) { _API = API; }

    /**
     *  This method is responsible for instantiating and setting the _conf variable.
     *  The core requirement for instantiating {@link NDConfiguration} interface implementation s
     *  is a shape array of integers which is being passed to the method... <br>
     *  <br>
     *
     * @param ndConstructor A producer of the {@link NDConfiguration} interface implementation.
     * @param makeVirtual A flag determining if the tensor should be actual or virtual (not fully allocated).
     * @param autoAllocate Determines if the underlying data array should be allocated or not.
     */
    public void configureFromNewShape(
            NDConstructor ndConstructor, boolean makeVirtual, boolean autoAllocate, DataType<?> type
    ) {
        _API.setType( type );
        _API.setIsVirtual( makeVirtual );
        if ( _API.getData() == null && autoAllocate )
            _API.setData( CPU.get().allocate( type, makeVirtual ? 1 : ndConstructor.getSize() ) );

        _API.setConf( ndConstructor.produceNDC( makeVirtual ) );
    }

    public void tryConstructing(
        NDConstructor ndConstructor,
        DataType<?> dataType,
        Object data,
        boolean trusted
    ) {
        LogUtil.nullArgCheck( ndConstructor, "ndConstructor", NDConstructor.class );
        LogUtil.nullArgCheck( ndConstructor.getShape(), "shape", int[].class );
        LogUtil.nullArgCheck( dataType, "dataType", DataType.class );
        if ( trusted ) {
            _API.setType( dataType );
            _API.setData( data );
            _API.setConf( ndConstructor.produceNDC( false ) );
            return;
        }
        LogUtil.nullArgCheck( data, "data", Object.class );

        int size = ndConstructor.getSize();
        if ( data instanceof List<?> ) {
            List<?> range = (List<?>) data;
            data = range.toArray();// TODO: This is probably wrong!
        }
        if ( data instanceof Object[] )
            data = _autoConvertAndOptimizeObjectArray( (Object[]) data, dataType, size );

        boolean isDefinitelyScalarValue = ( dataType == DataType.of( data.getClass() ) );

        if ( data instanceof Number && !isDefinitelyScalarValue ) {
            data = DataConverter.get().convert( data, dataType.getItemTypeClass() );
            isDefinitelyScalarValue = true;
        }

        if ( isDefinitelyScalarValue ) // This means that "data" is a single value!
            if ( constructAllFromOne(ndConstructor, data, dataType.getItemTypeClass() ) ) return;

        data = CPU.get().allocate( data, size );
        configureFromNewShape( ndConstructor, false, false, dataType );
        _API.setData( data );
    }

    private Object _autoConvertAndOptimizeObjectArray( Object[] data, DataType<?> dataType, int size ) {
        if ( Arrays.stream( data ).anyMatch( e -> DataType.of(e.getClass()) != dataType ) )
            for ( int i = 0; i < ( data ).length; i++ )
                ( data )[i] = DataConverter.get().convert( ( (Object[]) data )[i], dataType.getItemTypeClass() );

        return _optimizeObjectArray( dataType, data, size );
    }


    public boolean constructAllFromOne( NDConstructor ndConstructor, Object data, Class<?> type ) {

        DataType<Object> dataType = (DataType<Object>) DataType.of( type );
        int size = ndConstructor.getSize();
        _API.setType( dataType );
        _API.setIsVirtual( size > 1 );
        boolean success = _constructAllFromOne( ndConstructor.getSize(), data, type );
        _API.setConf( ndConstructor.produceNDC() );
        return success;
    }

    private boolean _constructAllFromOne( int size, Object data, Class<?> type )
    {
        if ( type == Double   .class ) { _constructAll( size, data, type ); return true; }
        if ( type == Float    .class ) { _constructAll( size, data, type ); return true; }
        if ( type == Integer  .class ) { _constructAll( size, data, type ); return true; }
        if ( type == Short    .class ) { _constructAll( size, data, type ); return true; }
        if ( type == Byte     .class ) { _constructAll( size, data, type ); return true; }
        if ( type == Long     .class ) { _constructAll( size, data, type ); return true; }
        if ( type == Boolean  .class ) { _constructAll( size, data, type ); return true; }
        if ( type == Character.class ) { _constructAll( size, data, type ); return true; }
        if ( Number.class.isAssignableFrom( type ) ) {
            _constructAll( size, ((Number)data).doubleValue(), Double.class ); return true;
        } else
            if ( !type.isArray() ) {
            _constructAll( size, data, type ); return true;
        }
        return false;
    }

    private void _constructAll( int size, Object value, Class<?> typeClass )
    {
        assert ( _API.getData() == null );
        DataType<Object> dataType = (DataType<Object>) DataType.of( typeClass );
        Object data = CPU.get().allocate( dataType, Math.min(size, 1), value );
        _API.setData( data );
    }

    public <V> void constructSeeded( Class<V> valueType, NDConstructor ndConstructor, Object seed )
    {
        int size = ndConstructor.getSize();
        Object data = CPU.get().allocate( DataType.of( valueType ), size );
        data = Randomization.fillRandomly( data, seed.toString() );
        configureFromNewShape( ndConstructor, false, false, DataType.of(valueType) );
        _API.setData( data );
    }

    /**
     *  If possible, turns the provided {@code Object} array into a memory compact array of primitive types.
     *
     * @param dataType The {@link DataType} of the elements in the provided array.
     * @param values The array of values which ought to be optimized into a flat array of primitives.
     * @param size The size of the optimized array of primitives.
     * @return An optimized flat array of primitives.
     */
    private static Object _optimizeObjectArray( DataType<?> dataType, Object[] values, int size ) {
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
        } else if ( dataType == DataType.of(Boolean.class) ) {
            boolean[] booleans = new boolean[size];
            for( int i = 0; i < size; i++ ) booleans[ i ] = (Boolean) values[ i % values.length ];
            data = booleans;
        } else if ( dataType == DataType.of(Character.class) ) {
            char[] chars = new char[size];
            for( int i = 0; i < size; i++ ) chars[ i ] = (Character) values[ i % values.length ];
            data = chars;
        }
        return data;
    }


}
