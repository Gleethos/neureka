package neureka;

import neureka.backend.main.implementations.elementwise.CPURandomization;
import neureka.common.utility.DataConverter;
import neureka.common.utility.LogUtil;
import neureka.devices.Device;
import neureka.dtype.DataType;
import neureka.ndim.NDConstructor;
import neureka.ndim.config.NDConfiguration;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 *  The {@link TsrConstructor} is an <b>internal API</b> for receiving a wide range
 *  of different inputs and using them to populate the fields
 *  of freshly instantiated {@link neureka.Tsr} instances.
 *  The existence of this class is a symptom of the fact that there
 *  is a very large API for creating tensors in Neureka.
 *  This means that all the code dealing with verifying and converting
 *  API input (provided by various {@link neureka.Tsr#of} as well as {@link neureka.Tsr#of} methods)
 *  sits inside this class instead of polluting the already very large
 *  {@link neureka.TsrImpl} class.
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
        void   setData( Data<?> o );
        void   setIsVirtual(  boolean isVirtual );
    }

    private final API _API;
    private final Device<Object> _targetDevice;
    private final NDConstructor _ndConstructor;

    /**
     *
     * @param targetDevice The {@link Device} to be used for the construction of the {@link neureka.Tsr}
     * @param ndConstructor A producer of the {@link NDConfiguration} interface implementation.
     * @param API An implementation of the {@link API} interface.
     */
    public TsrConstructor( Device<?> targetDevice, NDConstructor ndConstructor, API API ) {
        LogUtil.nullArgCheck( targetDevice, "targetDevice", Device.class, "Cannot construct a tensor without target device." );
        LogUtil.nullArgCheck( ndConstructor, "ndConstructor", NDConstructor.class, "Cannot construct tensor without shape information." );
        _targetDevice = (Device<Object>) targetDevice;
        _ndConstructor = ndConstructor;
        _API = API;
    }

    /**
     *  Constructs the tensor without any initial {@link Data}.
     *
     * @param makeVirtual A flag determining if the tensor should be actual or virtual (not fully allocated).
     * @param autoAllocate Determines if the underlying data array should be allocated or not.
     */
    void unpopulated(
            boolean makeVirtual, boolean autoAllocate, DataType<?> type
    ) {
        _API.setType( type );
        _API.setIsVirtual( makeVirtual );
        NDConfiguration ndc = _ndConstructor.produceNDC( makeVirtual );
        if ( autoAllocate ) _API.setData( _targetDevice.allocate( type, ndc ) );
        _API.setConf( ndc );
    }

    public void constructTrusted(
            DataType<?> dataType,
            Data<?> data
    ) {
        _API.setType( dataType );
        _API.setData( data );
        _API.setConf( _ndConstructor.produceNDC( false ) );
    }

    public void tryConstructing(
        DataType<?> dataType,
        Object data
    ) {
        LogUtil.nullArgCheck( _ndConstructor, "ndConstructor", NDConstructor.class );
        LogUtil.nullArgCheck( _ndConstructor.getShape(), "shape", int[].class );
        LogUtil.nullArgCheck( dataType, "dataType", DataType.class );
        LogUtil.nullArgCheck( data, "data", Object.class );

        int size = _ndConstructor.getSize();
        if ( data instanceof Object[] )
            data = _autoConvertAndOptimizeObjectArray( (Object[]) data, dataType, size );

        boolean isDefinitelyScalarValue = ( dataType == DataType.of( data.getClass() ) );

        if ( data instanceof Number && !isDefinitelyScalarValue ) {
            data = DataConverter.get().convert( data, dataType.getItemTypeClass() );
            isDefinitelyScalarValue = true;
        }

        if ( isDefinitelyScalarValue ) // This means that "data" is a single value!
            if ( newPopulatedFromOne( data, dataType.getItemTypeClass() ) ) return;

        _API.setType( dataType );
        _API.setIsVirtual( false );
        _API.setConf( _ndConstructor.produceNDC( false ) );
        _API.setData( _targetDevice.allocate( data, size ) );
    }

    private Object _autoConvertAndOptimizeObjectArray( Object[] data, DataType<?> dataType, int size ) {
        if ( Arrays.stream( data ).anyMatch( e -> DataType.of(e.getClass()) != dataType ) )
            for ( int i = 0; i < ( data ).length; i++ )
                ( data )[i] = DataConverter.get().convert( ( (Object[]) data )[i], dataType.getItemTypeClass() );

        return _optimizeObjectArray( dataType, data, size );
    }


    public boolean newPopulatedFromOne( Object singleItem, Class<?> type )
    {
        DataType<Object> dataType = (DataType<Object>) DataType.of( type );
        int size = _ndConstructor.getSize();
        _API.setType( dataType );
        _API.setIsVirtual( size > 1 );
        NDConfiguration ndc = _ndConstructor.produceNDC(_ndConstructor.getSize() > 1);
        Data<?> array = _constructAllFromOne( singleItem, ndc, type );
        _API.setData( array );
        _API.setConf( ndc );
        return singleItem != null;
    }

    private Data<?> _constructAllFromOne( Object singleItem, NDConfiguration ndc, Class<?> type )
    {
        if ( type == Double   .class ) return _constructAll(singleItem, ndc, type );
        if ( type == Float    .class ) return _constructAll(singleItem, ndc, type );
        if ( type == Integer  .class ) return _constructAll(singleItem, ndc, type );
        if ( type == Short    .class ) return _constructAll(singleItem, ndc, type );
        if ( type == Byte     .class ) return _constructAll(singleItem, ndc, type );
        if ( type == Long     .class ) return _constructAll(singleItem, ndc, type );
        if ( type == Boolean  .class ) return _constructAll(singleItem, ndc, type );
        if ( type == Character.class ) return _constructAll(singleItem, ndc, type );
        if ( Number.class.isAssignableFrom( type ) )
            return _constructAll(((Number)singleItem).doubleValue(), ndc, Double.class );
        else if ( !type.isArray() )
            return _constructAll(singleItem, ndc, type );
        else
            return null;
    }

    private Data<?> _constructAll( Object singleItem, NDConfiguration ndc, Class<?> typeClass )
    {
        DataType<Object> dataType = (DataType<Object>) DataType.of( typeClass );
        return _targetDevice.allocate( dataType, ndc, singleItem );
    }

    public <V> void newSeeded( Class<V> valueType, Object seed )
    {
        NDConfiguration ndc = _ndConstructor.produceNDC( false );
        Data<?> data = _targetDevice.allocate( DataType.of( valueType ), ndc.size() );
        Object out = CPURandomization.fillRandomly( data.getRef(), seed.toString() );
        assert out == data.getRef();
        _API.setType( DataType.of(valueType) );
        _API.setIsVirtual( false );
        _API.setConf( ndc );
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
        }
        else if ( values.length != size ) {
            Object[] objects = new Object[size];
            for( int i = 0; i < size; i++ ) objects[ i ] = values[ i % values.length ];
            data = objects;
        }
        else if ( dataType == DataType.of(Boolean.class) ) {
            boolean[] booleans = new boolean[size];
            for( int i = 0; i < size; i++ ) booleans[ i ] = (Boolean) values[ i % values.length ];
            data = booleans;
        }
        else if ( dataType == DataType.of(Character.class) ) {
            char[] chars = new char[size];
            for( int i = 0; i < size; i++ ) chars[ i ] = (Character) values[ i % values.length ];
            data = chars;
        }
        return data;
    }

}
