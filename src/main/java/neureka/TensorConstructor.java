package neureka;

import neureka.backend.main.implementations.elementwise.CPURandomization;
import neureka.common.utility.DataConverter;
import neureka.common.utility.LogUtil;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.dtype.DataType;
import neureka.math.args.Arg;
import neureka.ndim.NDConstructor;
import neureka.ndim.config.NDConfiguration;

/**
 *  The {@link TensorConstructor} is an <b>internal API</b> for receiving a wide range
 *  of different inputs and using them to populate the fields
 *  of freshly instantiated {@link Tensor} instances.
 *  The existence of this class is a symptom of the fact that there
 *  is a very large API for creating tensors in Neureka.
 *  This means that all the code dealing with verifying and converting
 *  API input (provided by various {@link Tensor#of} as well as {@link Tensor#of} methods)
 *  sits inside this class instead of polluting the already very large
 *  {@link TensorImpl} class.
 */
final class TensorConstructor
{
    /**
     *  An interface defining methods for configuring a {@link Tensor}
     *  in the making...
     */
    public interface API {
        void   setConf( NDConfiguration conf );
        void   setData( Data<?> o );
        void   setIsVirtual(  boolean isVirtual );
    }

    private final API _API;
    private final Device<Object> _targetDevice;
    private final NDConstructor _ndConstructor;

    /**
     *
     * @param targetDevice The {@link Device} to be used for the construction of the {@link Tensor}
     * @param ndConstructor A producer of the {@link NDConfiguration} interface implementation.
     * @param API An implementation of the {@link API} interface.
     */
    public TensorConstructor(Device<?> targetDevice, NDConstructor ndConstructor, API API ) {
        LogUtil.nullArgCheck( targetDevice, "targetDevice", Device.class, "Cannot construct a tensor without target device." );
        LogUtil.nullArgCheck( ndConstructor, "ndConstructor", NDConstructor.class, "Cannot construct tensor without shape information." );
        _targetDevice = (Device<Object>) targetDevice;
        _ndConstructor = ndConstructor;
        _API = API;
    }

    /**
     *  Constructs the tensor without any initial (filled) {@link Data}.
     *
     * @param makeVirtual A flag determining if the tensor should be actual or virtual (not fully allocated).
     * @param autoAllocate Determines if the underlying data array should be allocated or not.
     */
    void unpopulated(
            boolean makeVirtual, boolean autoAllocate, DataType<?> type
    ) {
        NDConfiguration ndc = _ndConstructor.produceNDC( makeVirtual );
        _API.setIsVirtual( makeVirtual );
        _API.setConf( ndc );
        if ( autoAllocate ) _API.setData( _targetDevice.allocate( type, ndc ) );
    }

    public void constructTrusted( Data<?> data ) {
        _API.setConf( _ndConstructor.produceNDC( false ) );
        _API.setData( data );
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
            data = CPU.get().allocate( dataType.getItemTypeClass(), size, data ).getOrNull();
        else
        {
            boolean isDefinitelyScalarValue = ( dataType == DataType.of(data.getClass()) );

            if ( data instanceof Number && !isDefinitelyScalarValue ) {
                data = DataConverter.get().convert( data, dataType.getItemTypeClass() );
                isDefinitelyScalarValue = true;
            }

            if ( isDefinitelyScalarValue ) // This means that "data" is a single value!
                if ( newPopulatedFromOne( data, dataType.getItemTypeClass() ) ) return;
        }

        NDConfiguration ndc = _ndConstructor.produceNDC( false );
        _API.setIsVirtual( false );
        _API.setConf( ndc );
        _API.setData( _targetDevice.allocateFromAll( dataType, ndc, data) );
    }

    public boolean newPopulatedFromOne( Object singleItem, Class<?> type )
    {
        int size = _ndConstructor.getSize();
        NDConfiguration ndc = _ndConstructor.produceNDC(_ndConstructor.getSize() > 1);
        DataType<Object> dataType = (DataType<Object>) DataType.of( type );
        Data<?> array = _targetDevice.allocateFromOne( dataType, ndc, singleItem );
        _API.setIsVirtual( size > 1 );
        _API.setConf( ndc );
        _API.setData( array );
        return singleItem != null;
    }

    public <V> void newSeeded( Class<V> valueType, Arg.Seed seed )
    {
        NDConfiguration ndc = _ndConstructor.produceNDC( false );
        Data<?> data = _targetDevice.allocate( DataType.of( valueType ), ndc );
        Object out = CPURandomization.fillRandomly( data.getOrNull(), seed );
        assert out == data.getOrNull();
        _API.setIsVirtual( false );
        _API.setConf( ndc );
        _API.setData( data );
    }

}
