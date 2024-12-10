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
    private final Args _Args;
    private final Device<Object> _targetDevice;
    private final NDConstructor _ndConstructor;

    /**
     *
     * @param targetDevice The {@link Device} to be used for the construction of the {@link Tensor}
     * @param ndConstructor A producer of the {@link NDConfiguration} interface implementation.
     * @param Args An implementation of the {@link Args} interface.
     */
    public TensorConstructor(Device<?> targetDevice, NDConstructor ndConstructor, Args Args) {
        LogUtil.nullArgCheck( targetDevice, "targetDevice", Device.class, "Cannot construct a tensor without target device." );
        LogUtil.nullArgCheck( ndConstructor, "ndConstructor", NDConstructor.class, "Cannot construct tensor without shape information." );
        _targetDevice = (Device<Object>) targetDevice;
        _ndConstructor = ndConstructor;
        _Args = Args;
    }

    /**
     *  Constructs the tensor without any initial (filled) {@link Data}.
     *
     * @param makeVirtual A flag determining if the tensor should be actual or virtual (not fully allocated).
     * @param autoAllocate Determines if the underlying data array should be allocated or not.
     */
    Args unpopulated(
            boolean makeVirtual, boolean autoAllocate, DataType<?> type
    ) {
        NDConfiguration ndc = _ndConstructor.produceNDC( makeVirtual );
        _Args.setIsVirtual( makeVirtual );
        _Args.setConf( ndc );
        if ( autoAllocate )
            _Args.setData( _targetDevice.allocate( type, ndc ) );
        return _Args;
    }

    public Args constructTrusted(Data<?> data ) {
        _Args.setConf( _ndConstructor.produceNDC( false ) );
        _Args.setData( data );
        return _Args;
    }

    public Args tryConstructing(
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

            if ( isDefinitelyScalarValue ) { // This means that "data" is a single value!
                newPopulatedFromOne( data, dataType.getItemTypeClass() );
                if ( data != null )
                    return _Args;
            }
        }

        NDConfiguration ndc = _ndConstructor.produceNDC( false );
        _Args.setIsVirtual( false );
        _Args.setConf( ndc );
        _Args.setData( _targetDevice.allocateFromAll( dataType, ndc, data) );
        return _Args;
    }

    public Args newPopulatedFromOne(Object singleItem, Class<?> type )
    {
        int size = _ndConstructor.getSize();
        NDConfiguration ndc = _ndConstructor.produceNDC(_ndConstructor.getSize() > 1);
        DataType<Object> dataType = (DataType<Object>) DataType.of( type );
        Data<?> array = _targetDevice.allocateFromOne( dataType, ndc, singleItem );
        _Args.setIsVirtual( size > 1 );
        _Args.setConf( ndc );
        _Args.setData( array );
        return _Args;
    }

    public <V> Args newSeeded(Class<V> valueType, Arg.Seed seed )
    {
        NDConfiguration ndc = _ndConstructor.produceNDC( false );
        Data<?> data = _targetDevice.allocate( DataType.of( valueType ), ndc );
        Object out = CPURandomization.fillRandomly( data.getOrNull(), seed );
        assert out == data.getOrNull();
        _Args.setIsVirtual( false );
        _Args.setConf( ndc );
        _Args.setData( data );
        return _Args;
    }

    /**
     *  An interface defining methods for configuring a {@link Tensor}
     *  in the making...
     */
    static class Args {
        private NDConfiguration _conf;
        private Data<?>         _data;
        private Boolean         _isVirtual;

        public void setConf( NDConfiguration conf ) { _conf = conf; }

        public void setData( Data<?> o ) { _data = o; }

        public void setIsVirtual( boolean isVirtual ) { _isVirtual = isVirtual; }

        public NDConfiguration getConf() { return _conf; }

        public Data<?> getData() { return _data; }

        public Boolean isVirtual() { return _isVirtual; }
    }

}
