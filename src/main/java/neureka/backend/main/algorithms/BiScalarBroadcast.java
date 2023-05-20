package neureka.backend.main.algorithms;

import neureka.Shape;
import neureka.Tensor;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.fun.SuitabilityPredicate;
import neureka.backend.api.template.algorithms.AbstractFunDeviceAlgorithm;
import neureka.devices.Device;
import neureka.dtype.NumericType;

public class BiScalarBroadcast extends AbstractFunDeviceAlgorithm<BiScalarBroadcast>
{
    public BiScalarBroadcast() {
        super("scalarization");
        setAutogradModeFor( call -> AutoDiffMode.FORWARD_AND_BACKWARD );
        setIsSuitableFor( call ->
                call.validate()
                    .allNotNull(t -> t.getDataType().typeClassImplements(NumericType.class))
                    .tensors(tensors -> {
                        if (tensors.length != 2 && tensors.length != 3) return false;
                        int offset = ( tensors.length -2 );
                        if (tensors[1 + offset].size() > 1 && !tensors[1 + offset].isVirtual()) return false;
                        return
                            (tensors.length == 2 && tensors[0] != null && tensors[1] != null)
                                    ||
                            (tensors.length == 3 && tensors[1] != null && tensors[2] != null);
                    })
                    .suitabilityIfValid(SuitabilityPredicate.VERY_GOOD)
        );
        setCallPreparation(
            call -> {
                int offset = ( call.input( Number.class, 0 ) == null ? 1 : 0 );
                Device<Number> device = call.getDeviceFor(Number.class);
                Shape outShape = call.input( offset ).shape();
                Class<Object> type = (Class<Object>) call.input( offset ).getItemType();
                Tensor output = Tensor.of( type, outShape, 0.0 ).mut().setIsIntermediate( true );
                output.mut().setIsVirtual( false );
                device.store( output );
                if ( call.arity() == 3 ) {
                    assert call.input( 0 ) == null;
                    return call.withInputAt( 0, output );
                }
                else
                    return call.withAddedInputAt( 0, output );
            }
        );
    }

}