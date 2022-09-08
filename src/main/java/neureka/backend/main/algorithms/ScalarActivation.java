package neureka.backend.main.algorithms;

import neureka.Tsr;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.fun.SuitabilityPredicate;
import neureka.backend.api.template.algorithms.AbstractFunDeviceAlgorithm;
import neureka.devices.Device;
import neureka.dtype.NumericType;

public class ScalarActivation extends AbstractFunDeviceAlgorithm<ScalarActivation>
{
    public ScalarActivation() {
        super("scalar activation");
        setAutogradModeFor( call -> AutoDiffMode.FORWARD_AND_BACKWARD );
        setIsSuitableFor( call ->
            call.validate()
                .allNotNull( t -> t.getDataType().typeClassImplements(NumericType.class) )
                .tensors( tensors ->  {
                    if ( tensors.length != 2 ) return false;
                    if ( !tensors[1].isVirtual() ) return false;
                    if ( tensors[0] != null && !tensors[0].isVirtual() ) return false;
                    return tensors[0] == null && tensors[1] != null || tensors[0].shape().equals(tensors[1].shape());
                })
                .suitabilityIfValid( SuitabilityPredicate.EXCELLENT )
        );
        setCallPreparation(
            call -> {
                Device<Number> device = call.getDeviceFor(Number.class);
                assert call.input( 0 ) == null;  // Creating a new tensor:
                int[] outShape = call.input( 1 ).getNDConf().shape();
                Class<Object> type = (Class<Object>) call.input( 1 ).getItemType();
                Tsr output = Tsr.of( type, outShape, 0.0 ).getUnsafe().setIsIntermediate( true );
                try {
                    device.store( output );
                } catch( Exception e ) {
                    e.printStackTrace();
                }
                return call.withInputAt( 0, output );
            }
        );
    }

}