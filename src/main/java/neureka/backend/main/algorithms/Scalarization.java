package neureka.backend.main.algorithms;

import neureka.Tsr;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.fun.SuitabilityPredicate;
import neureka.backend.api.template.algorithms.AbstractFunDeviceAlgorithm;
import neureka.devices.Device;
import neureka.dtype.NumericType;

public class Scalarization extends AbstractFunDeviceAlgorithm<Scalarization>
{
    public Scalarization() {
        super("scalarization");
        setAutogradModeFor( call -> AutoDiffMode.FORWARD_AND_BACKWARD );
        setIsSuitableFor( call ->
            call.validate()
                .allNotNull( t -> t.getDataType().typeClassImplements(NumericType.class) )
                //.first( Objects::isNull )
                .tensors( tensors ->  {
                    if ( tensors.length != 2 && tensors.length != 3 ) return false;
                    int offset = ( tensors.length == 2 ? 0 : 1 );
                    if ( tensors[1+offset].size() > 1 && !tensors[1+offset].isVirtual() ) return false;
                    return
                        (tensors.length == 2 && tensors[0] == null && tensors[1] != null)
                        ||
                        //tensors[1+offset].shape().stream().allMatch( d -> d == 1 )
                        //||
                        tensors[offset].shape().equals(tensors[1+offset].shape());
                })
                .suitabilityIfValid( SuitabilityPredicate.VERY_GOOD )
        );
        setCallPreparation(
            call -> {
                Device<Number> device = call.getDeviceFor(Number.class);
                assert call.input( 0 ) == null;  // Creating a new tensor:

                int[] outShape = call.input( 1 ).getNDConf().shape();
                Class<Object> type = (Class<Object>) call.input( 1 ).getItemType();
                Tsr output = Tsr.of( type, outShape, 0.0 ).getMut().setIsIntermediate( true );
                output.getMut().setIsVirtual( false );
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