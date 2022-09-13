package neureka.backend.main.algorithms;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.fun.ADActionSupplier;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.backend.api.template.algorithms.AbstractFunDeviceAlgorithm;
import neureka.backend.main.algorithms.internal.WithForward;
import neureka.backend.main.implementations.CLImplementation;
import neureka.calculus.args.Arg;
import neureka.devices.Device;
import neureka.dtype.NumericType;

/**
 *  This is lambda based {@link neureka.backend.api.Algorithm} implementation
 *  providing some basic functionality for implementing custom
 *  activation functions.
 */
public final class Activation extends AbstractFunDeviceAlgorithm<Activation>
{
    public Activation() {
        super("activation");
        setIsSuitableFor(
           call -> call.validate()
                       .allNotNull( t -> t.getDataType().typeClassImplements(NumericType.class) )
                       .basicSuitability()
        );
        setAutogradModeFor(
            call ->
                call
                    .validate()
                    .all( ( first, second ) -> first.shape().equals(second.shape()) )
                    .ifValid(AutoDiffMode.FORWARD_AND_BACKWARD)
                    .orElse(AutoDiffMode.BACKWARD_ONLY)
        );
        setDeviceExecution( (call, callback) -> AbstractDeviceAlgorithm.executeDeviceAlgorithm(call, callback), (ADActionSupplier) null );
        setCallPreparation(
            call -> {
                Device device = call.getDeviceFor(Number.class);
                if ( call.input(  0 ) == null ) // Creating a new tensor:
                {
                    int[] shape = call.input(  1 ).getNDConf().shape();
                    Class<Object> type = (Class<Object>) call.input(  1 ).getItemType();
                    Tsr<Object> output = Tsr.of(type).withShape(shape).all( 0.0 ).getUnsafe().setIsIntermediate( true );
                    output.setIsVirtual( false );
                    device.store( output );
                    call = call.withInputAt( 0, output );
                }
                return call;
            }
        );
    }

}
