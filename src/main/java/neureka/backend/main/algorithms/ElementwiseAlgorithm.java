package neureka.backend.main.algorithms;

import neureka.Tensor;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.Result;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.backend.api.template.algorithms.AbstractFunDeviceAlgorithm;
import neureka.devices.Device;
import neureka.dtype.NumericType;

/**
 *  This is lambda based {@link neureka.backend.api.Algorithm} implementation
 *  providing some basic functionality for implementing custom
 *  activation functions.
 */
public final class ElementwiseAlgorithm extends AbstractFunDeviceAlgorithm<ElementwiseAlgorithm>
{
    public ElementwiseAlgorithm() {
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
        setExecution( (outerCaller, outerCall) ->
                Result.of(AbstractDeviceAlgorithm.prepareAndExecute(
                        outerCall,
                        innerCall -> AbstractDeviceAlgorithm.executeDeviceAlgorithm( innerCall )
                ))
        );
        setCallPreparation(
            call -> {
                Device device = call.getDeviceFor(Number.class);
                if ( call.arity() < 2 ) call = call.withAddedInputAt(0, null);
                if ( call.input(  0 ) == null ) // Creating a new tensor:
                {
                    int[] shape = call.input(  1 ).getNDConf().shape();
                    Class<Object> type = (Class<Object>) call.input(  1 ).getItemType();
                    Tensor<Object> output = Tensor.of(type).withShape(shape).all( 0.0 ).mut().setIsIntermediate( true );
                    output.mut().setIsVirtual( false );
                    device.store( output );
                    call = call.withInputAt( 0, output );
                }
                return call;
            }
        );
    }

}
