package neureka.backend.main.algorithms;

import neureka.Shape;
import neureka.Tsr;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.Result;
import neureka.backend.api.fun.SuitabilityPredicate;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.backend.api.template.algorithms.AbstractFunDeviceAlgorithm;
import neureka.backend.api.template.algorithms.FallbackAlgorithm;
import neureka.devices.Device;
import neureka.dtype.NumericType;
import neureka.ndim.NDimensional;

public class ScalarAlgorithm extends AbstractFunDeviceAlgorithm<ScalarAlgorithm>
{
    public ScalarAlgorithm() {
        super("scalar activation");
        setAutogradModeFor(
                call -> call
                        .validate().allNotNullHaveSame(NDimensional::shape)
                        .ifValid(AutoDiffMode.FORWARD_AND_BACKWARD)
                        .orElse(AutoDiffMode.BACKWARD_ONLY)
        );
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
                Shape outShape = call.input( 1 ).shape();
                Class<Object> type = (Class<Object>) call.input( 1 ).getItemType();
                Tsr output = Tsr.of( type, outShape, 0.0 ).mut().setIsIntermediate( true );
                device.store( output );
                return call.withInputAt( 0, output );
            }
        );
        setExecution(
            (caller, call) ->
                Result.of(AbstractDeviceAlgorithm.prepareAndExecute(call,AbstractDeviceAlgorithm::executeDeviceAlgorithm))
                        .withAutoDiff( FallbackAlgorithm::ADAction )
        );

    }

}