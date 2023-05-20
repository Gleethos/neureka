package neureka.backend.main.algorithms;

import neureka.Shape;
import neureka.Tensor;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.DeviceAlgorithm;
import neureka.backend.api.Result;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.backend.api.template.algorithms.AbstractFunDeviceAlgorithm;

public class SumAlgorithm extends AbstractFunDeviceAlgorithm<SumAlgorithm>
{
    public SumAlgorithm() {
        super("sum_algorithm");
        setIsSuitableFor(
                call -> call.validate()
                        .allNotNull( t -> Number.class.isAssignableFrom(t.getItemType()) )
                        .basicSuitability()
        )
        .setAutogradModeFor( call -> AutoDiffMode.BACKWARD_ONLY )
        .setExecution( (caller, call) -> {
            Tensor<?>[] inputs = AbstractDeviceAlgorithm.flatten(caller, call).inputs();
            call = call.withInputs(inputs);
            Tensor<?> result = ((DeviceAlgorithm)call.getAlgorithm()).getImplementationFor(call.getDevice()).run(call);
            Shape originalShape = call.input(0).shape();
            return Result.of(
                            result.mut().setIsIntermediate(true)
                    )
                    .withADAction( target -> {
                        Tensor<Object> error = (Tensor<Object>) target.error();
                        assert error.size() == 1;
                        return Tensor.of(error.itemType(), originalShape, error.item()).to(error.getDevice());
                    });
        })
        .setCallPreparation( call ->
        {
            if ( call.input( 0 ) == null )
                call = call.withInputAt( 0, call.input( 1 ) );

            return call;
        })
        .buildFunAlgorithm();
    }
}
