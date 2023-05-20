package neureka.backend.main.algorithms;

import neureka.Shape;
import neureka.Tensor;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.Result;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.backend.api.template.algorithms.AbstractFunAlgorithm;

public class ScalarSumAlgorithm extends AbstractFunAlgorithm
{
    public ScalarSumAlgorithm() {
        super("scalar_sum_algorithm");
        setIsSuitableFor(
            call ->
                call.validate()
                .allNotNull( t -> Number.class.isAssignableFrom(t.getItemType()) )
                .allNotNull( t -> t.size() == 1 || t.isVirtual() )
                .suitabilityIfValid( PERFECT ) // You cannot come up with something faster than this! ;D
        )
        .setAutogradModeFor( call -> AutoDiffMode.BACKWARD_ONLY )
        .setExecution( (caller, call) -> {
            Tensor<?>[] inputs = AbstractDeviceAlgorithm.flatten(caller, call).inputs();
            call = call.withInputs(inputs);
            if ( call.input( 0 ) == null )
                call = call.withInputAt( 0, call.input( 1 ) );

            Tensor<?> in = call.input(0);
            Shape originalShape = in.shape();
            Number item = (Number) in.item();
            double sum = item.doubleValue() * in.size();
            Tensor<?> result = Tensor.of( in.itemType(), Shape.of( 1 ), sum ).to( in.getDevice() );
            return Result.of( result.mut().setIsIntermediate(true) )
                    .withADAction( target -> {
                        Tensor<Object> error = (Tensor<Object>) target.error();
                        assert error.size() == 1;
                        return Tensor.of(error.itemType(), originalShape, error.item()).to(error.getDevice());
                    });
        })
        .buildFunAlgorithm();
    }
}
