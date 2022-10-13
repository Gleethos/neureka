package neureka.backend.main.algorithms;

import neureka.Tsr;
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
            Tsr<?>[] inputs = AbstractDeviceAlgorithm.flatten(caller, call).inputs();
            call = call.withInputs(inputs);
            Tsr<?> result = ((DeviceAlgorithm)call.getAlgorithm()).getImplementationFor(call.getDevice()).run(call);
            int[] originalShape = call.input(0).getNDConf().shape();
            return Result.of(
                            result.mut().setIsIntermediate(true)
                    )
                    .withADAction( target -> {
                        Tsr<Object> error = (Tsr<Object>) target.error();
                        assert error.size() == 1;
                        return Tsr.of(error.itemType(), originalShape, error.item()).to(error.getDevice());
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
