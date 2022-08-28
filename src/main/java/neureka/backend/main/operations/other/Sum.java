package neureka.backend.main.operations.other;

import neureka.Tsr;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.DeviceAlgorithm;
import neureka.backend.api.Result;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.backend.main.operations.linear.internal.opencl.CLSum;
import neureka.backend.main.operations.other.internal.CPUReduce;
import neureka.backend.main.operations.other.internal.CPUSum;
import neureka.calculus.Function;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;

public class Sum extends AbstractOperation
{
    public Sum()
    {
        super(
            new OperationBuilder()
                .identifier(       "sumItems"       )
                .operator(         "sumItems"       )
                .arity(            1           )
                .isOperator(       false       )
                .isIndexer(        false       )
                .isDifferentiable( true        )
                .isInline(         false       )
        );

        setAlgorithm(
            DeviceAlgorithm
            .withName("sum_algorithm")
            .setIsSuitableFor(
                call -> call.validate()
                            .allNotNull( t -> Number.class.isAssignableFrom(t.getItemType()) )
                            .basicSuitability()
            )
            .setAutogradModeFor( call -> AutoDiffMode.BACKWARD_ONLY )
            .setExecution( (caller, call) -> {
                Tsr<?>[] inputs = AbstractDeviceAlgorithm.flatten(caller, call).inputs();
                call = call.withInputs(inputs);
                Tsr<?> result = ((DeviceAlgorithm)call.getAlgorithm()).getImplementationFor(call.getDevice()).run(call);

                return Result.of(
                            result.getUnsafe().setIsIntermediate(true)
                        )
                        .withADAction( target -> {
                            throw new UnsupportedOperationException("The sum operation is not yet supported for auto differentiation!");
                        });
            })
            .setCallPreparation( call ->
             {
                 if ( call.input( 0 ) == null )
                     call = call.withInputAt( 0, call.input( 1 ) );

                 return call;
             })
            .buildFunAlgorithm()
            .setImplementationFor( CPU.class, new CPUSum() )
            .setImplementationFor( OpenCLDevice.class, new CLSum() )
        );
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) { return src[ 0 ].call( inputs, j ); }
}
