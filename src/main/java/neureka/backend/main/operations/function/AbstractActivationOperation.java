package neureka.backend.main.operations.function;

import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.backend.main.algorithms.Activation;
import neureka.backend.main.algorithms.ScalarActivation;
import neureka.backend.main.algorithms.ScalarBroadcast;
import neureka.backend.main.functions.ScalarFun;
import neureka.backend.main.implementations.scalar.CLScalarActivation;
import neureka.backend.main.implementations.scalar.CPUElementwiseActivation;
import neureka.backend.main.implementations.scalar.CPUScalarActivation;
import neureka.backend.main.implementations.scalar.CPUScalarBroadcastActivation;
import neureka.calculus.Function;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;
import neureka.ndim.NDimensional;

abstract class AbstractActivationOperation extends AbstractOperation
{
    private final ScalarFun _fun;

    AbstractActivationOperation(ScalarFun fun)
    {
        super(
            new OperationBuilder()
                .identifier(      fun.id()       )
                .operator(        fun.id()       )
                .arity(            1             )
                .isOperator(       false         )
                .isIndexer(        false         )
                .isDifferentiable( true          )
                .isInline(         false         )
        );
        _fun = fun;
        setAlgorithm(
            new Activation().setSupplyADAgentFor( getDefaultAlgorithm() ).buildFunAlgorithm()
                .setImplementationFor( CPU.class, new CPUElementwiseActivation( fun ) )
                .setImplementationFor(
                    OpenCLDevice.class,
                    Activation.implementationForGPU( this.getIdentifier() )
                            .with( fun.activationCode() )
                            .and( fun.derivationCode() )
                )
        );

        setAlgorithm(
            new ScalarBroadcast(fun)
            .setAutogradModeFor(
                    call -> call
                            .validate().allNotNullHaveSame(NDimensional::shape)
                            .ifValid(AutoDiffMode.FORWARD_AND_BACKWARD)
                            .orElse(AutoDiffMode.BACKWARD_ONLY)
            )
            .setDeviceExecution( (call, callback) -> AbstractDeviceAlgorithm.executeDeviceAlgorithm( call, callback ) )
            .buildFunAlgorithm()
            .setImplementationFor( CPU.class, new CPUScalarBroadcastActivation( fun ) )
        );

        setAlgorithm(
            new ScalarActivation()
            .setAutogradModeFor(
                call -> call
                        .validate().allNotNullHaveSame(NDimensional::shape)
                        .ifValid(AutoDiffMode.FORWARD_AND_BACKWARD)
                        .orElse(AutoDiffMode.BACKWARD_ONLY)
            )
            .setDeviceExecution( (call, callback) -> AbstractDeviceAlgorithm.executeDeviceAlgorithm( call, callback ) )
            .buildFunAlgorithm()
            .setImplementationFor( CPU.class, new CPUScalarActivation( fun ) )
            .setImplementationFor( OpenCLDevice.class, new CLScalarActivation( fun ) )
        );
    }

    @Override
    public final String stringify( String[] children ) {
        String expression = String.join( ", ", children );
        if ( expression.startsWith("(") && expression.endsWith(")") ) return getIdentifier() + expression;
        return getIdentifier() + "(" + expression + ")";
    }

    @Override
    public final double calculate( double[] inputs, int j, int d, Function[] src ) {
        boolean derive = d >= 0;
        double inner = ( !derive ? 1 : src[ 0 ].derive( inputs, d, j ) );
        return _fun.calculate( src[ 0 ].call( inputs, j ),  derive ) * inner;
    }

}
