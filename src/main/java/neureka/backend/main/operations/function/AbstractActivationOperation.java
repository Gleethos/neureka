package neureka.backend.main.operations.function;

import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.backend.main.algorithms.Activation;
import neureka.backend.main.algorithms.ScalarActivation;
import neureka.backend.main.algorithms.ScalarBroadcast;
import neureka.backend.main.algorithms.internal.Fun;
import neureka.backend.main.operations.function.internal.ActivationFun;
import neureka.backend.main.operations.function.internal.CPUScalarBroadcastActivation;
import neureka.calculus.Function;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;
import neureka.ndim.NDimensional;

abstract class AbstractActivationOperation extends AbstractOperation
{
    private final ActivationFun _fun;

    AbstractActivationOperation(ActivationFun fun)
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
                .setImplementationFor( CPU.class, fun.elementwise() )
                .setImplementationFor(
                    OpenCLDevice.class,
                    Activation.implementationForGPU( this.getIdentifier() )
                            .with( fun.activationCode() )
                            .and( fun.derivationCode() )
                )
        );

        setAlgorithm(
            new ScalarBroadcast(Fun.F64ToF64.pair(fun::activate, fun::derive))
            .setAutogradModeFor(
                    call -> call
                            .validate().allNotNullHaveSame(NDimensional::shape)
                            .ifValid(AutoDiffMode.FORWARD_AND_BACKWARD)
                            .orElse(AutoDiffMode.BACKWARD_ONLY)
            )
            .setDeviceExecution( (call, callback) -> AbstractDeviceAlgorithm.executeDeviceAlgorithm( call, callback ) )
            .buildFunAlgorithm()
            .setImplementationFor(
                CPU.class,
                new CPUScalarBroadcastActivation( fun )
            )
        );

        setAlgorithm(
            new ScalarActivation(Fun.F64ToF64.pair(fun::activate, fun::derive))
            .setAutogradModeFor(
                call -> call
                        .validate().allNotNullHaveSame(NDimensional::shape)
                        .ifValid(AutoDiffMode.FORWARD_AND_BACKWARD)
                        .orElse(AutoDiffMode.BACKWARD_ONLY)
            )
            .setDeviceExecution( (call, callback) -> AbstractDeviceAlgorithm.executeDeviceAlgorithm( call, callback ) )
            .buildFunAlgorithm()
            .setImplementationFor(
                CPU.class,
                ScalarActivation.implementationForCPU()
                        .with(Fun.F64ToF64.pair(fun::activate, fun::derive))
                        .with(Fun.F32ToF32.pair(fun::activate, fun::derive))
                        .with(Fun.I32ToI32.pair(fun::activate, fun::derive))
                        .with(Fun.I64ToI64.pair(fun::activate, fun::derive))
                        .with(Fun.I8ToI8.pair(fun::activate, fun::derive))
                        .with(Fun.I16ToI16.pair(fun::activate, fun::derive))
                        .with(Fun.BoolToBool.pair(fun::activate, fun::derive))
                        .with(Fun.CharToChar.pair(fun::activate, fun::derive))
                        .get()
            )
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
