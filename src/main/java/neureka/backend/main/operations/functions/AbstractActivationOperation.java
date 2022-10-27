package neureka.backend.main.operations.functions;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.backend.api.Result;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.backend.api.template.algorithms.FallbackAlgorithm;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.backend.main.algorithms.Activation;
import neureka.backend.main.algorithms.ScalarActivation;
import neureka.backend.main.algorithms.ScalarBroadcast;
import neureka.backend.main.implementations.fun.api.ScalarFun;
import neureka.backend.main.implementations.scalar.CPUScalarBroadcastFunction;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.calculus.assembly.FunctionParser;
import neureka.devices.host.CPU;
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
            new Activation()
                .setSupplyADActionFor( getDefaultAlgorithm() )
                .buildFunAlgorithm()
        );

        setAlgorithm(
            new ScalarBroadcast(fun)
            .setAutogradModeFor(
                    call -> call
                            .validate().allNotNullHaveSame(NDimensional::shape)
                            .ifValid(AutoDiffMode.FORWARD_AND_BACKWARD)
                            .orElse(AutoDiffMode.BACKWARD_ONLY)
            )
            .setExecution( (caller, call) -> Result.of(AbstractDeviceAlgorithm.executeFor(caller, call, AbstractDeviceAlgorithm::executeDeviceAlgorithm)).withAutoDiff( FallbackAlgorithm::ADAction ))
            .buildFunAlgorithm()
            .setImplementationFor( CPU.class, new CPUScalarBroadcastFunction( fun ) )
        );

        setAlgorithm(
            new ScalarActivation()
            .setAutogradModeFor(
                call -> call
                        .validate().allNotNullHaveSame(NDimensional::shape)
                        .ifValid(AutoDiffMode.FORWARD_AND_BACKWARD)
                        .orElse(AutoDiffMode.BACKWARD_ONLY)
            )
            .setExecution( (caller, call) -> Result.of(AbstractDeviceAlgorithm.executeFor(caller, call, AbstractDeviceAlgorithm::executeDeviceAlgorithm)).withAutoDiff( FallbackAlgorithm::ADAction ))
            .buildFunAlgorithm()
        );
    }

    @Override
    public Result execute(Function caller, ExecutionCall<?> call )
    {
        if ( !caller.isFlat() ) {
            int d = call.getDerivativeIndex();
            ExecutionCall<?> flatCall = AbstractDeviceAlgorithm.flatten( caller, call.withArgs(Arg.DerivIdx.of(-1)) );
            if ( d < 0 ) {
                Function flat = new FunctionParser(Neureka.get().backend()).parse( flatCall.getOperation(), flatCall.arity(), true );
                return super.execute( flat, flatCall );
            } else {
                Function noAdFun = Function.of( caller.toString(), false );
                Function innerFun = noAdFun.getSubFunctions().get(0);
                Function flat = new FunctionParser(Neureka.get().backend()).parse( flatCall.getOperation(), flatCall.arity(), false );
                // The user wants the derivative! So we need to do inner times outer derivative! (because the function is not flat)
                ExecutionCall<?> inner = AbstractDeviceAlgorithm.flatten( noAdFun, call.withArgs(Arg.DerivIdx.of(-1)) );
                Result innerDerivResult = innerFun.getOperation().execute( innerFun, call.withOperation(innerFun.getOperation()) );
                Tsr<?> innerDeriv = innerDerivResult.get();
                Tsr<?> outerDeriv = super.execute( flat, inner.withArgs(Arg.DerivIdx.of(0)) ).get();
                Operation mul = Neureka.get().backend().getOperation("*");
                Function opFun = new FunctionParser(Neureka.get().backend()).parse( mul, 2, false );
                return mul.execute(
                            opFun,
                            ExecutionCall.of( innerDeriv, outerDeriv )
                                            .running(mul)
                                            .on(call.getDevice())
                        );
            }
        }
        return super.execute( caller, call );
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
