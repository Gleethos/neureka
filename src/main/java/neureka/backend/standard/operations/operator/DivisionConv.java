package neureka.backend.standard.operations.operator;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.DefaultADAgent;
import neureka.calculus.CalcUtil;
import neureka.calculus.args.Arg;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Convolution;
import neureka.calculus.Function;
import neureka.devices.Device;

public class DivisionConv extends AbstractOperation {

    public DivisionConv() {
        super(
                new OperationBuilder()
                        .setFunction(         "divide_conv"           )
                        .setOperator(         "d"                )
                        .setArity(            2                  )
                        .setIsOperator(       true               )
                        .setIsIndexer(        false              )
                        .setIsDifferentiable( true               )
                        .setIsInline(         false              )
        );
        setAlgorithm(
                Convolution.class,
                new Convolution()
                        .setCanPerformBackwardADFor( call -> true )
                        .setCanPerformForwardADFor(
                                call -> {
                                    Tsr<?> last = null;
                                    for ( Tsr<?> t : call.getTensors() ) {
                                        if ( last != null && !last.shape().equals(t.shape()) ) return false;
                                        last = t; // Note: shapes are cached!
                                    }
                                    return true;
                                }
                        )
                        .setSupplyADAgentFor(
                                (Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                                {
                                    Tsr<?> ctxDerivative = (Tsr<?>) call.getValOf(Arg.Derivative.class);
                                    Function mul = Neureka.get().context().getFunction().mul();
                                    if ( ctxDerivative != null ) {
                                        return new DefaultADAgent( ctxDerivative )
                                                .setForward( (node, forwardDerivative ) -> mul.call( new Tsr[]{ forwardDerivative, ctxDerivative } ) )
                                                .setBackward( (node, forwardDerivative ) -> mul.call( new Tsr[]{ forwardDerivative, ctxDerivative } ) );
                                    }
                                    Tsr[] inputs = call.getTensors();
                                    int d = call.getDerivativeIndex();
                                    if ( forward )
                                        throw new IllegalArgumentException("Convolution of does not support forward-AD!");
                                    else
                                    {
                                        Tsr<?> localDerivative = f.derive( inputs, d );
                                        return new DefaultADAgent( localDerivative )
                                                .setForward( (node, forwardDerivative ) -> mul.call(new Tsr[]{forwardDerivative, localDerivative}) )
                                                .setBackward( (node, backwardError ) -> mul.call(new Tsr[]{backwardError, localDerivative}) );
                                    }
                                }
                        )
                        .setHandleInsteadOfDevice( CalcUtil::executeFor)
                        .setHandleRecursivelyAccordingToArity( (call, goDeeperWith ) -> null )
                        .setInstantiateNewTensorsForExecutionIn(
                                call -> {
                                    Tsr[] tsrs = call.getTensors();
                                    int offset = ( tsrs[ 0 ] == null ) ? 1 : 0;
                                    return ExecutionCall.of(tsrs[offset], tsrs[1+offset]).andArgs(Arg.DerivIdx.of(-1)).running(Neureka.get().context().getOperation("idy")).on(call.getDevice());
                                }
                        )
                        .build()
        );
    }

    @Override
    public String stringify(String[] children) {
        StringBuilder reconstructed = new StringBuilder();
        for ( int i = 0; i < children.length; ++i ) {
            reconstructed.append( children[ i ] );
            if ( i < children.length - 1 ) {
                reconstructed.append(" d ");
            }
        }
        return "(" + reconstructed + ")";
    }

    @Override
    public String asDerivative(Function[] children, int derivationIndex) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        return 0;
    }

}
