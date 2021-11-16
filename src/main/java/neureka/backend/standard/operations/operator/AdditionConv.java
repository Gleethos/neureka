package neureka.backend.standard.operations.operator;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Convolution;
import neureka.calculus.CalcUtil;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.devices.Device;

public class AdditionConv extends AbstractOperation {

    public AdditionConv() {
        super(
                new OperationBuilder()
                        .setFunction(         "add_conv"              )
                        .setOperator(         "a"                )
                        .setArity(            2                  )
                        .setIsOperator(       true               )
                        .setIsIndexer(        false              )
                        .setIsDifferentiable( false              )
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
                                    Function mul = Neureka.get().backend().getFunction().mul();
                                    if ( ctxDerivative != null ) {
                                        return ADAgent.of( ctxDerivative )
                                                        .setForward( (node, forwardDerivative ) -> mul.execute( forwardDerivative, ctxDerivative ) )
                                                        .setBackward( (node, forwardDerivative ) -> mul.execute( forwardDerivative, ctxDerivative ) );
                                    }
                                    Tsr<?>[] inputs = call.getTensors();
                                    int d = call.getDerivativeIndex();
                                    if ( forward )
                                        throw new IllegalArgumentException("Convolution does not support forward-AD!");
                                    else
                                    {
                                        Tsr<?> localDerivative = f.executeDerive( inputs, d );
                                        return ADAgent.of( localDerivative )
                                                        .setForward( (node, forwardDerivative ) -> mul.execute( forwardDerivative, localDerivative ) )
                                                        .setBackward( (node, backwardError ) -> mul.execute( backwardError, localDerivative ) );
                                    }
                                }
                        )
                        .setExecutionDispatcher( CalcUtil::defaultRecursiveExecution)
                        .setCallPreparation(
                                call -> {
                                    Tsr<?>[] tensors = call.getTensors();
                                    int offset = ( tensors[ 0 ] == null ) ? 1 : 0;
                                    return ExecutionCall.of(tensors[offset], tensors[1+offset])
                                                        .andArgs(Arg.DerivIdx.of(-1))
                                                        .running(Neureka.get().backend().getOperation("idy"))
                                                        .on(call.getDevice());
                                }
                        )
                        .buildFunAlgorithm()
        );
    }

    @Override
    public String stringify(String[] children) {
        StringBuilder reconstructed = new StringBuilder();
        for ( int i = 0; i < children.length; ++i ) {
            reconstructed.append( children[ i ] );
            if ( i < children.length - 1 ) {
                reconstructed.append(" a ");
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
