package neureka.backend.main.operations.operator;

import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.backend.main.algorithms.BiElementWise;
import neureka.backend.main.algorithms.Broadcast;
import neureka.backend.main.algorithms.Scalarization;
import neureka.backend.main.operations.ElemWiseUtil;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.devices.Device;

import java.util.Arrays;
import java.util.stream.Collectors;

public class Addition extends AbstractOperation {

    public Addition()
    {
        super (
            new OperationBuilder()
            .identifier(       "add"      )
            .operator(         "+"        )
            .arity(            -1         )
            .isOperator(       true       )
            .isIndexer(        false      )
            .isDifferentiable( true       )
            .isInline(         false      )
        );

        setAlgorithm(
            new BiElementWise(ElemWiseUtil::forAdditions)
            .setSupplyADAgentFor( getDefaultAlgorithm() )
            .buildFunAlgorithm()
        );

        setAlgorithm(
            new Broadcast( AbstractDeviceAlgorithm::executeDeviceAlgorithm )
            .setAutogradModeFor( call -> AutoDiffMode.BACKWARD_ONLY )
            .setSupplyADAgentFor(
                ( Function f, ExecutionCall<? extends Device<?>> call ) ->
                {
                    if ( call.autogradMode().allowsForward() )
                        throw new IllegalArgumentException("Broadcast implementation does not support forward-AD!");
                    Tsr<?> ctxDerivative = (Tsr<?>) call.getValOf(Arg.Derivative.class);
                    assert ctxDerivative == null;
                    int d = call.getDerivativeIndex();
                    Tsr<?> derivative = ElemWiseUtil.newTsrLike(call.input( d==0?1:0 ), 0);
                    Tsr<?> toBeDerived = ElemWiseUtil.newTsrLike(call.input( d ), 0);
                    Device device = call.getDeviceFor(Number.class);
                    return ADAgent.of( derivative )
                            .withAD(
                                target ->
                                    this.getAlgorithm( Broadcast.class )
                                         .getImplementationFor( device )
                                         .run(
                                             ExecutionCall.of(
                                                 toBeDerived.setIsVirtual(false),
                                                 derivative,
                                                 target.error()
                                             )
                                             .andArgs( Arg.DerivIdx.of(d) )
                                             .running( this )
                                             .on( device )
                                         )
                            );
                }
            )
            .buildFunAlgorithm()
        );

        setAlgorithm(
            new Scalarization()
            .setDeviceExecution( (call, callback) -> ElemWiseUtil.forAdditions(call, callback) )
            .buildFunAlgorithm()
        );
    }

    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        return Arrays.stream( children )
                    .filter( child -> child.dependsOn(derivationIndex) )
                    .map( child -> child.getDerivative(derivationIndex) )
                    .map( Object::toString )
                    .collect( Collectors.joining( " "+getOperator()+" " ) );
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        if ( j < 0 ) return calculate( inputs, d, src );
        if ( d < 0 ) {
            double result = src[ 0 ].call( inputs, j );
            for ( int i = 1; i < src.length; i++ ) {
                final double current = src[ i ].call( inputs, j );
                result += current;
            }
            return result;
        } else {
            double derivative = 0;
            for ( Function function : src )
                derivative += function.derive(inputs, d, j);

            return derivative;
        }
    }

    
    public static double calculate( double[] inputs, int d, Function[] src ) {
        if ( d < 0 ) {
            double result = src[ 0 ].call( inputs );
            for ( int i = 1; i < src.length; i++ ) {
                final double current = src[ i ].call( inputs );
                result += current;
            }
            return result;
        } else {
            double derivative = 0;
            for ( Function function : src )
                derivative += function.derive( inputs, d );

            return derivative;
        }
    }




}
