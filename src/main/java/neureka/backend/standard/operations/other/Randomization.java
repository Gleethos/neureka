package neureka.backend.standard.operations.other;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationContext;
import neureka.backend.api.operations.OperationFactory;
import neureka.backend.standard.algorithms.Convolution;
import neureka.backend.standard.algorithms.Scalarization;
import neureka.backend.standard.implementations.HostImplementation;
import neureka.calculus.Function;
import neureka.devices.Device;
import neureka.devices.host.HostCPU;

import java.util.List;
import java.util.Random;

public class Randomization extends AbstractOperation
{

    public Randomization()
    {
        super(
                new OperationFactory()
                        .setFunction(         "random"    )
                        .setOperator(         "rand"        )
                        .setArity(            1          )
                        .setIsOperator(       true       )
                        .setIsIndexer(        false      )
                        .setIsDifferentiable( false       )
                        .setIsInline(         true      )
        );

        ScalarOperatorCreator< PrimaryNDIConsumer > creator =
                ( inputs, value, d ) -> {
                    return t1Idx -> {
                        int sum = 0;
                        int[] indices = t1Idx.get();
                        for ( int i : indices ) sum += i;
                        Random dice = new Random();
                        dice.setSeed( Double.doubleToLongBits( value + sum ) );
                        return dice.nextGaussian();
                    };
                };

        ScalarOperatorCreator< PrimaryNDXConsumer > creatorX =
                ( inputs, value, d ) -> {
                    return t1Idx -> {
                        int sum = 0;
                        for ( int indices : t1Idx) sum += indices;
                        Random dice = new Random();
                        dice.setSeed(Double.doubleToLongBits(value+sum));
                        return dice.nextGaussian();
                    };
                };

        Scalarization scalarization = new Scalarization()
        .setBackwardADAnalyzer( call -> true )
        .setForwardADAnalyzer(
                call -> {
                    if ( call.getOperation().supports(Convolution.class) ) return false;
                    if ( call.getOperation().getOperator().equals(",") ) return false; //Reshape
                    Tsr<?> last = null;
                    for ( Tsr<?> t : call.getTensors() ) {
                        if ( last != null && !last.shape().equals(t.shape()) ) return false;
                        last = t; // Note: shapes are cached!
                    }
                    return true;
                }
        )
        .setADAgentSupplier(
            ( Function f, ExecutionCall<Device> call, boolean forward ) ->
                    getDefaultAlgorithm().supplyADAgentFor( f, call, forward )
        )
        .setCallHook( (caller, call ) -> null )
        .setRJAgent( ( call, goDeeperWith ) -> null )
        .setDrainInstantiation(
                call -> {
                    Tsr[] tsrs = call.getTensors();
                    int offset = ( tsrs[ 0 ] == null ) ? 1 : 0;
                    return
                            ExecutionCall.builder()
                                .device( call.getDevice() )
                                .tensors( new Tsr[]{tsrs[offset], tsrs[1+offset]} )
                                .derivativeIndex( -1 )
                                .operation( OperationContext.get().instance("idy") )
                                .build();
                }
        )
        .build();

        setAlgorithm(
                Scalarization.class,
                scalarization.setImplementationFor(
                        HostCPU.class,
                        new HostImplementation(
                                call -> call.getDevice().getExecutor()
                                        .threaded (
                                                call.getTensor( 0 ).size(),
                                                (Neureka.instance().settings().indexing().isUsingArrayBasedIndexing())
                                                ? ( start, end ) ->
                                                        Scalarization.scalarize (
                                                                call.getTensor( 0 ),
                                                                start, end,
                                                                creatorX.create(
                                                                        call.getTensors(),
                                                                        call.getTensor( 1 ).value64( 0 ),
                                                                        call.getDerivativeIndex()
                                                                )
                                                        )
                                                : ( start, end ) ->
                                                        Scalarization.scalarize (
                                                            call.getTensor( 0 ),
                                                            start, end,
                                                            creator.create(
                                                                    call.getTensors(),
                                                                    call.getTensor( 1 ).value64( 0 ),
                                                                    call.getDerivativeIndex()
                                                            )
                                                )
                                        ),
                                3
                        )
                )
        );

    }

    @Override
    public String stringify( String[] children ) {
        String expression = String.join( ", ", children );
        if (expression.charAt( 0 ) == '(' && expression.charAt( expression.length() - 1 ) == ')') {
            return "rand" + expression;
        }
        return "rand" + "(" + expression + ")";
    }

    @Override
    public String asDerivative( Function[] children, int d ) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
            return src[ 0 ].call( inputs, j );
    }
}
