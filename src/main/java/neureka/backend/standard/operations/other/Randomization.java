package neureka.backend.standard.operations.other;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Convolution;
import neureka.backend.standard.algorithms.Scalarization;
import neureka.backend.standard.implementations.CPUImplementation;
import neureka.calculus.internal.CalcUtil;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.devices.host.CPU;
import neureka.ndim.iterators.NDIterator;

import java.util.Random;

public class Randomization extends AbstractOperation
{

    public Randomization()
    {
        super(
                new OperationBuilder()
                        .setFunction(         "random"   )
                        .setOperator(         "rand"     )
                        .setArity(            1          )
                        .setIsOperator(       true       )
                        .setIsIndexer(        false      )
                        .setIsDifferentiable( false      )
                        .setIsInline(         true       )
        );

        Scalarization scalarization = new Scalarization()
        .setCanPerformBackwardADFor( call -> true )
        .setCanPerformForwardADFor(
                call -> {
                    if ( call.getOperation().supports(Convolution.class) ) return false;
                    if ( call.getOperation().getOperator().equals(",")   ) return false;
                    Tsr<?> last = null;
                    for ( Tsr<?> t : call.getTensors() ) {
                        if ( last != null && !last.shape().equals(t.shape()) ) return false;
                        last = t; // Note: shapes are cached!
                    }
                    return true;
                }
        )
        .setSupplyADAgentFor( getDefaultAlgorithm() )
        .setExecutionDispatcher( CalcUtil::defaultRecursiveExecution)
        .setCallPreparation(
                call -> {
                    Tsr<?>[] tensors = call.getTensors();
                    int offset = ( tensors[ 0 ] == null ) ? 1 : 0;
                    return
                            ExecutionCall.of(tensors[offset], tensors[1+offset]).andArgs(Arg.DerivIdx.of(-1)).running(Neureka.get().backend().getOperation("idy")).on(call.getDevice());
                }
        )
        .buildFunAlgorithm();

        setAlgorithm(
                Scalarization.class,
                scalarization.setImplementationFor(
                        CPU.class,
                        CPUImplementation
                            .withArity(3)
                            .andImplementation(
                                call -> call.getDevice().getExecutor()
                                        .threaded(
                                                call.getTsrOfType( Number.class, 0 ).size(),
                                                ( i, end ) -> {
                                                    double value = call.getTsrOfType( Number.class, 1 ).getDataAs( double[].class )[ 0 ];
                                                    {
                                                        NDIterator t0Idx = NDIterator.of( call.getTsrOfType( Number.class, 0 ) );
                                                        NDIterator srcIdx = NDIterator.of( call.getTsrOfType( Number.class, 1 ) );
                                                        t0Idx.set( call.getTsrOfType( Number.class, 0 ).indicesOfIndex( i ) );
                                                        srcIdx.set( call.getTsrOfType( Number.class, 1 ).indicesOfIndex( i ) );
                                                        double[] t0_value = call.getTsrOfType( Number.class, 0 ).getDataAs( double[].class );
                                                        while ( i < end ) // increment on drain accordingly:
                                                        {
                                                            int sum = 0;
                                                            int[] indices = srcIdx.get();
                                                            for ( int index : indices ) sum += index;
                                                            Random dice = new Random();
                                                            dice.setSeed( Double.doubleToLongBits( value + sum ) );
                                                            // setInto _value in drn:
                                                            t0_value[ t0Idx.i() ] = dice.nextGaussian();
                                                            // increment on drain:
                                                            t0Idx.increment();
                                                            srcIdx.increment();
                                                            //NDConfiguration.Utility.increment(t0Idx, t0Shp);
                                                            i++;
                                                        }
                                                    }
                                                }
                                        )
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
    public String asDerivative( Function[] children, int derivationIndex) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
            return src[ 0 ].call( inputs, j );
    }
}
