package neureka.backend.standard.operations.other;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Convolution;
import neureka.backend.standard.algorithms.Scalarization;
import neureka.backend.standard.implementations.CPUImplementation;
import neureka.calculus.CalcUtil;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.devices.Device;
import neureka.devices.host.CPU;

import java.util.Random;

public class Randomization extends AbstractOperation
{

    public Randomization()
    {
        super(
                new OperationBuilder()
                        .setFunction(         "random"    )
                        .setOperator(         "rand"        )
                        .setArity(            1          )
                        .setIsOperator(       true       )
                        .setIsIndexer(        false      )
                        .setIsDifferentiable( false       )
                        .setIsInline(         true      )
        );

        ScalarOperatorCreator< PrimaryNDIConsumer > creator =
                ( inputs, value, d ) -> t1Idx -> {
                    int sum = 0;
                    int[] indices = t1Idx.get();
                    for ( int i : indices ) sum += i;
                    Random dice = new Random();
                    dice.setSeed( Double.doubleToLongBits( value + sum ) );
                    return dice.nextGaussian();
                };

        ScalarOperatorCreator<PrimaryNDAConsumer> creatorX =
                ( inputs, value, d ) -> t1Idx -> {
                    int sum = 0;
                    for ( int indices : t1Idx ) sum += indices;
                    Random dice = new Random();
                    dice.setSeed( Double.doubleToLongBits( value + sum ) );
                    return dice.nextGaussian();
                };

        Scalarization scalarization = new Scalarization()
        .setCanPerformBackwardADFor( call -> true )
        .setCanPerformForwardADFor(
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
        .setSupplyADAgentFor(
            ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                    getDefaultAlgorithm().supplyADAgentFor( f, call, forward )
        )
        .setExecutionDispatcher( CalcUtil::defaultRecursiveExecution)
        .setCallPreparation(
                call -> {
                    Tsr<?>[] tensors = call.getTensors();
                    int offset = ( tensors[ 0 ] == null ) ? 1 : 0;
                    return
                            ExecutionCall.of(tensors[offset], tensors[1+offset]).andArgs(Arg.DerivIdx.of(-1)).running(Neureka.get().context().getOperation("idy")).on(call.getDevice());
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
                                        .threaded (
                                                call.getTsrOfType( Number.class, 0 ).size(),
                                                (Neureka.get().settings().indexing().isUsingArrayBasedIndexing())
                                                ? ( start, end ) ->
                                                        Scalarization.scalarize (
                                                                call.getTsrOfType( Number.class, 0 ),
                                                                start, end,
                                                                creatorX.create(
                                                                        call.getTensors(),
                                                                        call.getTsrOfType( Number.class, 1 ).value64( 0 ),
                                                                        call.getValOf( Arg.DerivIdx.class )
                                                                )
                                                        )
                                                : ( start, end ) ->
                                                        Scalarization.scalarize (
                                                            call.getTsrOfType( Number.class, 0 ),
                                                            start, end,
                                                            creator.create(
                                                                    call.getTensors(),
                                                                    call.getTsrOfType( Number.class, 1 ).value64( 0 ),
                                                                    call.getValOf( Arg.DerivIdx.class )
                                                            )
                                                )
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
