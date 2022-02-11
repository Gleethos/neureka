package neureka.backend.standard.operations.other;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Activation;
import neureka.backend.standard.algorithms.Scalarization;
import neureka.backend.standard.implementations.CPUImplementation;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.calculus.internal.CalcUtil;
import neureka.devices.host.CPU;
import neureka.ndim.iterators.NDIterator;

import java.util.Arrays;

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

        setAlgorithm(
            new Activation()
                .setCanPerformBackwardADFor( call -> false )
                .setCanPerformForwardADFor( call -> false )
                .setSupplyADAgentFor( getDefaultAlgorithm() )
                .setExecutionDispatcher( CalcUtil::defaultRecursiveExecution )
                .setCallPreparation( call -> {
                    if ( call.getTensors()[0] == null )
                        call.getTensors()[0] = call.getTensors()[1];
                    return call;
                })
                .buildFunAlgorithm()
                .setImplementationFor(
                    CPU.class,
                    CPUImplementation
                        .withArity(3)
                        .andImplementation(
                            call ->
                                call
                                .getDevice()
                                .getExecutor()
                                .threaded(
                                    call.getTsrOfType( Number.class, 0 ).size(),
                                    ( i, end ) -> {

                                        NDIterator t0Idx = NDIterator.of( call.getTsrOfType( Number.class, 0 ) );
                                        t0Idx.set( call.getTsrOfType( Number.class, 0 ).indicesOfIndex( i ) );

                                        long seed = Arrays.hashCode( call.getTsrOfType( Number.class, 0 ).getNDConf().shape() );

                                        double[] t0_value = call.getTsrOfType( Number.class, 0 ).getDataAs( double[].class );
                                        double[] gaussian = { 0, 0 };

                                        if ( i % 2 == 1 ) {
                                            _gaussianFrom(seed + i, gaussian );
                                            // setting value in output:
                                            t0_value[ t0Idx.i() ] = gaussian[0];
                                            t0Idx.increment();
                                            i++;
                                        }

                                        for ( ; i < end; i += 2 ) // increment on drain accordingly:
                                        {
                                            _gaussianFrom( seed + i, gaussian );
                                            // setting value in output:
                                            t0_value[ t0Idx.i() ] = gaussian[0];

                                            // increment on drain:
                                            t0Idx.increment();

                                            t0_value[ t0Idx.i() ] = gaussian[1];

                                            // increment on drain:
                                            t0Idx.increment();
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
        if ( expression.charAt( 0 ) == '(' && expression.charAt( expression.length() - 1 ) == ')' ) {
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

    private static long _pow( long number, long power ) {
        if ( power == 0 ) return 1;
        long result = number;
        while ( power > 1 ) {
            result *= number;
            power--;
        }
        return result;
    }

    private void _gaussianFrom( long seed, double[] out ) {
        // See Knuth, ACP, Section 3.4.1 Algorithm C.
        double v1, v2, s;
        do {
            v1 = 2 * _doubleFrom( _pow( 7,  Math.abs(seed) ) ) - 1; // between -1 and 1
            v2 = 2 * _doubleFrom( _pow( 11, Math.abs(seed) ) ) - 1; // between -1 and 1
            s = v1 * v1 + v2 * v2;
            seed += 2;
        }
        while ( s >= 1 || s == 0 );

        double multiplier = StrictMath.sqrt( -2 * StrictMath.log(s) / s );

        out[0] = v1 * multiplier;
        out[1] = v2 * multiplier;
    }

    private static double _doubleFrom( long seed ) {
        return (((long)(_intFrom(26, seed)) << 27) + _intFrom(27, seed+42)) * DOUBLE_UNIT;
    }

    private static int _intFrom( int bits, long seed ) {
        long nextSeed = (seed * MULTIPLIER + ADDEND) & MASK;
        return (int)(nextSeed >>> (48 - bits));
    }

    private static final long MULTIPLIER = 0x5DEECE66DL;
    private static final long ADDEND = 0xBL;
    private static final long MASK = (1L << 48) - 1;
    private static final double DOUBLE_UNIT = 0x1.0p-53; // 1.0 / (1L << 53)

}
