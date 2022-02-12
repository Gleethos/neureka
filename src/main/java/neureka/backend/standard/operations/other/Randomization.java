package neureka.backend.standard.operations.other;

import neureka.backend.api.ExecutionCall;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Activation;
import neureka.backend.standard.implementations.CPUImplementation;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.calculus.internal.CalcUtil;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;
import neureka.ndim.iterators.NDIterator;

import java.util.Arrays;

public class Randomization extends AbstractOperation
{
    private static final long   MULTIPLIER = 0x5DEECE66DL;
    private static final long   ADDEND = 0xBL;
    private static final long   MASK = (1L << 48) - 1;
    private static final double DOUBLE_UNIT = 0x1.0p-53; // 1.0 / (1L << 53)

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
                .setCallPreparation( call ->
                {
                    if ( call.getTensors()[0] == null )
                        call.getTensors()[0] = call.getTensors()[1];

                    call.getTensors()[0].getUnsafe().incrementVersion(call);

                    int hash = Arrays.hashCode( call.getTensors()[0].getNDConf().shape() );
                    Arg.Seed seed = call.get(Arg.Seed.class);
                    if ( seed != null ) seed = Arg.Seed.of( initialScramble(seed.get() + hash) );
                    else seed = Arg.Seed.of( initialScramble(hash) );

                    call.setMetaArg(seed);
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
                                    _newWorkloadFor( call )
                                )
                        )
                )
                .setImplementationFor(
                    OpenCLDevice.class,
                    call -> {
                        throw new IllegalStateException("Not yet implemented");
                    }
                )
        );

    }

    private static CPU.RangeWorkload _newWorkloadFor( ExecutionCall<?> call ) {
        Class<?> type = call.getTensors()[0].getValueClass();
        long seed = call.getValOf(Arg.Seed.class);

        if ( type == Double.class )
            return ( i, end ) -> {

                NDIterator t0Idx = NDIterator.of( call.getTsrOfType( Double.class, 0 ) );
                t0Idx.set( call.getTsrOfType( Double.class, 0 ).indicesOfIndex( i ) );

                double[] t0_value = call.getTsrOfType( Double.class, 0 ).getDataAs( double[].class );
                double[] gaussian = { 0, 0 };

                if ( i % 2 == 1 ) {
                    gaussianFrom(seed + i, gaussian );
                    t0_value[ t0Idx.getIndexAndIncrement() ] = gaussian[0];
                    i++;
                }

                for ( ; i < end; i += 2 ) // increment on drain accordingly:
                {
                    gaussianFrom( seed + i, gaussian );
                    t0_value[ t0Idx.getIndexAndIncrement() ] = gaussian[0];
                    t0_value[ t0Idx.getIndexAndIncrement() ] = gaussian[1];
                }
            };
        else
            return ( i, end ) -> {

                NDIterator t0Idx = NDIterator.of( call.getTsrOfType( Float.class, 0 ) );
                t0Idx.set( call.getTsrOfType( Float.class, 0 ).indicesOfIndex( i ) );

                float[] t0_value = call.getTsrOfType( Float.class, 0 ).getDataAs( float[].class );
                double[] gaussian = { 0, 0 };

                if ( i % 2 == 1 ) {
                    gaussianFrom(seed + i, gaussian );
                    t0_value[ t0Idx.getIndexAndIncrement() ] = (float) gaussian[0];
                    i++;
                }

                for ( ; i < end; i += 2 ) // increment on drain accordingly:
                {
                    gaussianFrom( seed + i, gaussian );
                    t0_value[ t0Idx.getIndexAndIncrement() ] = (float) gaussian[0];
                    t0_value[ t0Idx.getIndexAndIncrement() ] = (float) gaussian[1];
                }
            };

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


    public static long initialScramble(long seed) {
        return (seed ^ MULTIPLIER) & MASK;
    }

    public static void gaussianFrom( long seed, double[] out ) {

        // See Knuth, ACP, Section 3.4.1 Algorithm C.
        double v1, v2, s;
        do {
            long seed1 = _next(seed );
            long seed2 = _next(seed1);
            long seed3 = _next(seed2);
            long seed4 = _next(seed3);
            v1 = 2 * _doubleFrom( seed1, seed2 ) - 1; // between -1 and 1
            v2 = 2 * _doubleFrom( seed3, seed4 ) - 1; // between -1 and 1
            s = v1 * v1 + v2 * v2;
            seed = seed4;
        }
        while ( s >= 1 || s == 0 );

        double multiplier = StrictMath.sqrt( -2 * StrictMath.log(s) / s );

        out[0] = v1 * multiplier;
        out[1] = v2 * multiplier;
    }

    private static long _next( long currentSeed ) {
        long oldseed, nextseed;
        do {
            oldseed = currentSeed;
            nextseed = (oldseed * MULTIPLIER + ADDEND) & MASK;
        } while ( oldseed == (currentSeed = nextseed) );
        return nextseed;
    }

    private static double _doubleFrom( long seed1, long seed2 ) {
        return (((long)(_intFrom(26, seed1)) << 27) + _intFrom(27, seed2)) * DOUBLE_UNIT;
    }

    private static int _intFrom( int bits, long seed ) {
        return (int)(seed >>> (48 - bits));
    }

}
