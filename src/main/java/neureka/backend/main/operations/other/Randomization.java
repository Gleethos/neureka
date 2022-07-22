package neureka.backend.main.operations.other;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.Result;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.backend.main.algorithms.Activation;
import neureka.backend.main.implementations.CPUImplementation;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.calculus.internal.CalcUtil;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;
import neureka.ndim.iterator.NDIterator;

import java.util.Arrays;

/**
 *  This {@link neureka.backend.api.Operation} takes an optional user seed,
 *  the shape of its input tensor, and
 *  the indices of individual elements within said tensor to generate
 *  floats or doubles with a gaussian distribution where the mean
 *  is 0 and the standard deviation is 1.
 *  This operation is very fast because it generates numbers in parallel unlike
 *  the JDKs random number generator class {@link java.util.Random}.
 */
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
                .setIdentifier(       "random"   )
                .setOperator(         "rand"     )
                .setArity(            1          )
                .setIsOperator(       true       )
                .setIsIndexer(        false      )
                .setIsDifferentiable( false      )
                .setIsInline(         true       )
        );

        setAlgorithm(
            new Activation()
                .setAutogradModeFor( call -> AutoDiffMode.NOT_SUPPORTED)
                .setExecution(
                   (caller, call) ->
                       Result.of(CalcUtil.executeFor( caller, call, CalcUtil::executeDeviceAlgorithm )).withAutoDiff(getDefaultAlgorithm())
                )
                .setCallPreparation( call ->
                {
                    if ( call.input( 0 ) == null )
                        call.setInput( 0, call.input( 1 ) );

                    call.input( 0 ).getUnsafe().incrementVersion(call);

                    int hash = Arrays.hashCode( call.input( 0 ).getNDConf().shape() );
                    Arg.Seed seed = call.get(Arg.Seed.class);
                    if ( seed != null ) seed = Arg.Seed.of( initialScramble(seed.get() + hash) );
                    else seed = Arg.Seed.of( initialScramble(hash) );

                    return call.withArgs(seed);
                })
                .buildFunAlgorithm()
                .setImplementationFor(
                    CPU.class,
                    CPUImplementation
                        .withArity(1)
                        .andImplementation(
                            call ->
                                call
                                .getDevice()
                                .getExecutor()
                                .threaded(
                                    call.input( Number.class, 0 ).size(),
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

    private static CPU.RangeWorkload _newWorkloadFor( ExecutionCall<?> call )
    {
        Tsr<?> tensor = call.input( 0 );
        tensor.setIsVirtual(false);
        Class<?> type = tensor.getItemClass();
        boolean isSimple = tensor.getNDConf().isSimple();
        long seed = call.getValOf(Arg.Seed.class);

        if ( type == Double.class ) {
            if ( isSimple )
                return (i, end) -> {
                    double[] t0_value = tensor.getUnsafe().getDataForWriting(double[].class);
                    double[] gaussian = {0, 0};
                    if ( i % 2 == 1 ) {
                        gaussianFrom(seed + i - 1, gaussian);
                        t0_value[i] = gaussian[1];
                        i++;
                    }
                    for ( ; i < end; i += 2 ) // increment on drain accordingly:
                    {
                        gaussianFrom(seed + i, gaussian);
                        t0_value[i + 0] = gaussian[0];
                        if ( i + 1 < end ) t0_value[i + 1] = gaussian[1];
                    }
                };
            else
                return (i, end) -> {
                    NDIterator t0Idx = NDIterator.of(tensor);
                    t0Idx.set(tensor.indicesOfIndex(i));
                    double[] t0_value = tensor.getUnsafe().getDataForWriting(double[].class);
                    double[] gaussian = {0, 0};
                    if ( i % 2 == 1 ) {
                        gaussianFrom(seed + i - 1, gaussian);
                        t0_value[t0Idx.getIndexAndIncrement()] = gaussian[1];
                        i++;
                    }
                    for ( ; i < end; i += 2 ) // increment on drain accordingly:
                    {
                        gaussianFrom(seed + i, gaussian);
                        t0_value[t0Idx.getIndexAndIncrement()] = gaussian[0];
                        if ( i + 1 < end ) t0_value[t0Idx.getIndexAndIncrement()] = gaussian[1];
                    }
                };
        } else {
            if ( isSimple )
                return (i, end) -> {
                    float[] t0_value = tensor.getUnsafe().getDataForWriting(float[].class);
                    double[] gaussian = {0, 0};
                    if ( i % 2 == 1 ) {
                        gaussianFrom(seed + i - 1, gaussian);
                        t0_value[i] = (float) gaussian[1];
                        i++;
                    }
                    for ( ; i < end; i += 2 ) // increment on drain accordingly:
                    {
                        gaussianFrom(seed + i, gaussian);
                        t0_value[i + 0] = (float) gaussian[0];
                        if ( i + 1 < end ) t0_value[i + 1] = (float) gaussian[1];
                    }
                };
            else
                return (i, end) -> {
                    NDIterator t0Idx = NDIterator.of(tensor);
                    t0Idx.set(tensor.indicesOfIndex(i));
                    float[] t0_value = tensor.getUnsafe().getDataForWriting(float[].class);
                    double[] gaussian = {0, 0};
                    if ( i % 2 == 1 ) {
                        gaussianFrom(seed + i - 1, gaussian);
                        t0_value[t0Idx.getIndexAndIncrement()] = (float) gaussian[1];
                        i++;
                    }
                    for ( ; i < end; i += 2 ) // increment on drain accordingly:
                    {
                        gaussianFrom(seed + i, gaussian);
                        t0_value[t0Idx.getIndexAndIncrement()] = (float) gaussian[0];
                        if ( i + 1 < end ) t0_value[t0Idx.getIndexAndIncrement()] = (float) gaussian[1];
                    }
                };
        }
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

    public static long initialScramble( long seed ) { return (seed ^ MULTIPLIER) & MASK; }

    public static void gaussianFrom( long seed, double[] out )
    {
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

    private static long _next( long currentSeed )
    {
        long oldseed, nextseed;
        do {
            oldseed = currentSeed;
            nextseed = (oldseed * MULTIPLIER + ADDEND) & MASK;
        } while ( oldseed == (currentSeed = nextseed) );
        return nextseed;
    }

    private static double _doubleFrom( long seed1, long seed2 )
    {
        return (((long)(_intFrom(26, seed1)) << 27) + _intFrom(27, seed2)) * DOUBLE_UNIT;
    }

    private static int _intFrom( int bits, long seed ) { return (int)(seed >>> (48 - bits)); }

}
