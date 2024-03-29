package neureka.backend.main.implementations.elementwise;

import neureka.Tensor;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.ImplementationFor;
import neureka.math.args.Arg;
import neureka.devices.host.CPU;
import neureka.ndim.iterator.NDIterator;

public strictfp class CPURandomization implements ImplementationFor<CPU>
{
    private static final long   MULTIPLIER = 0x5DEECE66DL;
    private static final long   ADDEND = 0xBL;
    private static final long   MASK = (1L << 48) - 1;
    private static final double DOUBLE_UNIT = 0x1.0p-53; // 1.0 / (1L << 53)

    @Override
    public Tensor<?> run(ExecutionCall<CPU> call) {
        call
            .getDevice()
            .getExecutor()
            .threaded(
                    call.input( 0 ).size(),
                    _newWorkloadFor( call )
            );

        return call.input( 0 );
    }


    public static <T> T fillRandomly( T data, Arg.Seed seed ) {
        return fillRandomly(data, seed.get());
    }

    public static <T> T fillRandomly( T data, String seed ) {
        return fillRandomly(data, Arg.Seed.of(seed).get());
    }

    public static <T> T fillRandomly( T data, long seed )
    {
        int size = 0;
        Class<?> type = null;
        if ( data instanceof int[]     ) { type = Integer.class;   size = ((int[]    )data).length; }
        if ( data instanceof double[]  ) { type = Double.class;    size = ((double[] )data).length; }
        if ( data instanceof float[]   ) { type = Float.class;     size = ((float[]  )data).length; }
        if ( data instanceof short[]   ) { type = Short.class;     size = ((short[]  )data).length; }
        if ( data instanceof long[]    ) { type = Long.class;      size = ((long[]   )data).length; }
        if ( data instanceof byte[]    ) { type = Byte.class;      size = ((byte[]   )data).length; }
        if ( data instanceof char[]    ) { type = Character.class; size = ((char[]   )data).length; }
        if ( data instanceof boolean[] ) { type = Boolean.class;   size = ((boolean[] )data).length; }
        if ( type == null )
            throw new IllegalArgumentException("Type '"+data.getClass()+"' not supported for randomization.");

        CPU.RangeWorkload workload = _newWorkloadFor(
                seed, type, null,
                new DataProvider() {
                    @Override
                    public <T> T get(Class<T> type) {
                        return (T) data;
                    }
                }
        );
        CPU.get().getExecutor().threaded( size, workload );
        return data;
    }

    private static CPU.RangeWorkload _newWorkloadFor( ExecutionCall<?> call ) {
        Tensor<?> tensor = call.input( 0 );
        tensor.mut().setIsVirtual(false);
        Class<?> type = tensor.getItemType();
        boolean isSimple = tensor.getNDConf().isSimple();
        NDIteratorProvider iter = i -> {
            NDIterator t0Idx = NDIterator.of(tensor);
            t0Idx.set(tensor.indicesOfIndex(i));
            return t0Idx;
        };
        long seed = call.getValOf(Arg.Seed.class);
        return _newWorkloadFor(
                seed, type, isSimple ? null : iter,
                new DataProvider() {
                    @Override
                    public <T> T get(Class<T> type) {
                        return tensor.mut().getDataForWriting(type);
                    }
                }
        );
    }

    private interface DataProvider {
        <T> T get(Class<T> type);
    }

    private interface NDIteratorProvider {
        NDIterator get(int i);
    }

    private static CPU.RangeWorkload _newWorkloadFor(
            long seed,
            Class<?> type,
            NDIteratorProvider iteratorProvider,
            DataProvider dataProvider
    ) {
        boolean isSimple = iteratorProvider == null;
        if ( type == Double.class ) {
            double[] t0_value = dataProvider.get(double[].class);
            if ( isSimple )
                return (i, end) -> {
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
                    NDIterator t0Idx = iteratorProvider.get(i);
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
        } else if ( type == Float.class ) {
            float[] t0_value = dataProvider.get(float[].class);
            if ( isSimple )
                return (i, end) -> {
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
                    NDIterator t0Idx = iteratorProvider.get(i);
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
        } else if (type == Long.class) {
            long[] t0_value = dataProvider.get(long[].class);
            if ( isSimple )
                return (i, end) -> {
                    for ( ; i < end; i ++ ) // increment on drain accordingly:
                        t0_value[i] = _nextLong(seedIndexScramble( seed, i ));
                };
            else
                return (i, end) -> {
                    NDIterator t0Idx = iteratorProvider.get(i);
                    for ( ; i < end; i ++ ) // increment on drain accordingly:
                        t0_value[t0Idx.getIndexAndIncrement()] = _nextLong(seedIndexScramble( seed, i ));
                };
        } else if (type == Integer.class) {
            int[] t0_value = dataProvider.get(int[].class);
            if ( isSimple )
                return (i, end) -> {
                    for ( ; i < end; i++ )
                        t0_value[i] = _nextInt(seedIndexScramble( seed, i ));
                };
            else
                return (i, end) -> {
                    NDIterator t0Idx = iteratorProvider.get(i);
                    for ( ; i < end; i++ )
                        t0_value[t0Idx.getIndexAndIncrement()] = _nextInt(seedIndexScramble( seed, i ));
                };
        } else if (type == Byte.class) {
            byte[] t0_value = dataProvider.get(byte[].class);
            if ( isSimple )
                return (i, end) -> {
                    for ( ; i < end; i++ )
                        t0_value[i] = _nextByte(seedIndexScramble( seed, i ));
                };
            else
                return (i, end) -> {
                    NDIterator t0Idx = iteratorProvider.get(i);
                    for ( ; i < end; i++ )
                        t0_value[t0Idx.getIndexAndIncrement()] = _nextByte(seedIndexScramble( seed, i ));
                };
        } else if (type == Short.class) {
            short[] t0_value = dataProvider.get(short[].class);
            if ( isSimple )
                return (i, end) -> {
                    for ( ; i < end; i++ )
                        t0_value[i] = _nextShort(seedIndexScramble( seed, i ));
                };
            else
                return (i, end) -> {
                    NDIterator t0Idx = iteratorProvider.get(i);
                    for ( ; i < end; i++ )
                        t0_value[t0Idx.getIndexAndIncrement()] = _nextShort(seedIndexScramble( seed, i ));
                };
        } else if (type == Boolean.class) {
            boolean[] t0_value = dataProvider.get(boolean[].class);
            if ( isSimple )
                return (i, end) -> {
                    for ( ; i < end; i++ )
                        t0_value[i] = _nextBoolean(seedIndexScramble( seed, i ));
                };
            else
                return (i, end) -> {
                    NDIterator t0Idx = iteratorProvider.get(i);
                    for ( ; i < end; i++ )
                        t0_value[t0Idx.getIndexAndIncrement()] = _nextBoolean(seedIndexScramble( seed, i ));
                };
        } else if (type == Character.class) {
            char[] t0_value = dataProvider.get(char[].class);
            if ( isSimple )
                return (i, end) -> {
                    for ( ; i < end; i++ )
                        t0_value[i] = _nextChar(seedIndexScramble( seed, i ));
                };
            else
                return (i, end) -> {
                    NDIterator t0Idx = iteratorProvider.get(i);
                    for ( ; i < end; i++ )
                        t0_value[t0Idx.getIndexAndIncrement()] = _nextChar(seedIndexScramble( seed, i ));
                };
        }
        else throw new IllegalStateException("Unsupported type: " + type);
    }

    /**
     *  A simple method which takes a long seed and an int (the current item index) and
     *  does some pseudo-random number generating (~ Linear congruential generator).
     */
    private static long seedIndexScramble( long seed, long i ) {
        i = ( i * 0x105139C0C031L + 0x4c0e1e9f367dL     ) ^ seed;
        i = ( i * 0x196E6109L     + 0x6c6f72656e64616eL ) ^ seed;
        i = ( i * 0x653L          + 0xCBC85B449DL       ) ^ seed;
        return ( i * seed ) ^ 0xa785a819cd72c6fdL;
    }

    public static long initialScramble( long seed ) { return (seed ^ MULTIPLIER) & MASK; }

    public static void gaussianFrom( long seed, double[] out )
    {
        // See Knuth, ACP, Section 3.4.1 Algorithm C.
        double v1, v2, s;
        do {
            long seed1 = _nextSeed(seed );
            long seed2 = _nextSeed(seed1);
            long seed3 = _nextSeed(seed2);
            long seed4 = _nextSeed(seed3);
            v1 = 2 * _nextDouble( seed1, seed2 ) - 1; // between -1 and 1
            v2 = 2 * _nextDouble( seed3, seed4 ) - 1; // between -1 and 1
            s = v1 * v1 + v2 * v2;
            seed = seed4;
        }
        while ( s >= 1 || s == 0 );

        double multiplier = StrictMath.sqrt( -2 * StrictMath.log(s) / s );

        out[0] = v1 * multiplier;
        out[1] = v2 * multiplier;
    }

    private static long _nextLong( long seed ) {
        long seed1 = _nextSeed(seed);
        long seed2 = _nextSeed(seed1);
        return ((long)(_next(32, seed1)) << 32) + _next(32, seed2);
    }

    private static byte _nextByte( long seed ) {
        return (byte) _nextInt(seed);
    }

    private static boolean _nextBoolean(long seed) {
        return _next(1, _nextSeed(seed)) != 0;
    }

    private static short _nextShort( long seed ) {
        return (short) _nextInt(seed);
    }

    private static long _nextSeed( long currentSeed )
    {
        long oldseed, nextseed;
        do {
            oldseed = currentSeed;
            nextseed = (oldseed * MULTIPLIER + ADDEND) & MASK;
        } while ( oldseed == (currentSeed = nextseed) );
        return nextseed;
    }

    private static double _nextDouble(long seed1, long seed2 ) {
        return (((long)(_next(26, seed1)) << 27) + _next(27, seed2)) * DOUBLE_UNIT;
    }

    private static int _nextInt( long seed ) {
        return _next(32, _nextSeed(seed));
    }

    private static int _next( int bits, long seed ) { return (int)(seed >>> (48 - bits)); }

    private static char _nextChar( long seed ) {
        return (char) _nextInt(seed);
    }

}
