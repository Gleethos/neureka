package neureka.backend.main.implementations.elementwise;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.ImplementationFor;
import neureka.backend.main.implementations.fun.api.CPUFun;
import neureka.backend.main.implementations.fun.api.ScalarFun;
import neureka.math.args.Arg;
import neureka.devices.host.CPU;
import neureka.ndim.iterator.NDIterator;

public class CPUElementwiseFunction implements ImplementationFor<CPU>
{
    private final ScalarFun _fun;

    public CPUElementwiseFunction( ScalarFun fun ) { _fun = fun; }

    @Override
    public Tsr<?> run( ExecutionCall<CPU> call ) {
        call.getDevice()
            .getExecutor()
            .threaded(
                call.input(0).size(),
                _workloadFor(call)
            );

        return call.input(0);
    }

    private CPU.RangeWorkload _workloadFor(
            ExecutionCall<CPU> call
    ) {
        Tsr<?> t0_drn = call.input( 0 );
        Tsr<?> t1_src = call.input( 1 );
        Class<?> typeClass = t0_drn.getItemType();
        Class<?> rightTypeClass = t1_src.getItemType();

        int d = call.getValOf(Arg.DerivIdx.class);
        CPUFun f = d < 0 ? _fun.getActivation() : _fun.getDerivative();

        assert !t0_drn.isVirtual();
        assert !t1_src.isVirtual();

        boolean isSimple = t0_drn.getNDConf().isSimple() && t1_src.getNDConf().isSimple();

        CPU.RangeWorkload workload = null;

        if ( typeClass == Double.class )
        {
            double[] t0_value = t0_drn.mut().getDataForWriting( double[].class );

            if ( rightTypeClass == Integer.class )
            {
                int[] t1_value = (int[]) t1_src.mut().getData().getRef();
                workload = (i, end) -> {
                    NDIterator t0Idx = NDIterator.of(t0_drn);
                    NDIterator t1Idx = NDIterator.of(t1_src);
                    t0Idx.set(t0_drn.indicesOfIndex(i));
                    t1Idx.set(t0_drn.indicesOfIndex(i));
                    while (i < end) { // increment on drain accordingly:
                        //setInto _value in drn:
                        t0_value[t0Idx.i()] = f.invoke(t1_value[t1Idx.i()]);
                        //increment on drain:
                        t0Idx.increment();
                        t1Idx.increment();
                        i++;
                    }
                };
            }
            else
            {
                double[] t1_value = t1_src.mut().getDataAs(double[].class);
                if ( isSimple )
                    workload = (start, end) -> {
                        for ( int i = start; i < end; i++ ) t0_value[i] = f.invoke(t1_value[i]);
                    };
                else
                    workload = (i, end) -> {
                        NDIterator t0Idx = NDIterator.of(t0_drn);
                        NDIterator t1Idx = NDIterator.of(t1_src);
                        t0Idx.set(t0_drn.indicesOfIndex(i));
                        t1Idx.set(t0_drn.indicesOfIndex(i));
                        while (i < end) { // increment on drain accordingly:
                            //setInto _value in drn:
                            t0_value[t0Idx.i()] = f.invoke(t1_value[t1Idx.i()]);
                            //increment on drain:
                            t0Idx.increment();
                            t1Idx.increment();
                            i++;
                        }
                    };
            }
        }
        else if ( typeClass == Float.class )
        {
            float[] t0_value = t0_drn.mut().getDataForWriting( float[].class );
            float[] t1_value = t1_src.mut().getDataAs(float[].class);
            if ( isSimple )
                workload = (start, end) -> {
                    for ( int i = start; i < end; i++ ) t0_value[i] = f.invoke(t1_value[i]);
                };
            else
                workload = (i, end) -> {
                    NDIterator t0Idx = NDIterator.of( t0_drn );
                    NDIterator t1Idx = NDIterator.of( t1_src );
                    t0Idx.set( t0_drn.indicesOfIndex( i ) );
                    t1Idx.set( t0_drn.indicesOfIndex( i ) );
                    while ( i < end ) { // increment on drain accordingly:
                        //setInto _value in drn:
                        t0_value[t0Idx.i()] = f.invoke(t1_value[t1Idx.i()]);
                        //increment on drain:
                        t0Idx.increment();
                        t1Idx.increment();
                        i++;
                    }
                };
        }
        else if ( typeClass == Integer.class )
        {
            int[] t0_value = (int[]) t0_drn.mut().getData().getRef();
            int[] t1_value = t1_src.mut().getDataAs(int[].class);
            if ( isSimple )
                workload = (start, end) -> {
                    for ( int i = start; i < end; i++ ) t0_value[i] = f.invoke(t1_value[i]);
                };
            else
                workload = (i, end) -> {
                    NDIterator t0Idx = NDIterator.of( t0_drn );
                    NDIterator t1Idx = NDIterator.of( t1_src );
                    t0Idx.set( t0_drn.indicesOfIndex( i ) );
                    t1Idx.set( t0_drn.indicesOfIndex( i ) );
                    while ( i < end ) { // increment on drain accordingly:
                        //setInto _value in drn:
                        t0_value[t0Idx.i()] = f.invoke(t1_value[t1Idx.i()]);
                        //increment on drain:
                        t0Idx.increment();
                        t1Idx.increment();
                        i++;
                    }
                };
        }
        else if ( typeClass == Long.class )
        {
            long[] t0_value = (long[]) t0_drn.mut().getData().getRef();
            long[] t1_value = t1_src.mut().getDataAs(long[].class);
            if ( isSimple )
                workload = (start, end) -> {
                    for ( int i = start; i < end; i++ ) t0_value[i] = f.invoke(t1_value[i]);
                };
            else
                workload = (i, end) -> {
                    NDIterator t0Idx = NDIterator.of( t0_drn );
                    NDIterator t1Idx = NDIterator.of( t1_src );
                    t0Idx.set( t0_drn.indicesOfIndex( i ) );
                    t1Idx.set( t0_drn.indicesOfIndex( i ) );
                    while ( i < end ) { // increment on drain accordingly:
                        //setInto _value in drn:
                        t0_value[t0Idx.i()] = f.invoke(t1_value[t1Idx.i()]);
                        //increment on drain:
                        t0Idx.increment();
                        t1Idx.increment();
                        i++;
                    }
                };
        }
        else if ( typeClass == Byte.class )
        {
            byte[] t0_value = (byte[]) t0_drn.mut().getData().getRef();
            byte[] t1_value = t1_src.mut().getDataAs(byte[].class);
            if ( isSimple )
                workload = (start, end) -> {
                    for ( int i = start; i < end; i++ ) t0_value[i] = f.invoke(t1_value[i]);
                };
            else
                workload = (i, end) -> {
                    NDIterator t0Idx = NDIterator.of( t0_drn );
                    NDIterator t1Idx = NDIterator.of( t1_src );
                    t0Idx.set( t0_drn.indicesOfIndex( i ) );
                    t1Idx.set( t0_drn.indicesOfIndex( i ) );
                    while ( i < end ) { // increment on drain accordingly:
                        //setInto _value in drn:
                        t0_value[t0Idx.i()] = f.invoke(t1_value[t1Idx.i()]);
                        //increment on drain:
                        t0Idx.increment();
                        t1Idx.increment();
                        i++;
                    }
                };
        }
        else if ( typeClass == Short.class )
        {
            short[] t0_value = (short[]) t0_drn.mut().getData().getRef();
            short[] t1_value = t1_src.mut().getDataAs(short[].class);
            if ( isSimple )
                workload = (start, end) -> {
                    for ( int i = start; i < end; i++ ) t0_value[i] = f.invoke(t1_value[i]);
                };
            else
                workload = (i, end) -> {
                    NDIterator t0Idx = NDIterator.of( t0_drn );
                    NDIterator t1Idx = NDIterator.of( t1_src );
                    t0Idx.set( t0_drn.indicesOfIndex( i ) );
                    t1Idx.set( t0_drn.indicesOfIndex( i ) );
                    while ( i < end ) { // increment on drain accordingly:
                        //setInto _value in drn:
                        t0_value[t0Idx.i()] = f.invoke(t1_value[t1Idx.i()]);
                        //increment on drain:
                        t0Idx.increment();
                        t1Idx.increment();
                        i++;
                    }
                };
        }
        else if ( typeClass == Boolean.class )
        {
            boolean[] t0_value = (boolean[]) t0_drn.mut().getData().getRef();
            boolean[] t1_value = t1_src.mut().getDataAs(boolean[].class);
            if ( isSimple )
                workload = (start, end) -> {
                    for ( int i = start; i < end; i++ ) t0_value[i] = f.invoke(t1_value[i]);
                };
            else
                workload = (i, end) -> {
                    NDIterator t0Idx = NDIterator.of( t0_drn );
                    NDIterator t1Idx = NDIterator.of( t1_src );
                    t0Idx.set( t0_drn.indicesOfIndex( i ) );
                    t1Idx.set( t0_drn.indicesOfIndex( i ) );
                    while ( i < end ) { // increment on drain accordingly:
                        //setInto _value in drn:
                        t0_value[t0Idx.i()] = f.invoke(t1_value[t1Idx.i()]);
                        //increment on drain:
                        t0Idx.increment();
                        t1Idx.increment();
                        i++;
                    }
                };
        }
        else if ( typeClass == Character.class )
        {
            char[] t0_value = (char[]) t0_drn.mut().getData().getRef();
            char[] t1_value = t1_src.mut().getDataAs(char[].class);
            if ( isSimple )
                workload = (start, end) -> {
                    for ( int i = start; i < end; i++ ) t0_value[i] = f.invoke(t1_value[i]);
                };
            else
                workload = (i, end) -> {
                    NDIterator t0Idx = NDIterator.of( t0_drn );
                    NDIterator t1Idx = NDIterator.of( t1_src );
                    t0Idx.set( t0_drn.indicesOfIndex( i ) );
                    t1Idx.set( t0_drn.indicesOfIndex( i ) );
                    while ( i < end ) { // increment on drain accordingly:
                        //setInto _value in drn:
                        t0_value[t0Idx.i()] = f.invoke(t1_value[t1Idx.i()]);
                        //increment on drain:
                        t0Idx.increment();
                        t1Idx.increment();
                        i++;
                    }
                };
        } else {
            try {
                Object[] t0_value = (Object[]) t0_drn.mut().getData().getRef();
                Object[] t1_value = t1_src.mut().getDataAs(Object[].class);
                if (isSimple)
                    workload = (start, end) -> {
                        for (int i = start; i < end; i++) t0_value[i] = f.invoke(t1_value[i]);
                    };
                else
                    workload = (i, end) -> {
                        NDIterator t0Idx = NDIterator.of(t0_drn);
                        NDIterator t1Idx = NDIterator.of(t1_src);
                        t0Idx.set(t0_drn.indicesOfIndex(i));
                        t1Idx.set(t0_drn.indicesOfIndex(i));
                        while (i < end) { // increment on drain accordingly:
                            //setInto _value in drn:
                            t0_value[t0Idx.i()] = f.invoke(t1_value[t1Idx.i()]);
                            //increment on drain:
                            t0Idx.increment();
                            t1Idx.increment();
                            i++;
                        }
                    };
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        if ( workload == null )
            throw new IllegalArgumentException(
                    "Operand types '"+typeClass.getSimpleName()+"' and '"+rightTypeClass.getSimpleName()+"' not supported."
            );

        return workload;
    }

}
