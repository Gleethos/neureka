package neureka.backend.main.implementations.elementwise;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.ImplementationFor;
import neureka.backend.main.implementations.fun.api.CPUBiFun;
import neureka.devices.host.CPU;
import neureka.ndim.iterator.NDIterator;

public abstract class CPUBiElementWise implements ImplementationFor<CPU>
{
    protected abstract CPUBiFun _getFun();
    protected abstract CPUBiFun _getDeriveAt0();
    protected abstract CPUBiFun _getDeriveAt1();

    @Override
    public Tsr<?> run(ExecutionCall<CPU> call) {
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
        Class<?> typeClass = call.input( 1 ).getItemType();

        int d = call.getDerivativeIndex();
        CPUBiFun f = ( d == 0 ? _getDeriveAt0() : ( d == 1 ? _getDeriveAt1() : _getFun() ) );

        CPU.RangeWorkload workload = null;

        if ( typeClass == Double.class )
            workload = _newWorkloadF64(  call.input( 0 ), call.input( 1 ), call.input( 2 ), f );

        if ( typeClass == Float.class )
            workload = _newWorkloadF32(  call.input( 0 ), call.input( 1 ), call.input( 2 ), f );

        if ( typeClass == Integer.class )
            workload = _newWorkloadI32(  call.input( 0 ), call.input( 1 ), call.input( 2 ), f );

        if ( workload == null )
            throw new IllegalArgumentException("");
        else
            return workload;
    }

    private static CPU.RangeWorkload _newWorkloadF64(
            Tsr<?> t0_drn, Tsr<?> t1_src, Tsr<?> t2_src,
            CPUBiFun operation
    ) {
        t1_src.setIsVirtual( false );
        t2_src.setIsVirtual( false );
        double[] t0_val = t0_drn.getUnsafe().getDataForWriting( double[].class );
        double[] t1_val = t1_src.getUnsafe().getDataAs( double[].class );
        double[] t2_val = t2_src.getUnsafe().getDataAs( double[].class );

        assert t0_val != null;
        assert t1_val != null;
        assert t2_val != null;

        boolean isSimple = t0_drn.getNDConf().isSimple() && t1_src.getNDConf().isSimple() && t2_src.getNDConf().isSimple();

        if ( t0_drn.isVirtual() && t1_src.isVirtual() && t2_src.isVirtual() )
            return (start, end) -> t0_val[ 0 ] = operation.invoke( t1_val[0], t2_val[1] );
        else {
            if ( isSimple )
                return (start, end) -> {
                    for ( int i = start; i < end; i++ ) t0_val[i] = operation.invoke(t1_val[i], t2_val[i]);
                };
            else
                return (i, end) -> {
                    NDIterator t0Idx = NDIterator.of(t0_drn);
                    NDIterator t1Idx = NDIterator.of(t1_src);
                    NDIterator t2Idx = NDIterator.of(t2_src);
                    t0Idx.set(t0_drn.indicesOfIndex(i));
                    t1Idx.set(t1_src.indicesOfIndex(i));
                    t2Idx.set(t2_src.indicesOfIndex(i));
                    while ( i < end ) {//increment on drain accordingly:
                        //setInto _value in drn:
                        t0_val[t0Idx.i()] = operation.invoke(t1_val[t1Idx.i()], t2_val[t2Idx.i()]);
                        //increment on drain:
                        t0Idx.increment();
                        t1Idx.increment();
                        t2Idx.increment();
                        i++;
                    }
                };
        }
    }

    private static CPU.RangeWorkload _newWorkloadF32(
            Tsr<?> t0_drn, Tsr<?> t1_src, Tsr<?> t2_src,
            CPUBiFun operation
    ) {
        t1_src.setIsVirtual( false );
        t2_src.setIsVirtual( false );

        float[] t0_val = t0_drn.getUnsafe().getDataForWriting( float[].class );
        float[] t1_val = t1_src.getUnsafe().getDataAs( float[].class );
        float[] t2_val = t2_src.getUnsafe().getDataAs( float[].class );

        assert t0_val != null;
        assert t1_val != null;
        assert t2_val != null;

        boolean isSimple = t0_drn.getNDConf().isSimple() && t1_src.getNDConf().isSimple() && t2_src.getNDConf().isSimple();

        if ( t0_drn.isVirtual() && t1_src.isVirtual() && t2_src.isVirtual() )
            return (start, end) -> t0_val[ 0 ] = operation.invoke( t1_val[0], t2_val[1] );
        else {
            if ( isSimple )
                return  (start, end) -> {
                    for ( int i = start; i < end; i++ ) t0_val[i] = operation.invoke(t1_val[i], t2_val[i]);
                };
            else
                return (i, end) -> {
                    NDIterator t0Idx = NDIterator.of(t0_drn);
                    NDIterator t1Idx = NDIterator.of(t1_src);
                    NDIterator t2Idx = NDIterator.of(t2_src);
                    t0Idx.set(t0_drn.indicesOfIndex(i));
                    t1Idx.set(t1_src.indicesOfIndex(i));
                    t2Idx.set(t2_src.indicesOfIndex(i));
                    while ( i < end ) {//increment on drain accordingly:
                        //setInto _value in drn:
                        t0_val[t0Idx.i()] = operation.invoke(t1_val[t1Idx.i()], t2_val[t2Idx.i()]);
                        //increment on drain:
                        t0Idx.increment();
                        t1Idx.increment();
                        t2Idx.increment();
                        i++;
                    }
                };
        }
    }

    private static CPU.RangeWorkload _newWorkloadI32(
            Tsr<?> t0_drn, Tsr<?> t1_src, Tsr<?> t2_src,
            CPUBiFun operation
    ) {
        t1_src.setIsVirtual( false );
        t2_src.setIsVirtual( false );
        int[] t0_val = (int[]) t0_drn.getUnsafe().getData().getRef();
        int[] t1_val = t1_src.getUnsafe().getDataAs( int[].class );
        int[] t2_val = t2_src.getUnsafe().getDataAs( int[].class );

        assert t0_val != null;
        assert t1_val != null;
        assert t2_val != null;

        boolean isSimple = t0_drn.getNDConf().isSimple() && t1_src.getNDConf().isSimple() && t2_src.getNDConf().isSimple();

        if ( t0_drn.isVirtual() && t1_src.isVirtual() && t2_src.isVirtual() )
            return (start, end) -> t0_val[ 0 ] = operation.invoke( t1_val[0], t2_val[1] );
        else {
            if ( isSimple )
                return  (start, end) -> {
                    for ( int i = start; i < end; i++ ) t0_val[i] = operation.invoke(t1_val[i], t2_val[i]);
                };
            else
                return (i, end) -> {
                    NDIterator t0Idx = NDIterator.of(t0_drn);
                    NDIterator t1Idx = NDIterator.of(t1_src);
                    NDIterator t2Idx = NDIterator.of(t2_src);
                    t0Idx.set(t0_drn.indicesOfIndex(i));
                    t1Idx.set(t1_src.indicesOfIndex(i));
                    t2Idx.set(t2_src.indicesOfIndex(i));
                    while ( i < end ) {//increment on drain accordingly:
                        //setInto _value in drn:
                        t0_val[t0Idx.i()] = operation.invoke(t1_val[t1Idx.i()], t2_val[t2Idx.i()]);
                        //increment on drain:
                        t0Idx.increment();
                        t1Idx.increment();
                        t2Idx.increment();
                        i++;
                    }
                };
        }
    }

}
