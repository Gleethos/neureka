package neureka.backend.standard.algorithms;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.algorithms.AbstractFunctionalAlgorithm;
import neureka.backend.standard.algorithms.internal.Fun;
import neureka.backend.standard.algorithms.internal.FunArray;
import neureka.backend.standard.algorithms.internal.WithForward;
import neureka.backend.standard.implementations.CLImplementation;
import neureka.calculus.internal.CalcUtil;
import neureka.calculus.internal.RecursiveExecutor;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.dtype.NumericType;
import neureka.ndim.NDimensional;
import neureka.ndim.iterator.NDIterator;
import org.jetbrains.annotations.Contract;

public final class Operator extends AbstractFunctionalAlgorithm<Operator>
{
    public Operator( RecursiveExecutor finalExecutor ) {
        super("operator");
        setIsSuitableFor(
            call -> call.validate()
                    .allNotNullHaveSame(NDimensional::size)
                    .allNotNullHaveSame(NDimensional::shape)
                    .allNotNull( t -> t.getDataType().typeClassImplements( NumericType.class ) )
                    .basicSuitability()
        );
        setCanPerformBackwardADFor( call -> true );
        setCanPerformForwardADFor( call -> true );
        setExecutionDispatcher( (caller, call) -> CalcUtil.executeFor( caller, call, finalExecutor ) );
        setCallPreparation(
            call -> {
                Device<Object> device = (Device<Object>) call.getDevice();
                if ( call.input( 0 ) == null ) // Creating a new tensor:
                {
                    int[] outShape = call.input( 1 ).getNDConf().shape();

                    Class<Object> type = (Class<Object>) call.input(  1 ).getValueClass();
                    Tsr<Object> output = Tsr.of( type ).withShape( outShape ).all( 0.0 ).getUnsafe().setIsIntermediate( true );
                    output.setIsVirtual( false );
                    try {
                        device.store( output );
                    } catch( Exception e ) {
                        e.printStackTrace();
                    }
                    call.setInput( 0, output );
                }
                return call;
            }
        );
    }

    public static WithForward<String> implementationForGPU(String postfix ) {
        return
            forward ->
                backward ->
                    CLImplementation
                        .compiler()
                        .arity( -1 )
                        .kernelSource( Neureka.get().utility().readResource("kernels/operator_template.cl") )
                        .activationSource( forward )
                        .differentiationSource( backward )
                        .kernelPostfix( postfix )
                        .execution(
                            call -> {
                                int offset = (call.getTsrOfType( Number.class, 0 ) != null) ? 0 : 1;
                                int gwz = (call.getTsrOfType( Number.class, 0 ) != null) ? call.getTsrOfType( Number.class, 0 ).size() : call.getTsrOfType( Number.class, 1 ).size();
                                call.getDevice()
                                    .getKernel(call)
                                    .passAllOf( call.getTsrOfType( Number.class, offset ) )
                                    .passAllOf( call.getTsrOfType( Number.class, offset + 1 ) )
                                    .passAllOf( call.getTsrOfType( Number.class, offset + 2 ) )
                                    .pass( call.getTsrOfType( Number.class, 0 ).rank() )
                                    .pass( call.getDerivativeIndex() )
                                    .call( gwz );
                            }
                        )
                        .build();
    }

    public static Functions.Builder<Fun> implementationForCPU() {
        return Functions.implementation( -1, Operator::_newWorkloadFor );
    }

    private static CPU.RangeWorkload _newWorkloadFor(
            ExecutionCall<CPU> call,
            Functions<Fun> pairs
    ) {

        FunArray<Fun.F64F64ToF64> funF64 = pairs.get(Fun.F64F64ToF64.class);
        FunArray<Fun.F32F32ToF32> funF32 = pairs.get(Fun.F32F32ToF32.class);
        FunArray<Fun.I32I32ToI32> funI32 = pairs.get(Fun.I32I32ToI32.class);
        Class<?> typeClass = call.input( 1 ).getValueClass();

        int d = call.getDerivativeIndex();

        CPU.RangeWorkload workload = null;

        if ( typeClass == Double.class )
            workload = _newWorkloadF64(  call.input( 0 ), call.input( 1 ), call.input( 2 ), funF64.get(d) );

        if ( typeClass == Float.class )
            workload = _newWorkloadF32(  call.input( 0 ), call.input( 1 ), call.input( 2 ), funF32.get(d) );

        if ( typeClass == Integer.class )
            workload = _newWorkloadI32(  call.input( 0 ), call.input( 1 ), call.input( 2 ), funI32.get(d) );

        if ( workload == null )
            throw new IllegalArgumentException("");
        else
            return workload;
    }


    @Contract(pure = true)
    private static CPU.RangeWorkload _newWorkloadF64(
            Tsr<?> t0_drn, Tsr<?> t1_src, Tsr<?> t2_src,
            Fun.F64F64ToF64 operation
    ) {
        t1_src.setIsVirtual( false );
        t2_src.setIsVirtual( false );
        double[] t0_val = (double[]) t0_drn.getUnsafe().getData();
        double[] t1_val = t1_src.getDataAs( double[].class );
        double[] t2_val = t2_src.getDataAs( double[].class );

        assert t0_val != null;
        assert t1_val != null;
        assert t2_val != null;

        boolean isSimple = t0_drn.getNDConf().isSimple() && t1_src.getNDConf().isSimple() && t2_src.getNDConf().isSimple();

        if ( t0_drn.isVirtual() && t1_src.isVirtual() && t2_src.isVirtual() ) {
            return (start, end) -> t0_val[ 0 ] = operation.invoke( t1_val[0], t2_val[1] );
        } else {
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

    @Contract(pure = true)
    private static CPU.RangeWorkload _newWorkloadF32(
            Tsr<?> t0_drn, Tsr<?> t1_src, Tsr<?> t2_src,
            Fun.F32F32ToF32 operation
    ) {
        t1_src.setIsVirtual( false );
        t2_src.setIsVirtual( false );

        float[] t0_val = (float[]) t0_drn.getUnsafe().getData();
        float[] t1_val = t1_src.getDataAs( float[].class );
        float[] t2_val = t2_src.getDataAs( float[].class );

        assert t0_val != null;
        assert t1_val != null;
        assert t2_val != null;

        boolean isSimple = t0_drn.getNDConf().isSimple() && t1_src.getNDConf().isSimple() && t2_src.getNDConf().isSimple();

        if ( t0_drn.isVirtual() && t1_src.isVirtual() && t2_src.isVirtual() ) {
            return (start, end) -> t0_val[ 0 ] = operation.invoke( t1_val[0], t2_val[1] );
        } else {
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


    @Contract(pure = true)
    private static CPU.RangeWorkload _newWorkloadI32(
            Tsr<?> t0_drn, Tsr<?> t1_src, Tsr<?> t2_src,
            Fun.I32I32ToI32 operation
    ) {
        t1_src.setIsVirtual( false );
        t2_src.setIsVirtual( false );
        int[] t0_val = (int[]) t0_drn.getUnsafe().getData();
        int[] t1_val = t1_src.getDataAs( int[].class );
        int[] t2_val = t2_src.getDataAs( int[].class );

        assert t0_val != null;
        assert t1_val != null;
        assert t2_val != null;

        boolean isSimple = t0_drn.getNDConf().isSimple() && t1_src.getNDConf().isSimple() && t2_src.getNDConf().isSimple();

        if ( t0_drn.isVirtual() && t1_src.isVirtual() && t2_src.isVirtual() ) {
            return (start, end) -> t0_val[ 0 ] = operation.invoke( t1_val[0], t2_val[1] );
        } else {
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


    public static class FunTriple<T extends Fun> implements FunArray<T> {

        private final T _a, _d1, _d2;

        public FunTriple(T a, T d, T d2 ) {
            _a  = a; _d1 = d; _d2 = d2;
        }

        @Override
        public T get(int derivativeIndex) {
            return ( derivativeIndex < 0 ? _a : ( derivativeIndex == 0 ) ? _d1 : _d2 );
        }

        @Override
        public Class<T> getType() {
            return (Class<T>) _a.getClass();
        }
    }

}