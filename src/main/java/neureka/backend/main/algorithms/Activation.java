package neureka.backend.main.algorithms;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.fun.ADAgentSupplier;
import neureka.backend.api.template.algorithms.AbstractFunDeviceAlgorithm;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.main.algorithms.internal.Fun;
import neureka.backend.main.algorithms.internal.WithForward;
import neureka.backend.main.implementations.CLImplementation;
import neureka.calculus.args.Arg;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.dtype.NumericType;
import neureka.ndim.iterator.NDIterator;

/**
 *  This is lambda based {@link neureka.backend.api.Algorithm} implementation
 *  providing some basic functionality for implementing custom
 *  activation functions.
 */
public final class Activation extends AbstractFunDeviceAlgorithm<Activation>
{
    public Activation() {
        super("activation");
        setIsSuitableFor(
           call -> call.validate()
                       .allNotNull( t -> t.getDataType().typeClassImplements(NumericType.class) )
                       .basicSuitability()
        );
        setAutogradModeFor(
            call ->
                call
                    .validate()
                    .all( ( first, second ) -> first.shape().equals(second.shape()) )
                    .ifValid(AutoDiffMode.FORWARD_AND_BACKWARD)
                    .orElse(AutoDiffMode.BACKWARD_ONLY)
        );
        setDeviceExecution( (context, callback) -> AbstractDeviceAlgorithm.executeDeviceAlgorithm(context.call(), callback), (ADAgentSupplier) null );
        setCallPreparation(
            call -> {
                Device device = call.getDeviceFor(Number.class);
                if ( call.input(  0 ) == null ) // Creating a new tensor:
                {
                    int[] shape = call.input(  1 ).getNDConf().shape();
                    Class<Object> type = (Class<Object>) call.input(  1 ).getItemType();
                    Tsr<Object> output = Tsr.of(type).withShape(shape).all( 0.0 ).getUnsafe().setIsIntermediate( true );
                    output.setIsVirtual( false );
                    device.store( output );
                    call = call.withInputAt( 0, output );
                }
                return call;
            }
        );
    }

    public String getKernelSource() {
        return Neureka.get().utility().readResource("kernels/activation_template.cl");
    }

    public static WithForward<String> implementationForGPU( String postfix ) {
        return
            forward ->
                backward ->
                    CLImplementation
                        .compiler()
                        .arity( 2 )
                        .kernelSource( Neureka.get().utility().readResource("kernels/activation_template.cl") )
                        .activationSource( forward )
                        .differentiationSource( backward )
                        .kernelPostfix( postfix )
                        .execution(
                            call -> {
                                int offset = (call.input( Number.class, 0 ) != null) ? 0 : 1;
                                int gwz = (call.input( Number.class, 0 ) != null) ? call.input( Number.class, 0 ).size() : call.input( Number.class, 1 ).size();
                                // Drain tensor needs to be 'actual'! :
                                if (!call.input( Number.class, offset + 1).isVirtual()) call.input( Number.class, offset).setIsVirtual( false );
                                call.getDevice()
                                    .getKernel(call)
                                    .passAllOf( call.input( Number.class, offset ) )
                                    .passAllOf( call.input( Number.class, offset + 1 ) )
                                    .pass( call.input( Number.class, 0 ).rank() )
                                    .pass( call.getValOf( Arg.DerivIdx.class ) )
                                    .call( gwz );

                                return call.input( 0 );
                            }
                        )
                        .build();
    }

    public static Functions.Builder<Fun> implementationForCPU() {
        return Functions.implementation( 1, Activation::_newWorkloadFor );
    }

    private static CPU.RangeWorkload _newWorkloadFor(
            ExecutionCall<CPU> call,
            Functions<Fun> funs
    ) {
        Tsr<?> t0_drn = call.input( 0 );
        Tsr<?> t1_src = call.input( 1 );
        Class<?> typeClass = t0_drn.getItemType();
        Class<?> rightTypeClass = t1_src.getItemType();

        assert !t0_drn.isVirtual();
        assert !t1_src.isVirtual();

        boolean isSimple = t0_drn.getNDConf().isSimple() && t1_src.getNDConf().isSimple();

        int d = call.getDerivativeIndex();

        CPU.RangeWorkload workload = null;

        if ( typeClass == Double.class )
        {
            Fun.F64ToF64 fun = funs.get(Fun.F64ToF64.class).get(d);
            double[] t0_value = t0_drn.getUnsafe().getDataForWriting( double[].class );

            if ( rightTypeClass == Integer.class )
            {
                int[] t1_value = (int[]) t1_src.getUnsafe().getData();
                workload = (i, end) -> {
                    NDIterator t0Idx = NDIterator.of(t0_drn);
                    NDIterator t1Idx = NDIterator.of(t1_src);
                    t0Idx.set(t0_drn.indicesOfIndex(i));
                    t1Idx.set(t0_drn.indicesOfIndex(i));
                    while (i < end) { // increment on drain accordingly:
                        //setInto _value in drn:
                        t0_value[t0Idx.i()] = fun.invoke(t1_value[t1Idx.i()]);
                        //increment on drain:
                        t0Idx.increment();
                        t1Idx.increment();
                        i++;
                    }
                };
            }
            else
            {
                double[] t1_value = t1_src.getUnsafe().getDataAs(double[].class);
                if ( isSimple )
                    workload = (start, end) -> {
                        for ( int i = start; i < end; i++ ) t0_value[i] = fun.invoke(t1_value[i]);
                    };
                else
                    workload = (i, end) -> {
                        NDIterator t0Idx = NDIterator.of(t0_drn);
                        NDIterator t1Idx = NDIterator.of(t1_src);
                        t0Idx.set(t0_drn.indicesOfIndex(i));
                        t1Idx.set(t0_drn.indicesOfIndex(i));
                        while (i < end) { // increment on drain accordingly:
                            //setInto _value in drn:
                            t0_value[t0Idx.i()] = fun.invoke(t1_value[t1Idx.i()]);
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
            Fun.F32ToF32 fun = funs.get(Fun.F32ToF32.class).get(d);
            assert fun != null;
            float[] t0_value = t0_drn.getUnsafe().getDataForWriting( float[].class );
            float[] t1_value = t1_src.getUnsafe().getDataAs(float[].class);
            if ( isSimple )
                workload = (start, end) -> {
                    for ( int i = start; i < end; i++ ) t0_value[i] = fun.invoke(t1_value[i]);
                };
            else
                workload = (i, end) -> {
                    NDIterator t0Idx = NDIterator.of( t0_drn );
                    NDIterator t1Idx = NDIterator.of( t1_src );
                    t0Idx.set( t0_drn.indicesOfIndex( i ) );
                    t1Idx.set( t0_drn.indicesOfIndex( i ) );
                    while ( i < end ) { // increment on drain accordingly:
                        //setInto _value in drn:
                        t0_value[t0Idx.i()] = fun.invoke(t1_value[t1Idx.i()]);
                        //increment on drain:
                        t0Idx.increment();
                        t1Idx.increment();
                        i++;
                    }
                };
        }
        else if ( typeClass == Integer.class )
        {
            Fun.I32ToI32 fun = funs.get(Fun.I32ToI32.class).get(d);
            assert fun != null;
            int[] t0_value = (int[]) t0_drn.getUnsafe().getData();
            int[] t1_value = t1_src.getUnsafe().getDataAs(int[].class);
            if ( isSimple )
                workload = (start, end) -> {
                    for ( int i = start; i < end; i++ ) t0_value[i] = fun.invoke(t1_value[i]);
                };
            else
                workload = (i, end) -> {
                    NDIterator t0Idx = NDIterator.of( t0_drn );
                    NDIterator t1Idx = NDIterator.of( t1_src );
                    t0Idx.set( t0_drn.indicesOfIndex( i ) );
                    t1Idx.set( t0_drn.indicesOfIndex( i ) );
                    while ( i < end ) { // increment on drain accordingly:
                        //setInto _value in drn:
                        t0_value[t0Idx.i()] = fun.invoke(t1_value[t1Idx.i()]);
                        //increment on drain:
                        t0Idx.increment();
                        t1Idx.increment();
                        i++;
                    }
                };
        }
        else if ( typeClass == Long.class )
        {
            Fun.I64ToI64 fun = funs.get(Fun.I64ToI64.class).get(d);
            assert fun != null;
            long[] t0_value = (long[]) t0_drn.getUnsafe().getData();
            long[] t1_value = t1_src.getUnsafe().getDataAs(long[].class);
            if ( isSimple )
                workload = (start, end) -> {
                    for ( int i = start; i < end; i++ ) t0_value[i] = fun.invoke(t1_value[i]);
                };
            else
                workload = (i, end) -> {
                    NDIterator t0Idx = NDIterator.of( t0_drn );
                    NDIterator t1Idx = NDIterator.of( t1_src );
                    t0Idx.set( t0_drn.indicesOfIndex( i ) );
                    t1Idx.set( t0_drn.indicesOfIndex( i ) );
                    while ( i < end ) { // increment on drain accordingly:
                        //setInto _value in drn:
                        t0_value[t0Idx.i()] = fun.invoke(t1_value[t1Idx.i()]);
                        //increment on drain:
                        t0Idx.increment();
                        t1Idx.increment();
                        i++;
                    }
                };
        }
        else if ( typeClass == Byte.class )
        {
            Fun.I8ToI8 fun = funs.get(Fun.I8ToI8.class).get(d);
            assert fun != null;
            byte[] t0_value = (byte[]) t0_drn.getUnsafe().getData();
            byte[] t1_value = t1_src.getUnsafe().getDataAs(byte[].class);
            if ( isSimple )
                workload = (start, end) -> {
                    for ( int i = start; i < end; i++ ) t0_value[i] = fun.invoke(t1_value[i]);
                };
            else
                workload = (i, end) -> {
                    NDIterator t0Idx = NDIterator.of( t0_drn );
                    NDIterator t1Idx = NDIterator.of( t1_src );
                    t0Idx.set( t0_drn.indicesOfIndex( i ) );
                    t1Idx.set( t0_drn.indicesOfIndex( i ) );
                    while ( i < end ) { // increment on drain accordingly:
                        //setInto _value in drn:
                        t0_value[t0Idx.i()] = fun.invoke(t1_value[t1Idx.i()]);
                        //increment on drain:
                        t0Idx.increment();
                        t1Idx.increment();
                        i++;
                    }
                };
        }
        else if ( typeClass == Short.class )
        {
            Fun.I16ToI16 fun = funs.get(Fun.I16ToI16.class).get(d);
            assert fun != null;
            short[] t0_value = (short[]) t0_drn.getUnsafe().getData();
            short[] t1_value = t1_src.getUnsafe().getDataAs(short[].class);
            if ( isSimple )
                workload = (start, end) -> {
                    for ( int i = start; i < end; i++ ) t0_value[i] = fun.invoke(t1_value[i]);
                };
            else
                workload = (i, end) -> {
                    NDIterator t0Idx = NDIterator.of( t0_drn );
                    NDIterator t1Idx = NDIterator.of( t1_src );
                    t0Idx.set( t0_drn.indicesOfIndex( i ) );
                    t1Idx.set( t0_drn.indicesOfIndex( i ) );
                    while ( i < end ) { // increment on drain accordingly:
                        //setInto _value in drn:
                        t0_value[t0Idx.i()] = fun.invoke(t1_value[t1Idx.i()]);
                        //increment on drain:
                        t0Idx.increment();
                        t1Idx.increment();
                        i++;
                    }
                };
        }
        else if ( typeClass == Boolean.class )
        {
            Fun.BoolToBool fun = funs.get(Fun.BoolToBool.class).get(d);
            assert fun != null;
            boolean[] t0_value = (boolean[]) t0_drn.getUnsafe().getData();
            boolean[] t1_value = t1_src.getUnsafe().getDataAs(boolean[].class);
            if ( isSimple )
                workload = (start, end) -> {
                    for ( int i = start; i < end; i++ ) t0_value[i] = fun.invoke(t1_value[i]);
                };
            else
                workload = (i, end) -> {
                    NDIterator t0Idx = NDIterator.of( t0_drn );
                    NDIterator t1Idx = NDIterator.of( t1_src );
                    t0Idx.set( t0_drn.indicesOfIndex( i ) );
                    t1Idx.set( t0_drn.indicesOfIndex( i ) );
                    while ( i < end ) { // increment on drain accordingly:
                        //setInto _value in drn:
                        t0_value[t0Idx.i()] = fun.invoke(t1_value[t1Idx.i()]);
                        //increment on drain:
                        t0Idx.increment();
                        t1Idx.increment();
                        i++;
                    }
                };
        }
        else if ( typeClass == Character.class )
        {
            Fun.CharToChar fun = funs.get(Fun.CharToChar.class).get(d);
            assert fun != null;
            char[] t0_value = (char[]) t0_drn.getUnsafe().getData();
            char[] t1_value = t1_src.getUnsafe().getDataAs(char[].class);
            if ( isSimple )
                workload = (start, end) -> {
                    for ( int i = start; i < end; i++ ) t0_value[i] = fun.invoke(t1_value[i]);
                };
            else
                workload = (i, end) -> {
                    NDIterator t0Idx = NDIterator.of( t0_drn );
                    NDIterator t1Idx = NDIterator.of( t1_src );
                    t0Idx.set( t0_drn.indicesOfIndex( i ) );
                    t1Idx.set( t0_drn.indicesOfIndex( i ) );
                    while ( i < end ) { // increment on drain accordingly:
                        //setInto _value in drn:
                        t0_value[t0Idx.i()] = fun.invoke(t1_value[t1Idx.i()]);
                        //increment on drain:
                        t0Idx.increment();
                        t1Idx.increment();
                        i++;
                    }
                };
        } else {
            try {
                Fun.ObjToObj fun = funs.get(Fun.ObjToObj.class).get(d);
                assert fun != null;
                Object[] t0_value = (Object[]) t0_drn.getUnsafe().getData();
                Object[] t1_value = t1_src.getUnsafe().getDataAs(Object[].class);
                if (isSimple)
                    workload = (start, end) -> {
                        for (int i = start; i < end; i++) t0_value[i] = fun.invoke(t1_value[i]);
                    };
                else
                    workload = (i, end) -> {
                        NDIterator t0Idx = NDIterator.of(t0_drn);
                        NDIterator t1Idx = NDIterator.of(t1_src);
                        t0Idx.set(t0_drn.indicesOfIndex(i));
                        t1Idx.set(t0_drn.indicesOfIndex(i));
                        while (i < end) { // increment on drain accordingly:
                            //setInto _value in drn:
                            t0_value[t0Idx.i()] = fun.invoke(t1_value[t1Idx.i()]);
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
