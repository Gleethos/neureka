package neureka.backend.standard.algorithms;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Fun;
import neureka.backend.api.algorithms.AbstractFunctionalAlgorithm;
import neureka.calculus.CalcUtil;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.dtype.NumericType;
import neureka.ndim.iterators.NDIterator;

/**
 *  This is lambda based {@link neureka.backend.api.Algorithm} implementation
 *  providing some basic functionality for implementing custom
 *  activation functions.
 */
public class Activation extends AbstractFunctionalAlgorithm<Activation>
{

    public Activation() {
        super("activation");
        setIsSuitableFor(
                call -> call.validate()
                            .allNotNull( t -> t.getDataType().typeClassImplements(NumericType.class) )
                            .basicSuitability()
        );
        setCanPerformBackwardADFor( call -> true );
        setCanPerformForwardADFor(
                        call -> call
                                .validate()
                                .all( ( first, second ) -> first.shape().equals(second.shape()) )
                                .isValid()
                );
        setExecutionDispatcher( CalcUtil::defaultRecursiveExecution);
        setCallPreparation(
                        call -> {
                            Tsr<?>[] inputs = call.getTensors();
                            Device device = call.getDeviceFor(Number.class);
                            if ( inputs[ 0 ] == null ) // Creating a new tensor:
                            {
                                int[] shape = inputs[ 1 ].getNDConf().shape();
                                Class<Object> type = (Class<Object>) inputs[ 1 ].getValueClass();
                                Tsr<Object> output = Tsr.of(type).withShape(shape).all( 0.0 );
                                output.setIsVirtual( false );
                                try {
                                    device.store( output );
                                } catch( Exception e ) {
                                    e.printStackTrace();
                                }
                                inputs[ 0 ] = output;
                            }
                            return call;
                        }
        );
    }

    public String getKernelSource() {
        return Neureka.get().utility().readResource("kernels/activation_template.cl");
    }


    public static Functions.Builder<Fun> implementationForCPU() {
        return Functions.implementation(
                            (call, pairs) ->
                                call.getDevice()
                                    .getExecutor()
                                    .threaded(
                                            call.getTsrOfType( Number.class, 0 ).size(),
                                            _newWorkloadFor(call, pairs)
                                    )
                        );
    }

    private static CPU.RangeWorkload _newWorkloadFor(
            ExecutionCall<CPU> call,
            Functions<Fun> funs
    ) {
        Class<?> typeClass = call.getTensors()[0].getValueClass();
        Class<?> rightTypeClass = call.getTensors()[1].getValueClass();

        Tsr<?> t0_drn = call.getTensors()[0];
        Tsr<?> t1_src = call.getTensors()[1];
        boolean noSlices = !t0_drn.getNDConf().isSlice() && !t1_src.getNDConf().isSlice();

        int d = call.getDerivativeIndex();

        CPU.RangeWorkload workload = null;

        if ( typeClass == Double.class )
        {
            Fun.F64ToF64 fun = funs.get(Fun.F64ToF64.class).get(d);
            double[] t0_value = (double[]) t0_drn.getData();

            if ( rightTypeClass == Integer.class )
            {
                int[] t1_value = (int[]) t1_src.getData();
                workload = (i, end) -> {
                    NDIterator t0Idx = NDIterator.of(t0_drn);
                    NDIterator t1Idx = NDIterator.of(t1_src);
                    t0Idx.set(t0_drn.IndicesOfIndex(i));
                    t1Idx.set(t0_drn.IndicesOfIndex(i));
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
                double[] t1_value = t1_src.getDataAs(double[].class);
                if ( noSlices )
                    workload = (start, end) -> {
                        for ( int i = start; i < end; i++ ) t0_value[i] = fun.invoke(t1_value[i]);
                    };
                else
                    workload = (i, end) -> {
                        NDIterator t0Idx = NDIterator.of(t0_drn);
                        NDIterator t1Idx = NDIterator.of(t1_src);
                        t0Idx.set(t0_drn.IndicesOfIndex(i));
                        t1Idx.set(t0_drn.IndicesOfIndex(i));
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
            float[] t0_value = (float[]) t0_drn.getData();
            float[] t1_value = (float[]) t1_src.getData();
            if ( noSlices )
                workload = (start, end) -> {
                    for ( int i = start; i < end; i++ ) t0_value[i] = fun.invoke(t1_value[i]);
                };
            else
                workload = (i, end) -> {
                    NDIterator t0Idx = NDIterator.of( t0_drn );
                    NDIterator t1Idx = NDIterator.of( t1_src );
                    t0Idx.set( t0_drn.IndicesOfIndex( i ) );
                    t1Idx.set( t0_drn.IndicesOfIndex( i ) );
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

        if ( workload == null ) {
            throw new IllegalArgumentException(
                "Operand types '"+typeClass.getSimpleName()+"' and '"+rightTypeClass.getSimpleName()+"'."
            );
        }

        return workload;
    }

}
