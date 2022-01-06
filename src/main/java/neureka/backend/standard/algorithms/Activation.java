package neureka.backend.standard.algorithms;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.PrimitiveFun;
import neureka.backend.api.Operation;
import neureka.backend.api.algorithms.AbstractFunctionalAlgorithm;
import neureka.calculus.CalcUtil;
import neureka.calculus.args.Arg;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.dtype.NumericType;
import neureka.ndim.iterators.NDIterator;
import org.jetbrains.annotations.Contract;

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
                            Tsr<?>[] tensors = call.getTensors();
                            Device device = call.getDeviceFor(Number.class);
                            if ( tensors[ 0 ] == null ) // Creating a new tensor:
                            {
                                int[] shp = tensors[ 1 ].getNDConf().shape();
                                Class<Object> type = (Class<Object>) tensors[ 1 ].getValueClass();
                                Tsr<Object> output = Tsr.of(type).withShape(shp).all( 0.0 );
                                output.setIsVirtual( false );
                                try {
                                    device.store( output );
                                } catch( Exception e ) {
                                    e.printStackTrace();
                                }
                                tensors[ 0 ] = output;
                            }
                            return call;
                        }
        );
    }

    public String getKernelSource() {
        return Neureka.get().utility().readResource("kernels/activation_template.cl");
    }

    public static CPU.RangeWorkload newWorkloadFor(
            ExecutionCall<CPU> call,
            Fun<PrimitiveFun.PrimaryF64> funF64,
            Fun<PrimitiveFun.PrimaryF32> funF32
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
            PrimitiveFun.PrimaryF64 fun = funF64.get(d);
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
            PrimitiveFun.PrimaryF32 fun = funF32.get(d);
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

    public static class Fun<T> {

        private final T _a;
        private final T _d;

        public Fun(T a, T d) {
            _a = a;
            _d = d;
        }

        public T get(int d) { return ( d < 0 ? _a : _d ); }

    }

}
