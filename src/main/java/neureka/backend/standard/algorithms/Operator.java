package neureka.backend.standard.algorithms;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Fun;
import neureka.backend.api.Operation;
import neureka.backend.api.algorithms.AbstractFunctionalAlgorithm;
import neureka.calculus.CalcUtil;
import neureka.calculus.RecursiveExecutor;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.dtype.NumericType;
import neureka.ndim.iterators.NDIterator;
import org.jetbrains.annotations.Contract;

import java.util.List;
import java.util.function.Supplier;

public class Operator extends AbstractFunctionalAlgorithm<Operator>
{
    public Operator( RecursiveExecutor finalExecutor ) {
        super("operator");
        setIsSuitableFor(
                call -> {
                    List<Integer> shape = ( call.getTensors()[ 0 ] == null ) ? call.getTensors()[ 1 ].shape() : call.getTensors()[ 0 ].shape();
                    int size = shape.stream().reduce(1, ( x, y ) -> x * y );
                    return call.validate()
                            .allNotNull( t -> t.size() == size && shape.equals( t.shape() ) )
                            .allNotNull( t -> t.getDataType().typeClassImplements( NumericType.class ) )
                            .basicSuitability();
                }
        );
        setCanPerformBackwardADFor( call -> true );
        setCanPerformForwardADFor( call -> true );
        setExecutionDispatcher( (caller, call) -> CalcUtil.executeFor( caller, call, finalExecutor ) );
        setCallPreparation(
                call -> {
                    Tsr<?>[] tsrs = call.getTensors();
                    Device<Double> device = (Device<Double>) call.getDevice();
                    if ( tsrs[ 0 ] == null ) // Creating a new tensor:
                    {
                        int[] shp = tsrs[ 1 ].getNDConf().shape();
                        Tsr<Double> output = Tsr.of( shp, 0.0 );
                        output.setIsVirtual( false );
                        try {
                            device.store( output );
                        } catch( Exception e ) {
                            e.printStackTrace();
                        }
                        tsrs[ 0 ] = output;
                    }
                    return call;
                }
        );
    }

    public String getKernelSource() {
        return Neureka.get().utility().readResource("kernels/operator_template.cl");
    }

    public static CPU.RangeWorkload newWorkloadFor(
            ExecutionCall<CPU> call,
            Activation.Funs<neureka.backend.api.Fun.F64F64ToF64> funF64,
            Activation.Funs<neureka.backend.api.Fun.F32F32ToF32> funF32
    ) {
        Class<?> typeClass = call.getTensors()[1].getValueClass();
        Class<?> rightTypeClass = call.getTensors()[2].getValueClass();

        int d = call.getDerivativeIndex();

        CPU.RangeWorkload workload = null;

        if ( typeClass == Double.class && rightTypeClass == Double.class )
            workload = _newWorkloadF64(  call.getTensors()[0], call.getTensors()[1], call.getTensors()[2], funF64.get(d) );

        return workload;
    }


    @Contract(pure = true)
    private static CPU.RangeWorkload _newWorkloadF64(
            Tsr<?> t0_drn, Tsr<?> t1_src, Tsr<?> t2_src,
            neureka.backend.api.Fun.F64F64ToF64 operation
    ) {

        t1_src.setIsVirtual( false );
        t2_src.setIsVirtual( false );
        double[] t1_val = t1_src.getDataAs( double[].class );
        double[] t2_val = t2_src.getDataAs( double[].class );

        if ( t0_drn.isVirtual() && t1_src.isVirtual() && t2_src.isVirtual() ) {
            return (start, end) ->
                        ( (double[]) t0_drn.getData() )[ 0 ] = operation.invoke( t1_val[0], t2_val[1] );
        } else {
            double[] t0_value = t0_drn.getDataAs(double[].class);
            return (i, end) -> {
                NDIterator t0Idx = NDIterator.of(t0_drn);
                NDIterator t1Idx = NDIterator.of(t1_src);
                NDIterator t2Idx = NDIterator.of(t2_src);
                t0Idx.set(t0_drn.IndicesOfIndex(i));
                t1Idx.set(t1_src.IndicesOfIndex(i));
                t2Idx.set(t2_src.IndicesOfIndex(i));
                while ( i < end ) {//increment on drain accordingly:
                    //setInto _value in drn:
                    t0_value[t0Idx.i()] = operation.invoke(t1_val[t1Idx.i()], t2_val[t2Idx.i()]);
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
    public static void operate(
            Tsr<?> t0_drn, Tsr<?> t1_src, Tsr<?> t2_src,
            int d, int i, int end,
            Operation.SecondaryF64NDFun operation
    ) {
        if ( t0_drn.isVirtual() && t1_src.isVirtual() && t2_src.isVirtual() ) {
            ( (double[]) t0_drn.getData() )[ 0 ] = operation.execute( NDIterator.of( t1_src ), NDIterator.of( t2_src ) );
        } else {
            //int[] t0Shp = t0_drn.getNDConf().shape(); // Tsr t0_origin, Tsr t1_handle, Tsr t2_drain ... when d>=0
            double[] t0_value = t0_drn.getDataAs( double[].class );
            NDIterator t0Idx = NDIterator.of( t0_drn );
            NDIterator t1Idx = NDIterator.of( t1_src );
            NDIterator t2Idx = NDIterator.of( t2_src );
            t0Idx.set( t0_drn.IndicesOfIndex( i ) );
            t1Idx.set( t1_src.IndicesOfIndex( i ) );
            t2Idx.set( t2_src.IndicesOfIndex( i ) );
            while ( i < end ) {//increment on drain accordingly:
                //setInto _value in drn:
                t0_value[ t0Idx.i() ] = operation.execute( t1Idx, t2Idx );
                //increment on drain:
                t0Idx.increment();
                t1Idx.increment();
                t2Idx.increment();
                i++;
            }
        }
    }

    public static class Funs<T> extends FunPair<T> {

        public Funs(T a, T d) {
            super(a, d);
        }
    }

}