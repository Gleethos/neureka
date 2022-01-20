package neureka.backend.standard.algorithms;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.algorithms.AbstractFunctionalAlgorithm;
import neureka.calculus.internal.CalcUtil;
import neureka.calculus.internal.RecursiveExecutor;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.dtype.NumericType;
import neureka.ndim.iterators.NDIterator;
import org.jetbrains.annotations.Contract;

import java.util.List;

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
                Tsr<?>[] inputs = call.getTensors();
                Device<Object> device = (Device<Object>) call.getDevice();
                if ( inputs[ 0 ] == null ) // Creating a new tensor:
                {
                    int[] outShape = inputs[ 1 ].getNDConf().shape();

                    Class<Object> type = (Class<Object>) inputs[ 1 ].getValueClass();
                    Tsr<Object> output = Tsr.of( type ).withShape( outShape ).all( 0.0 ).getUnsafe().setIsIntermediate( true );
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
        return Neureka.get().utility().readResource("kernels/operator_template.cl");
    }

    public static Functions.Builder<Fun> implementationForCPU() {
        return Functions.implementation(
            (call, pairs) ->
                call.getDevice()
                    .getExecutor()
                    .threaded(
                        call.getTsrOfType( Number.class, 0 ).size(),
                        _newWorkloadFor( call, pairs )
                    )
        );
    }


    private static CPU.RangeWorkload _newWorkloadFor(
            ExecutionCall<CPU> call,
            Functions<Fun> pairs
    ) {

        FunArray<Fun.F64F64ToF64> funF64 = pairs.get(Fun.F64F64ToF64.class);
        FunArray<Fun.F32F32ToF32> funF32 = pairs.get(Fun.F32F32ToF32.class);
        Class<?> typeClass = call.getTensors()[1].getValueClass();
        Class<?> rightTypeClass = call.getTensors()[2].getValueClass();

        int d = call.getDerivativeIndex();

        CPU.RangeWorkload workload = null;

        if ( typeClass == Double.class )
            workload = _newWorkloadF64(  call.getTensors()[0], call.getTensors()[1], call.getTensors()[2], funF64.get(d) );

        if ( typeClass == Float.class )
            workload = _newWorkloadF32(  call.getTensors()[0], call.getTensors()[1], call.getTensors()[2], funF32.get(d) );

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
        double[] t1_val = t1_src.getDataAs( double[].class );
        double[] t2_val = t2_src.getDataAs( double[].class );

        assert t1_val != null;
        assert t2_val != null;

        if ( t0_drn.isVirtual() && t1_src.isVirtual() && t2_src.isVirtual() ) {
            return (start, end) ->
                        ( (double[]) t0_drn.getData() )[ 0 ] = operation.invoke( t1_val[0], t2_val[1] );
        } else {
            double[] t0_value = t0_drn.getDataAs(double[].class);
            return (i, end) -> {
                NDIterator t0Idx = NDIterator.of(t0_drn);
                NDIterator t1Idx = NDIterator.of(t1_src);
                NDIterator t2Idx = NDIterator.of(t2_src);
                t0Idx.set(t0_drn.indicesOfIndex(i));
                t1Idx.set(t1_src.indicesOfIndex(i));
                t2Idx.set(t2_src.indicesOfIndex(i));
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
    private static CPU.RangeWorkload _newWorkloadF32(
            Tsr<?> t0_drn, Tsr<?> t1_src, Tsr<?> t2_src,
            Fun.F32F32ToF32 operation
    ) {
        t1_src.setIsVirtual( false );
        t2_src.setIsVirtual( false );
        float[] t1_val = t1_src.getDataAs( float[].class );
        float[] t2_val = t2_src.getDataAs( float[].class );

        assert t1_val != null;
        assert t2_val != null;

        if ( t0_drn.isVirtual() && t1_src.isVirtual() && t2_src.isVirtual() ) {
            return (start, end) ->
                    ( (double[]) t0_drn.getData() )[ 0 ] = operation.invoke( t1_val[0], t2_val[1] );
        } else {
            float[] t0_value = t0_drn.getDataAs(float[].class);
            return (i, end) -> {
                NDIterator t0Idx = NDIterator.of(t0_drn);
                NDIterator t1Idx = NDIterator.of(t1_src);
                NDIterator t2Idx = NDIterator.of(t2_src);
                t0Idx.set(t0_drn.indicesOfIndex(i));
                t1Idx.set(t1_src.indicesOfIndex(i));
                t2Idx.set(t2_src.indicesOfIndex(i));
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