package neureka.backend.standard.algorithms;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.algorithms.AbstractFunctionalAlgorithm;
import neureka.backend.api.algorithms.fun.SuitabilityPredicate;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.dtype.NumericType;
import neureka.ndim.iterators.NDIterator;
import org.jetbrains.annotations.Contract;

public class Scalarization extends AbstractFunctionalAlgorithm< Scalarization >
{
    public Scalarization() {
        super("scalarization");
        setCanPerformBackwardADFor( call -> true );
        setCanPerformForwardADFor( call -> true );
        setIsSuitableFor( call ->
            call.validate()
                .allNotNull( t -> t.getDataType().typeClassImplements(NumericType.class) )
                //.first( Objects::isNull )
                .tensors( tensors ->  {
                    if ( tensors.length != 2 && tensors.length != 3 ) return false;
                    int offset = ( tensors.length == 2 ? 0 : 1 );
                    if ( tensors[1+offset].size() > 1 && !tensors[1+offset].isVirtual() ) return false;
                    return
                        //tensors[1+offset].shape().stream().allMatch( d -> d == 1 )
                        //||
                        tensors[offset].shape().equals(tensors[1+offset].shape());
                })
                .suitabilityIfValid( SuitabilityPredicate.VERY_GOOD )
        );
        setCallPreparation(
            call -> {
                Tsr<?>[] inputs = call.getTensors();
                Device<Number> device = call.getDeviceFor(Number.class);
                assert inputs[ 0 ] == null;  // Creating a new tensor:

                int[] outShape = inputs[ 1 ].getNDConf().shape();
                Class<Object> type = (Class<Object>) inputs[ 1 ].getValueClass();
                Tsr output = Tsr.of( type, outShape, 0.0 ).getUnsafe().setIsIntermediate( true );
                output.setIsVirtual( false );
                try {
                    device.store( output );
                } catch( Exception e ) {
                    e.printStackTrace();
                }
                inputs[ 0 ] = output;

                return call;
            }
        );
    }


    public String getKernelSource() {
        return Neureka.get().utility().readResource("kernels/scalarization_template.cl");
    }


    public static Functions.Builder<Fun> implementationForCPU() {
        return Functions.implementation(
                3,
                (call, pairs) ->
                    call.getDevice()
                        .getExecutor()
                        .threaded(
                            call.getTsrOfType( Number.class, 0 ).size(),
                            _workloadFor( call, pairs )
                        )
        );
    }
    @Contract(pure = true)
    private static CPU.RangeWorkload _workloadFor(
        ExecutionCall<CPU> call,
        Functions<Fun> functions
    ) {
        int offset = ( call.getTensors().length == 3 ? 1 : 0 );
        Tsr<?> t0_drn = call.getTensors()[0];
        Tsr<?> src    = call.getTensors()[offset];

        Class<?> typeClass = call.getTensors()[1].getValueClass();

        CPU.RangeWorkload workload = null;

        if ( typeClass == Double.class ) {
            double value = call.getTsrOfType(Number.class, 1 + offset).getDataAs(double[].class)[0];
            Fun.F64F64ToF64 operation = functions.get(Fun.F64F64ToF64.class).get(call.getDerivativeIndex());
            double[] t0_value = t0_drn.getDataAs(double[].class);
            double[] t1_value = src.getDataAs(double[].class);
            workload = ( i, end ) -> {
                NDIterator t0Idx = NDIterator.of(t0_drn);
                NDIterator srcIdx = NDIterator.of(src);
                t0Idx.set(t0_drn.indicesOfIndex(i));
                srcIdx.set(src.indicesOfIndex(i));
                while (i < end) // increment on drain accordingly:
                {
                    // setInto _value in drn:
                    t0_value[t0Idx.i()] = operation.invoke(t1_value[srcIdx.i()], value);
                    // increment on drain:
                    t0Idx.increment();
                    srcIdx.increment();
                    //NDConfiguration.Utility.increment(t0Idx, t0Shp);
                    i++;
                }
            };
        }
        if ( typeClass == Float.class ) {
            float value = call.getTsrOfType(Number.class, 1 + offset).getDataAs(float[].class)[0];
            Fun.F32F32ToF32 operation = functions.get(Fun.F32F32ToF32.class).get(call.getDerivativeIndex());
            float[] t0_value = t0_drn.getDataAs(float[].class);
            float[] t1_value = src.getDataAs(float[].class);
            workload = ( i, end ) -> {
                NDIterator t0Idx = NDIterator.of(t0_drn);
                NDIterator srcIdx = NDIterator.of(src);
                t0Idx.set(t0_drn.indicesOfIndex(i));
                srcIdx.set(src.indicesOfIndex(i));
                while (i < end) // increment on drain accordingly:
                {
                    // setInto _value in drn:
                    t0_value[t0Idx.i()] = operation.invoke(t1_value[srcIdx.i()], value);
                    // increment on drain:
                    t0Idx.increment();
                    srcIdx.increment();
                    //NDConfiguration.Utility.increment(t0Idx, t0Shp);
                    i++;
                }
            };
        }
        if ( typeClass == Integer.class ) {
            int value = call.getTsrOfType(Number.class, 1 + offset).getDataAs(int[].class)[0];
            Fun.I32I32ToI32 operation = functions.get(Fun.I32I32ToI32.class).get(call.getDerivativeIndex());
            int[] t0_value = t0_drn.getDataAs(int[].class);
            int[] t1_value = src.getDataAs(int[].class);
            workload = ( i, end ) -> {
                NDIterator t0Idx = NDIterator.of(t0_drn);
                NDIterator srcIdx = NDIterator.of(src);
                t0Idx.set(t0_drn.indicesOfIndex(i));
                srcIdx.set(src.indicesOfIndex(i));
                while (i < end) // increment on drain accordingly:
                {
                    // setInto _value in drn:
                    t0_value[t0Idx.i()] = operation.invoke(t1_value[srcIdx.i()], value);
                    // increment on drain:
                    t0Idx.increment();
                    srcIdx.increment();
                    //NDConfiguration.Utility.increment(t0Idx, t0Shp);
                    i++;
                }
            };
        }

        if ( workload == null )
            throw new IllegalArgumentException("");
        else
            return workload;
    }

}