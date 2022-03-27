package neureka.backend.standard.algorithms;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.algorithms.AbstractFunctionalAlgorithm;
import neureka.backend.api.algorithms.fun.SuitabilityPredicate;
import neureka.backend.standard.algorithms.internal.Fun;
import neureka.backend.standard.algorithms.internal.FunArray;
import neureka.calculus.args.Arg;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;
import neureka.dtype.NumericType;
import neureka.ndim.iterator.NDIterator;
import org.jetbrains.annotations.Contract;

public class ScalarBroadcast extends AbstractFunctionalAlgorithm<ScalarBroadcast>
{
    public ScalarBroadcast(FunArray<Fun.F64ToF64> funs) {
        super("scalar broadcast");
        setCanPerformBackwardADFor( call -> true );
        setCanPerformForwardADFor( call -> true );
        setIsSuitableFor( call ->
                call.validate()
                        .allNotNull( t -> t.getDataType().typeClassImplements(NumericType.class) )
                        //.first( Objects::isNull )
                        .tensors( tensors ->  {
                            if ( tensors.length != 2 ) return false;
                            if ( !tensors[1].isVirtual() ) return false;
                            if ( tensors[0] != null && tensors[0].isVirtual() ) return false;
                            return tensors[0] == null && tensors[1] != null || tensors[0].shape().equals(tensors[1].shape());
                        })
                        .suitabilityIfValid( SuitabilityPredicate.VERY_GOOD )
        );
        setCallPreparation(
                call -> {
                    Device<Number> device = call.getDeviceFor(Number.class);
                    assert call.input( 0 ) == null;  // Creating a new tensor:
                    int[] outShape = call.input( 1 ).getNDConf().shape();
                    Class<Object> type = (Class<Object>) call.input( 1 ).getValueClass();
                    Tsr output = Tsr.of( type, outShape, 0.0 ).getUnsafe().setIsIntermediate( true );
                    output.setIsVirtual( false );
                    try {
                        device.store( output );
                    } catch( Exception e ) {
                        e.printStackTrace();
                    }
                    call.setInput( 0, output );
                    return call;
                }
        );
        setImplementationFor(
            OpenCLDevice.class,
            call -> {
                double value =  funs.get( call.getValOf( Arg.DerivIdx.class ) ).invoke( call.input( Number.class, 1 ).at(0).get().doubleValue() );
                Tsr<Number> t = call.input( Number.class, 0 );
                int gwz = t.size();
                call.getDevice()
                        .getKernel("scalar_broadcast")
                        .passAllOf(t)
                        .pass((float) value)
                        .pass(t.rank())
                        .call( gwz );
            }
        );
    }


    public static String getKernelSource() {
        return Neureka.get().utility().readResource("kernels/scalar_broadcast.cl");
    }


    public static Functions.Builder<Fun> implementationForCPU() {
        return Functions.implementation( 2, ScalarBroadcast::_workloadFor );
    }

    @Contract(pure = true)
    private static CPU.RangeWorkload _workloadFor(
            ExecutionCall<CPU> call,
            Functions<Fun> functions
    ) {
        Tsr<Number> t0_drn = call.input( Number.class, 0 );
        Tsr<Number> src    = call.input( Number.class, 1 );

        Class<?> typeClass = t0_drn.getValueClass();

        CPU.RangeWorkload workload = null;

        if ( typeClass == Double.class ) {
            double value = src.at(0).get().doubleValue();
            Fun.F64ToF64 operation = functions.get(Fun.F64ToF64.class).get(call.getDerivativeIndex());
            double[] t0_value = t0_drn.getUnsafe().getDataAs(double[].class);
            double finalValue = operation.invoke(value);
            workload = ( i, end ) -> {
                NDIterator t0Idx = NDIterator.of(t0_drn);
                NDIterator srcIdx = NDIterator.of(src);
                t0Idx.set(t0_drn.indicesOfIndex(i));
                srcIdx.set(src.indicesOfIndex(i));
                while ( i < end ) // increment on drain accordingly:
                {
                    // setInto _value in drn:
                    t0_value[t0Idx.i()] = finalValue;
                    // increment on drain:
                    t0Idx.increment();
                    srcIdx.increment();
                    //NDConfiguration.Utility.increment(t0Idx, t0Shp);
                    i++;
                }
            };
        }
        if ( typeClass == Float.class ) {
            float value = src.at(0).get().floatValue();
            Fun.F32ToF32 operation = functions.get(Fun.F32ToF32.class).get(call.getDerivativeIndex());
            float[] t0_value = t0_drn.getUnsafe().getDataAs(float[].class);
            float finalValue = operation.invoke(value);
            workload = ( i, end ) -> {
                NDIterator t0Idx = NDIterator.of(t0_drn);
                NDIterator srcIdx = NDIterator.of(src);
                t0Idx.set(t0_drn.indicesOfIndex(i));
                srcIdx.set(src.indicesOfIndex(i));
                while (i < end) // increment on drain accordingly:
                {
                    // setInto _value in drn:
                    t0_value[t0Idx.i()] = finalValue;
                    // increment on drain:
                    t0Idx.increment();
                    srcIdx.increment();
                    //NDConfiguration.Utility.increment(t0Idx, t0Shp);
                    i++;
                }
            };
        }
        if ( typeClass == Integer.class ) {
            int value = src.at(0).get().intValue();
            Fun.I32ToI32 operation = functions.get(Fun.I32ToI32.class).get(call.getDerivativeIndex());
            int[] t0_value = t0_drn.getUnsafe().getDataAs(int[].class);
            int finalValue = operation.invoke(value);
            workload = ( i, end ) -> {
                NDIterator t0Idx = NDIterator.of(t0_drn);
                NDIterator srcIdx = NDIterator.of(src);
                t0Idx.set(t0_drn.indicesOfIndex(i));
                srcIdx.set(src.indicesOfIndex(i));
                while (i < end) // increment on drain accordingly:
                {
                    // setInto _value in drn:
                    t0_value[t0Idx.i()] = finalValue;
                    // increment on drain:
                    t0Idx.increment();
                    srcIdx.increment();
                    //NDConfiguration.Utility.increment(t0Idx, t0Shp);
                    i++;
                }
            };
        }
        if ( t0_drn.getUnsafe().getData().getClass() == Object[].class ) {
            Object value = src.at(0).get();
            Fun.ObjToObj operation = functions.get(Fun.ObjToObj.class).get(call.getDerivativeIndex());
            Object[] t0_value = t0_drn.getUnsafe().getDataAs(Object[].class);
            Object finalValue = operation.invoke(value);
            workload = (i, end ) -> {
                NDIterator t0Idx = NDIterator.of(t0_drn);
                NDIterator srcIdx = NDIterator.of(src);
                t0Idx.set(t0_drn.indicesOfIndex(i));
                srcIdx.set(src.indicesOfIndex(i));
                while (i < end) // increment on drain accordingly:
                {
                    // setInto _value in drn:
                    t0_value[t0Idx.i()] = finalValue;
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