package neureka.backend.main.operations.function.internal;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.ImplementationFor;
import neureka.backend.main.algorithms.Functions;
import neureka.backend.main.algorithms.internal.Fun;
import neureka.devices.host.CPU;
import neureka.ndim.iterator.NDIterator;

public class CPUScalarBroadcastActivation implements ImplementationFor<CPU>
{
    private final ImplementationFor<CPU> _impl;


    public CPUScalarBroadcastActivation(ActivationFun fun) {
        _impl = Functions.implementation( 2, CPUScalarBroadcastActivation::_workloadFor )
                .with(Fun.F64ToF64.pair(fun::activate, fun::derive))
                .with(Fun.F32ToF32.pair(fun::activate, fun::derive))
                .with(Fun.I32ToI32.pair(fun::activate, fun::derive))
                .with(Fun.I64ToI64.pair(fun::activate, fun::derive))
                .with(Fun.I8ToI8.pair(fun::activate, fun::derive))
                .with(Fun.I16ToI16.pair(fun::activate, fun::derive))
                .with(Fun.BoolToBool.pair(fun::activate, fun::derive))
                .with(Fun.CharToChar.pair(fun::activate, fun::derive))
                .get();
    }



    @Override
    public Tsr<?> run(ExecutionCall<CPU> call) {
        return _impl.run(call);
    }

    private static CPU.RangeWorkload _workloadFor(
            ExecutionCall<CPU> call,
            Functions<Fun> functions
    ) {
        Tsr<Number> t0_drn = call.input( Number.class, 0 );
        Tsr<Number> src    = call.input( Number.class, 1 );

        Class<?> typeClass = t0_drn.getItemType();

        CPU.RangeWorkload workload = null;

        if ( typeClass == Double.class ) {
            double value = src.at(0).get().doubleValue();
            Fun.F64ToF64 operation = functions.get(Fun.F64ToF64.class).getFor( call );
            double[] t0_value = t0_drn.getUnsafe().getDataForWriting(double[].class);
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
            Fun.F32ToF32 operation = functions.get(Fun.F32ToF32.class).getFor( call );
            float[] t0_value = t0_drn.getUnsafe().getDataForWriting(float[].class);
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
            Fun.I32ToI32 operation = functions.get(Fun.I32ToI32.class).getFor( call );
            int[] t0_value = t0_drn.getUnsafe().getDataForWriting(int[].class);
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
        if ( t0_drn.getUnsafe().getData().getRef().getClass() == Object[].class ) {
            Object value = src.at(0).get();
            Fun.ObjToObj operation = functions.get(Fun.ObjToObj.class).getFor( call );
            Object[] t0_value = t0_drn.getUnsafe().getDataForWriting(Object[].class);
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
