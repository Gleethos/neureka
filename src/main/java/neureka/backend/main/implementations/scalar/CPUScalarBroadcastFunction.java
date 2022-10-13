package neureka.backend.main.implementations.scalar;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.ImplementationFor;
import neureka.backend.main.implementations.fun.api.CPUFun;
import neureka.backend.main.implementations.fun.api.ScalarFun;
import neureka.calculus.args.Arg;
import neureka.devices.host.CPU;
import neureka.ndim.iterator.NDIterator;

public class CPUScalarBroadcastFunction implements ImplementationFor<CPU>
{
    private final ScalarFun _fun;

    public CPUScalarBroadcastFunction(ScalarFun fun ) { _fun = fun; }

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
        Tsr<Number> t0_drn = call.input( Number.class, 0 );
        Tsr<Number> src    = call.input( Number.class, 1 );

        Class<?> typeClass = t0_drn.getItemType();

        CPU.RangeWorkload workload = null;

        int d = call.getValOf(Arg.DerivIdx.class);
        CPUFun f = d < 0 ? _fun.getActivation() : _fun.getDerivative();

        if ( typeClass == Double.class ) {
            double value = src.at(0).get().doubleValue();
            double[] t0_value = t0_drn.mut().getDataForWriting(double[].class);
            double finalValue = f.invoke(value);
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
            float[] t0_value = t0_drn.mut().getDataForWriting(float[].class);
            float finalValue = f.invoke(value);
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
            int[] t0_value = t0_drn.mut().getDataForWriting(int[].class);
            int finalValue = f.invoke(value);
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
        if ( t0_drn.mut().getData().getRef().getClass() == Object[].class ) {
            Object value = src.at(0).get();
            Object[] t0_value = t0_drn.mut().getDataForWriting(Object[].class);
            Object finalValue = f.invoke(value);
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
