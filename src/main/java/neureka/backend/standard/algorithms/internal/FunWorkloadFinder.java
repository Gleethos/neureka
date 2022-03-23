package neureka.backend.standard.algorithms.internal;

import neureka.backend.api.ExecutionCall;
import neureka.backend.standard.algorithms.Functions;
import neureka.devices.host.CPU;

@FunctionalInterface
public interface FunWorkloadFinder<F extends Fun> {

    CPU.RangeWorkload get(ExecutionCall<CPU> call, Functions<F> pairs);

}
