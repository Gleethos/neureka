package neureka.backend.standard.algorithms.internal;

import neureka.backend.api.ExecutionCall;
import neureka.backend.standard.algorithms.Functions;
import neureka.devices.host.CPU;

public interface FunImplementation<F extends Fun> {

    void get(ExecutionCall<CPU> call, Functions<F> pairs);

}
