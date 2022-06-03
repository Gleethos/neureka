package neureka.backend.main.algorithms.internal;

import neureka.backend.api.ExecutionCall;
import neureka.backend.main.algorithms.Functions;
import neureka.devices.host.CPU;

public interface FunImplementation<F extends Fun> {

    void get(ExecutionCall<CPU> call, Functions<F> pairs);

}
