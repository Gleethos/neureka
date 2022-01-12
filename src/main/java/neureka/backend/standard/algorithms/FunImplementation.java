package neureka.backend.standard.algorithms;

import neureka.backend.api.ExecutionCall;
import neureka.devices.host.CPU;

public interface FunImplementation<F extends Fun> {

    void get(ExecutionCall<CPU> call, Functions<F> pairs);

}
