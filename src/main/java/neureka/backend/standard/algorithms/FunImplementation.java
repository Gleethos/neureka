package neureka.backend.standard.algorithms;

import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Fun;
import neureka.devices.host.CPU;

public interface FunImplementation<F extends Fun> {

    void get(ExecutionCall<CPU> call, Functions<F> pairs);

}
