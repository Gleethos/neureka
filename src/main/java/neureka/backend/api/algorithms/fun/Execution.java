package neureka.backend.api.algorithms.fun;

import neureka.backend.api.ExecutionCall;
import neureka.calculus.Function;
import neureka.devices.Device;

public interface Execution {

    Result execute(Function caller, ExecutionCall<? extends Device<?>> call );

}
