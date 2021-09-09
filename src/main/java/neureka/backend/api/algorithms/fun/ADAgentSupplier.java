package neureka.backend.api.algorithms.fun;

import neureka.autograd.ADAgent;
import neureka.backend.api.ExecutionCall;
import neureka.calculus.Function;
import neureka.devices.Device;

public interface ADAgentSupplier {

    ADAgent supplyADAgentFor(
            Function f,
            ExecutionCall<? extends Device<?>> call,
            boolean forward
    );

}
