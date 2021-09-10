package neureka.backend.api.algorithms.fun;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.calculus.implementations.FunctionNode;
import neureka.devices.Device;

public interface Execution {

    Tsr<?> execute( FunctionNode caller, ExecutionCall<? extends Device<?>> call );

}
