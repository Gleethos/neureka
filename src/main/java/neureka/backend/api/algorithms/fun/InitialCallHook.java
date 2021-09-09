package neureka.backend.api.algorithms.fun;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.calculus.implementations.FunctionNode;
import neureka.devices.Device;

public interface InitialCallHook {

    Tsr<?> handle(FunctionNode caller, ExecutionCall<? extends Device<?>> call );

}
