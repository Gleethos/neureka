package neureka.backend.api.algorithms.api;

import neureka.backend.api.ExecutionCall;
import neureka.devices.Device;

public interface ForwardADChecker {

    boolean canPerformForwardADFor( ExecutionCall<? extends Device<?>> call );

}
