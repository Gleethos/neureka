package neureka.backend.api.algorithms.api;

import neureka.backend.api.ExecutionCall;
import neureka.devices.Device;

public interface BackwardADChecker {

    boolean canPerformBackwardADFor( ExecutionCall<? extends Device<?>> call );

}
