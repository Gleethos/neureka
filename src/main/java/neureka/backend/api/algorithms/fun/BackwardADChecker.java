package neureka.backend.api.algorithms.fun;

import neureka.backend.api.ExecutionCall;
import neureka.devices.Device;

public interface BackwardADChecker {

    boolean canPerformBackwardADFor( ExecutionCall<? extends Device<?>> call );

}
