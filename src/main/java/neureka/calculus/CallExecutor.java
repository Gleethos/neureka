package neureka.calculus;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.devices.Device;

public interface CallExecutor {

    Tsr<?> execute( ExecutionCall<? extends Device<?>> call );

}
