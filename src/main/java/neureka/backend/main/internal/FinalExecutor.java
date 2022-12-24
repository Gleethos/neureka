package neureka.backend.main.internal;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.devices.Device;

public interface FinalExecutor {

    Tsr<?> execute( ExecutionCall<? extends Device<?>> call );

}
