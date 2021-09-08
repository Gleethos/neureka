package neureka.backend.api.algorithms.api;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.devices.Device;

/**
 *  Orchestrates execution recursively according to arity.
 */
public interface RecursiveExecutor {

    Tsr<?> execute( ExecutionCall<? extends Device<?>> call, CallExecutor goDeeperWith );

}
