package neureka.calculus.internal;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.devices.Device;

/**
 *  Used in a recursion as final execution call by implementations of the {@link RecursiveExecutor}
 *  interface within the {@link CalcUtil}
 *  in order to execute (and break down) chained operators pairwise!
 */
public interface CallExecutor {

    Tsr<?> execute( ExecutionCall<? extends Device<?>> call );

}
