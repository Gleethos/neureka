package neureka.backend;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.devices.Device;

public interface FunImplementationFor<D extends Device<?>> {


    Tsr<?> runAndGetFirstTensor(ExecutionCall<D> call );


}
