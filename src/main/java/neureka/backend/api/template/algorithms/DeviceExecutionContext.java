package neureka.backend.api.template.algorithms;

import neureka.backend.api.ExecutionCall;
import neureka.calculus.Function;

public class DeviceExecutionContext
{
    private final ExecutionCall<?> call;

    public DeviceExecutionContext( ExecutionCall<?> call ) {
        this.call = call;
    }

    public ExecutionCall<?> call() {
        return call;
    }

}
