package neureka.backend.api.template.algorithms;

import neureka.backend.api.ExecutionCall;
import neureka.calculus.Function;

public class DeviceExecutionContext
{
    private final ExecutionCall<?> initialCall;
    private final ExecutionCall<?> call;
    private final Function caller;

    public DeviceExecutionContext( ExecutionCall<?> initialCall, ExecutionCall<?> call, Function caller ) {
        this.initialCall = initialCall;
        this.call = call;
        this.caller = caller;
    }

    public ExecutionCall<?> initialCall() {
        return initialCall;
    }

    public ExecutionCall<?> call() {
        return call;
    }

    public Function caller() {
        return caller;
    }

}
