package neureka.calculus.environment;

import neureka.acceleration.Device;

public interface ExecutorFor<TargetDevice extends Device>
{
    interface ExecutionOn<TargetDevice extends Device>
    {
        void call (ExecutionCall<TargetDevice> call);
    }

    ExecutionOn<TargetDevice> getExecution();

    int arity();

}
