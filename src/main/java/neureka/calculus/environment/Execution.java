package neureka.calculus.environment;

import neureka.acceleration.Device;

public interface Execution
{
    interface ExecutionLambda<TargetDevice extends Device>
    {
        void call (TargetDevice device, OperationTypeImplementation.ExecutionCall call);
    }

    ExecutionLambda getLambda();

    int arity();

}
