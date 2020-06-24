package neureka.calculus.environment;

import neureka.acceleration.Device;

public interface Execution<TargetDevice extends Device>
{
    interface ExecutionLambda<TargetDevice extends Device>
    {
        void call (OperationTypeImplementation.ExecutionCall<TargetDevice> call);
    }

    ExecutionLambda<TargetDevice> getLambda();

    int arity();

}
