package neureka.acceleration.opencl.execution;

import neureka.acceleration.opencl.KernelBuilder;
import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.executors.Execution;

public class CLExecution extends AbstractCLExecution<CLExecution.CLExecutionLambda>
{
    interface CLExecutionLambda
    {
        void call ( KernelBuilder kernelBuilder );
    }

    private CLExecutionLambda _lambda;
    private int _arity;

    public CLExecution (
            CLExecutionLambda lambda,
            int arity,
            String templateName,
            String kernelSource,
            String activationSource,
            String differentiationSource,
            OperationType type
    ) {
        super(templateName, kernelSource, activationSource, differentiationSource, type);
        _lambda = lambda;
        _arity = arity;
    }

    @Override
    public CLExecutionLambda getLambda() {
        return _lambda;
    }

    @Override
    public int arity() {
        return _arity;
    }


}
