package neureka.acceleration.opencl;

import neureka.calculus.environment.executors.Execution;

public class CLExecution implements Execution<CLExecution.CLExecutionLambda>
{

    interface CLExecutionLambda {
        void call(KernelBuilder kernelBuilder);
    }

    @Override
    public CLExecutionLambda getLambda() {
        return null;
    }


}
