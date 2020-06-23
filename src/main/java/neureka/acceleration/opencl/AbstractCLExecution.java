package neureka.acceleration.opencl;

import neureka.calculus.environment.executors.Execution;
import neureka.calculus.environment.executors.TypeExecutor;

public class AbstractCLExecution implements Execution
{



    @Override
    public boolean canExecute(TypeExecutor.ExecutionCall call) {
        return false;
    }

    @Override
    public void execute(TypeExecutor.ExecutionCall call) {

    }
}
