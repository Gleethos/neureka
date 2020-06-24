package neureka.acceleration.host.execution;

import neureka.acceleration.host.HostCPU;
import neureka.calculus.environment.Execution;

public class HostExecution implements Execution
{
    private ExecutionLambda<HostCPU> _creator;
    private int _arity;

    public HostExecution(ExecutionLambda<HostCPU> creator, int arity){
        _creator = creator;
        _arity = arity;
    }

    @Override
    public ExecutionLambda<HostCPU> getLambda() {
        return _creator;
    }

    @Override
    public int arity() {
        return _arity;
    }
}
