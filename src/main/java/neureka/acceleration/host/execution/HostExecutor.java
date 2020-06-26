package neureka.acceleration.host.execution;

import neureka.acceleration.host.HostCPU;
import neureka.calculus.environment.ExecutorFor;

public class HostExecutor implements ExecutorFor<HostCPU>
{
    private ExecutionOn<HostCPU> _creator;
    private int _arity;

    public HostExecutor(ExecutionOn<HostCPU> creator, int arity){
        _creator = creator;
        _arity = arity;
    }

    @Override
    public ExecutionOn<HostCPU> getExecution() {
        return _creator;
    }

    @Override
    public int arity() {
        return _arity;
    }
}
