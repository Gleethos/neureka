package neureka.devices.host.execution;

import neureka.devices.host.HostCPU;
import neureka.calculus.backend.executions.ExecutorFor;

/**
 * This class is the ExecutorFor &lt; HostCPU &gt; implementation
 * used to properly call a HostCPU instance via the
 * ExecutionOn &lt; HostCPU &gt; lambda implementation
 * receiving an instance of the ExecutionCall class.
 */
public class HostExecutor implements ExecutorFor< HostCPU >
{
    private final ExecutionOn<HostCPU> _creator;
    private final int _arity;

    public HostExecutor(ExecutionOn<HostCPU> creator, int arity)
    {
        _creator = creator;
        _arity = arity;
    }

    @Override
    public ExecutionOn<HostCPU> getExecution()
    {
        return _creator;
    }

    @Override
    public int arity() {
        return _arity;
    }
}
