package neureka.backend.api.algorithms.fun;

import neureka.Tsr;
import neureka.autograd.ADAction;
import neureka.autograd.ADAgent;
import neureka.autograd.DefaultADAgent;
import neureka.common.utility.LogUtil;

/**
 *  An immutable wrapper for a tensor as a result of anb {@link Execution}
 *  as well as an {@link ADAgentSupplier} for providing auto-differentiation support.
 */
public class Result
{
    private final Tsr<?> _tensor;
    private final ADAgentSupplier _agent;

    public static Result of(Tsr<?> tensor) { return new Result(tensor, null); }

    private Result( Tsr<?> tensor, ADAgentSupplier agent ) {
        _tensor = tensor;
        _agent = agent;
    }

    public Result withADAction( ADAction action ) {
        return this.withAutoDiff( (caller, call, forward) -> ADAgent.withAD(action) );
    }

    public Result withADAgent( ADAgent agent ) {
        return this.withAutoDiff( (caller, call, forward) -> agent );
    }

    public Result withAutoDiff( ADAgentSupplier agent ) {
        LogUtil.nullArgCheck( agent, "agent", ADAgent.class );
        if ( _agent != null )
            throw new IllegalArgumentException("Autograd algorithm already specified!");
        return new Result( _tensor, agent );
    }

    public <V> Tsr<V> get() { return (Tsr<V>) _tensor; }

    public ADAgentSupplier getAgentSupplier() { return _agent; }

}
