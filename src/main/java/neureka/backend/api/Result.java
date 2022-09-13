package neureka.backend.api;

import neureka.Tsr;
import neureka.autograd.ADAction;
import neureka.autograd.ADAction;
import neureka.backend.api.fun.ADActionSupplier;
import neureka.backend.api.fun.Execution;
import neureka.common.utility.LogUtil;

/**
 *  An immutable wrapper for a tensor as a result of anb {@link Execution}
 *  as well as an {@link ADActionSupplier} for providing auto-differentiation support.
 */
public final class Result
{
    private final Tsr<?> _tensor;
    private final ADActionSupplier _agent;

    public static Result of(Tsr<?> tensor) { return new Result(tensor, null); }

    private Result( Tsr<?> tensor, ADActionSupplier agent ) {
        _tensor = tensor;
        _agent = agent;
    }

    public Result withADAction( ADAction action ) {
        return this.withAutoDiff( (caller, call) -> ADAction.of(action) );
    }

    public Result withAutoDiff( ADActionSupplier agent ) {
        LogUtil.nullArgCheck( agent, "agent", ADAction.class );
        if ( _agent != null )
            throw new IllegalArgumentException("Autograd algorithm already specified!");
        return new Result( _tensor, agent );
    }

    public <V> Tsr<V> get() { return (Tsr<V>) _tensor; }

    public ADActionSupplier getAgentSupplier() { return _agent; }

}
