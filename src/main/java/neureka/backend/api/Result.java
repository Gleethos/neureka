package neureka.backend.api;

import neureka.Tensor;
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
    private final Tensor<?> _tensor;
    private final ADActionSupplier _agent;

    public static Result of( Tensor<?> tensor ) {
        LogUtil.nullArgCheck( tensor, "tensor", Tensor.class, "An operation may not return 'null'!" );
        return new Result(tensor, null);
    }

    private Result(Tensor<?> tensor, ADActionSupplier agent ) {
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

    public <V> Tensor<V> get() { return (Tensor<V>) _tensor; }

    public ADActionSupplier getAgentSupplier() { return _agent; }

}
