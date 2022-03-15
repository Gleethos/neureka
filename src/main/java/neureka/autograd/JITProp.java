package neureka.autograd;

import neureka.common.composition.Component;
import neureka.Tsr;

import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;

/**
 *  This class keeps track of graph nodes which require
 *  back-propagation in order
 *  to be able to continue the process at a later point in time
 *  (based on some configurable conditions).
 *
 * @param <V> The type parameter of the involved tensors.
 */
public final class JITProp<V> implements Component<Tsr<V>>
{
    private Set<GraphNode<V>> _finished;

    private  Set<GraphNode<V>> _pending;

    public JITProp( Set<GraphNode<V>> pending ) {
        _pending = new HashSet<>();
        _pending.addAll( pending ); // Every JITProp component has their own Set.
        //... otherwise this would lead to finished JIT-Propagations where in fact traversals are still pending...
    }

    /**
     * @param pending A set of GraphNode&lt;V&gt; instance which are saved for future backprop continuation.
     */
    public void addPending( Set<GraphNode<V>> pending ) {
        if ( pending.isEmpty() )
            throw new IllegalStateException("Trying to add empty pending errors set to JITProp.");
        if ( !isDone() )
            throw new IllegalStateException("Trying to add pending errors to JITProp which is done.");
        _pending.addAll( pending );
    }

    /**
     *
     * @param finishedJITProps The reference to a GraphNote which has finished (JITed) backpropation.
     */
    public void noteFinished( GraphNode<V> finishedJITProps ) {
        if ( _finished == null ) _finished = new HashSet<>();
        _finished.add( finishedJITProps );
        if ( _pending != null ) {
            Set<GraphNode<V>> intersection = _finished.stream().filter(_pending::contains).collect(Collectors.toSet());
            _finished.removeAll( intersection );
            _pending.removeAll( intersection );
            if ( _finished.isEmpty() ) _finished = null;
            if ( _pending.isEmpty() ) _pending = null;
        }
    }

    public int finishedCount() { return _finished == null ? 0 : _finished.size(); }

    public int pendingCount() { return _pending == null ? 0 : _pending.size(); }


    /**
     *  This method triggers the continuation of the back-propagation which
     *  has been put on hold by saving the pending graph nodes inside this class. <br>
     *  The execution request happens when gradients are immediately required by a tensor,
     *  which is the case when the tensor is about to apply its gradients. <br>
     *  However because the gradient has not yet been fully calculated this method
     *  will be called first (assuming the tensor has a JITProp component stored).
     */
    public void execute() {
        if ( _pending == null ) return;
        _pending.forEach( n -> {
            if ( _finished == null || !_finished.contains( n ) ) {
                PendingError<V> pe = n.getAndRemovePendingError();
                if ( !pe.isFullyAccumulated() )
                    throw new IllegalStateException("Pending error has not received expected accumulation.");
                n.backwardJIT( pe.getAccumulatedError() ); // Continue back-prop recursively!
            }
        });
        if ( pendingCount() > 0 )
            throw new IllegalStateException("Pending error has not received expected accumulation.");
        _pending = null;
    }

    /**
     * @return The truth value determining if the back-propagation has been completed.
     */
    public boolean isDone() { return ( _finished == null && _pending == null ); }




    @Override
    public String toString() {
        int finished = ( _finished == null ? 0 : _finished.size() );
        int pending  = ( _pending  == null ? 0 : _pending.size()  );
        return this.getClass().getSimpleName()+"@"+hashCode()+"[finished="+finished+",pending="+pending+",isDone="+isDone()+"]";
    }


    @Override
    public boolean update( OwnerChangeRequest<Tsr<V>> changeRequest ) {
        changeRequest.executeChange();
        return true;
    }
}
