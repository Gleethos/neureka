package neureka.autograd;

import neureka.Component;
import neureka.Tsr;

import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;

public class JITProp<ValueType> implements Component<Tsr<ValueType>>
{
    private Set<GraphNode<ValueType>> _finished;

    private  Set<GraphNode<ValueType>> _pending;

    public JITProp(Set<GraphNode<ValueType>> pendings){
        _pending = new HashSet<>();
        _pending.addAll(pendings); // Every JITProp component has their own Set.
        //... otherwise this would lead to finished JIT-Propagations where in fact traversals are still pending...
    }

    /**
     *
     * @param pendings A set of GraphNode<ValueType> instance which are saved for future backprop continuation.
     */
    public void addPending(Set<GraphNode<ValueType>> pendings){
        if(pendings.isEmpty()) throw new IllegalStateException("Trying to add empty pending errors set to JITProp.");
        if(!isDone()) throw new IllegalStateException("Trying to add pending errors to JITProp which is done.");
        _pending.addAll(pendings);
    }

    /**
     *
     * @param finishedJITProps The reference to a GraphNote which has finished (JITed) backpropation.
     */
    public void noteFinished(GraphNode<ValueType> finishedJITProps){
        if(_finished==null) _finished = new HashSet<>();
        _finished.add(finishedJITProps);
        if (_pending!=null) {
            Set<GraphNode<ValueType>> intersection = _finished.stream().filter(_pending::contains).collect(Collectors.toSet());
            _finished.removeAll(intersection);
            _pending.removeAll(intersection);
            if(_finished.isEmpty()) _finished = null;
            if(_pending.isEmpty()) _pending = null;
        }
    }

    public int finishedCount(){
        return (_finished==null)?0:_finished.size();
    }

    public int pendingCount(){
        return (_pending==null)?0:_pending.size();
    }


    /**
     *
     */
    public void execute(){
        if(_pending==null) return;
        _pending.forEach( n -> {
            if(_finished==null || !_finished.contains(n)){
                PendingError pe = n.getAndRemovePendingError();
                if(!pe.isFullyAccumulated()) throw new IllegalStateException("Pending error has not received expected accumulation.");
                n.backwardJIT(pe.getAccumulatedError());//Continue backprop recursively!
            }
        });
        if (pendingCount()>0) throw new IllegalStateException("Pending error has not received expected accumulation.");
        _pending = null;
    }

    /**
     *
     * @return
     */
    public boolean isDone(){
        return (_finished==null && _pending==null);
    }




    @Override
    public String toString() {
        int finished = (_finished==null)?0:_finished.size();
        int pending = (_pending==null)?0:_pending.size();
        return "(JIT@"+hashCode()+"):{finished:"+finished+", pending:"+pending+", isDone:"+isDone()+"}";
    }


    @Override
    public void update(Tsr<ValueType> oldOwner, Tsr<ValueType> newOwner) {

    }
}
