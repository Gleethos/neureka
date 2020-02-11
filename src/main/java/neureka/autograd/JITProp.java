package neureka.autograd;

import java.util.HashSet;
import java.util.Set;

public class JITProp
{
    private Set<GraphNode> _finished;

    private  Set<GraphNode> _pending;

    public JITProp(Set<GraphNode> pendings){//Map<GraphNode, PendingError> pendings){
        _pending = pendings;
    }

    /**
     *
     * @param pendings
     */
    public void addPending(Set<GraphNode> pendings){//Map<GraphNode, PendingError> pendings){
        if(!isDone()) throw new IllegalStateException("[JITProp][addPending]: Trying to add pending errors to JITProp which is done.");
        _pending.addAll(pendings);
    }

    /**
     *
     * @param finishedJITProps
     */
    public void noteFinished(GraphNode finishedJITProps){
        if(_finished==null){
            _finished = new HashSet<>();// finishedJITProps.keySet();
        }
        _finished.add(finishedJITProps);//.addAll(finishedJITProps.keySet());

    }

    public Set<GraphNode> finished(){
        return _finished;
    }

    /**
     *
     */
    public void execute(){
        _pending.forEach((n)->{
            if(_finished==null || !_finished.contains(n)){
                PendingError pe = n.getAndRemovePendingError();
                if(!pe.isFullyAccumulated()) throw new IllegalStateException("[JITProp][execute]: Pending error has not received expected accumulation.");
                n.backwardJIT(pe.getAccumulatedError());//Continue backprop recursively!
            }
        });
        _pending = null;
    }

    /**
     *
     * @return
     */
    public boolean isDone(){
        return (_finished==null && _pending==null);
    }










}
