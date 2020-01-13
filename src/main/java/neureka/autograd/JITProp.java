package neureka.autograd;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class JITProp {

    private Set<GraphNode> _finished;

    private  Map<GraphNode, PendingError> _pending;

    public JITProp(Map<GraphNode, PendingError> pendings){
        _pending = pendings;
    }

    public void addPending(Map<GraphNode, PendingError> pendings){
        if(!isDone()) throw new IllegalStateException("[JITProp][addPending]: Trying to add pending errors to JITProp which is done.");
        _pending.putAll(pendings);
    }

    public void noteFinished(GraphNode finishedJITProps){
        if(_finished==null){
            _finished = new HashSet<>();// finishedJITProps.keySet();
        }
        _finished.add(finishedJITProps);//.addAll(finishedJITProps.keySet());

    }

    public Set<GraphNode> finished(){
        return _finished;
    }

    public void execute(){
        _pending.forEach((n, p)->{
            if(_finished==null || !_finished.contains(n)){
                if(!p.isFullyAccumulated()) throw new IllegalStateException("[JITProp][execute]: Pending error has not received expected accumulation.");
                n.backwardJIT(p.getAccumulatedError(), n);//Continue backprop recursively!
            }
        });
        //_finished = null;
        _pending = null;
    }

    public boolean isDone(){
        return (_finished==null && _pending==null);
    }










}
