package neureka.core.function.environment;

import neureka.core.Tsr;
import neureka.core.function.factory.autograd.GraphLock;
import neureka.core.function.factory.autograd.GraphNode;
import neureka.core.function.Function;

import java.util.HashMap;
import java.util.TreeMap;
import java.util.function.Supplier;

public class Cache
{
    private final HashMap<String, Function> FUNCTIONS = new HashMap<>();

    public synchronized HashMap<String, Function> FUNCTIONS(){
        return this.FUNCTIONS;
    }

    private final TreeMap<GraphLock, TreeMap<GraphNode, Tsr>> PROCESSING = new TreeMap<>((a, b)->((int)(a.hashCode()-b.hashCode())));

    public synchronized void free(GraphLock lock)//Tsr[] input
    {
        PROCESSING.remove(lock);
        lock.release();
    }

    public synchronized Tsr preprocess(Tsr[] input, Function function, Supplier<Tsr> activation)
    {
        Tsr untracked = null;
        for(Tsr t : input){
            if(t.has(GraphNode.class)){
                untracked = t;
            }
        }
        if(untracked==null){//If graph tracking (nodes) has not yet been initialized!
            return Function.setup.commit(input, function);
        }
        //GraphLock newLock = new GraphLock(function, input);
        GraphLock lock = ((GraphNode)untracked.find(GraphNode.class)).lock();
        for(Tsr t : input){
            if(t.has(GraphNode.class)){
                ((GraphNode)t.find(GraphNode.class)).optainLocking(lock);
            } else {
                GraphNode rg = new GraphNode(t, null, null, lock);
                t.add(rg);
            }
        }
        GraphNode node = (GraphNode) input[0].find(GraphNode.class);
        Tsr result = _get(node);
        if(result==null){
            result = activation.get();
            _put(result, node);
        }
        //add references/child to graph node?
        return result;
    }

    private synchronized Tsr _get(GraphNode node)
    {
        if(PROCESSING.containsKey(node.lock())){
            if(PROCESSING.get(node.lock()).containsKey(node)){
                return PROCESSING.get(node.lock()).get(node);
            }
        }
        return null;
    }

    private synchronized void _put(Tsr t, GraphNode node)
    {
        if(node.isCachable()) {
            TreeMap<GraphNode, Tsr> variables;
            if (!PROCESSING.containsKey(node.lock())) {
                variables = new TreeMap<>((a, b) -> (int) (a.nid() - b.nid()));
                PROCESSING.put(node.lock(), variables);
            } else {
                variables = PROCESSING.get(node.lock());
            }
            variables.put(node, t);
        }
    }

}
