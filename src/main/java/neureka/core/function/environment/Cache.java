package neureka.core.function.environment;

import neureka.core.T;
import neureka.core.function.factory.autograd.GraphLock;
import neureka.core.function.factory.autograd.GraphNode;
import neureka.core.function.IFunction;

import java.util.HashMap;
import java.util.TreeMap;
import java.util.function.Supplier;

public class Cache
{
    private final HashMap<String, IFunction> FUNCTIONS = new HashMap<>();

    public synchronized HashMap<String, IFunction> FUNCTIONS(){
        return this.FUNCTIONS;
    }

    private final TreeMap<GraphLock, TreeMap<GraphNode, T>> TENSORS = new TreeMap<>((a, b)->((int)(a.hashCode()-b.hashCode())));

    public synchronized void free(T[] input)
    {
        for(T t : input){
            TENSORS.remove(((GraphNode)t.find(GraphNode.class)).lock());
            t.remove(GraphNode.class);
        }
    }

    public synchronized T handle(T[] input, IFunction function, Supplier<T> activation)
    {
        T untracked = null;
        for(T t : input){
            if(t.has(GraphNode.class)){
                untracked = t;
            }
        }
        if(untracked==null){//If graph tracking (nodes) has not yet been initialized!
            return IFunction.setup.commit(new T(), input, function);
        }
        for(T t : input){
            if(!t.has(GraphNode.class)){
                GraphLock gid = ((GraphNode)untracked.find(GraphNode.class)).lock();
                GraphNode rg = new GraphNode(t, null, null, gid);
                t.add(rg);
            }
        }
        GraphNode node = (GraphNode) input[0].find(GraphNode.class);
        T result = get(node);
        if(result==null){
            result = activation.get();
            put(result, node);
        }
        return result;
    }

    private synchronized T get(GraphNode node)
    {
        if(TENSORS.containsKey(node.lock())){
            if(TENSORS.get(node.lock()).containsKey(node)){
                return TENSORS.get(node.lock()).get(node);
            }
        }
        return null;
    }

    private synchronized void put(T t, GraphNode node)
    {
        if(node.isCachable()) {
            TreeMap<GraphNode, T> variables;
            if (!TENSORS.containsKey(node.lock())) {
                variables = new TreeMap<>((a, b) -> (int) (a.nid() - b.nid()));
                TENSORS.put(node.lock(), variables);
            } else {
                variables = TENSORS.get(node.lock());
            }
            variables.put(node, t);
        }
    }

}
