package neureka.core.function.environment;

import neureka.core.T;
import neureka.core.function.factory.autograd.GraphLock;
import neureka.core.function.factory.autograd.GraphNode;
import neureka.core.function.IFunction;

import java.util.TreeMap;
import java.util.function.Supplier;

public class TensorCache
{
    private final TreeMap<GraphLock, TreeMap<GraphNode, T>> TENSORS = new TreeMap<>((a, b)->((int)(a.hashCode()-b.hashCode())));

    public synchronized void free(T[] input)
    {
        for(T t : input){
            TENSORS.remove(((GraphNode)t.find(GraphNode.class)).gid());
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
        if(untracked==null){
            return IFunction.execute(new T(), input, function);
        }
        for(T t : input){
            if(!t.has(GraphNode.class)){
                GraphLock gid = ((GraphNode)untracked.find(GraphNode.class)).gid();
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
        if(TENSORS.containsKey(node.gid())){
            if(TENSORS.get(node.gid()).containsKey(node)){
                return TENSORS.get(node.gid()).get(node);
            }
        }
        return null;
    }

    private synchronized void put(T t, GraphNode node)
    {
        if(node.isCachable()) {
            TreeMap<GraphNode, T> variables = null;
            if (!TENSORS.containsKey(node.gid())) {
                variables = new TreeMap<>((a, b) -> (int) (a.nid() - b.nid()));
                TENSORS.put(node.gid(), variables);
            } else {
                variables = TENSORS.get(node.gid());
            }
            variables.put(node, t);
        }
    }

}
