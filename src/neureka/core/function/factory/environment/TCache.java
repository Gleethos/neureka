package neureka.core.function.factory.environment;

import neureka.core.T;
import neureka.core.function.autograd.TGraphLock;
import neureka.core.function.autograd.TGraphNode;
import neureka.core.function.TFunction;

import java.util.TreeMap;
import java.util.function.Supplier;

public class TCache
{
    private final TreeMap<TGraphLock, TreeMap<TGraphNode, T>> TENSORS = new TreeMap<>((a, b)->((int)(a.hashCode()-b.hashCode())));

    public synchronized void free(T[] input){
        for(T t : input){
            TENSORS.remove(((TGraphNode)t.find(TGraphNode.class)).gid());
             t.remove(TGraphNode.class);
        }
    }
    public synchronized T handle(T[] input, TFunction function, Supplier<T> activation){
        T untracked = null;
        for(T t : input){
            if(t.has(TGraphNode.class)){
                untracked = t;
            }
        }
        if(untracked==null){
            return TFunction.execute(new T(), input, function);
        }
        for(T t : input){
            if(!t.has(TGraphNode.class)){
                TGraphLock gid = ((TGraphNode)untracked.find(TGraphNode.class)).gid();
                TGraphNode rg = new TGraphNode(t, null, null, gid);
                t.add(rg);
            }
        }
        TGraphNode node = (TGraphNode) input[0].find(TGraphNode.class);
        T result = get(node);
        if(result==null){
            result = activation.get();
            put(result, node);
        }
        return result;
    }
    private synchronized T get(TGraphNode node){//function and source
        if(TENSORS.containsKey(node.gid())){
            if(TENSORS.get(node.gid()).containsKey(node)){
                return TENSORS.get(node.gid()).get(node);
            }
        }
        return null;
    }
    private synchronized void put(T t, TGraphNode node){
        if(node.isCachable()) {
            TreeMap<TGraphNode, T> variables = null;
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
