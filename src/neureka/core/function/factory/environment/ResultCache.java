package neureka.core.function.factory.environment;

import neureka.core.T;
import neureka.core.autograd.TGradientNode;
import neureka.core.function.TFunction;
import neureka.core.function.TLock;

import java.util.TreeMap;
import java.util.function.Supplier;

public class ResultCache
{
    private final TreeMap<TLock, TreeMap<TGradientNode, T>> TENSORS = new TreeMap<>((a, b)->((int)(a.hashCode()-b.hashCode())));

    public synchronized void free(T[] input){
        for(T t : input){
            TENSORS.remove(((TGradientNode)t.find(TGradientNode.class)).gid());
             t.remove(TGradientNode.class);
        }
    }
    public synchronized T handle(T[] input, TFunction function, Supplier<T> activation){
        T untracked = null;
        for(T t : input){
            if(t.has(TGradientNode.class)){
                untracked = t;
            }
        }
        if(untracked==null){
            return TFunction.execute(new T(), input, function);
        }
        for(T t : input){
            if(!t.has(TGradientNode.class)){
                TLock gid = ((TGradientNode)untracked.find(TGradientNode.class)).gid();
                TGradientNode rg = new TGradientNode(t, null, null, gid);
                t.add(rg);
            }
        }
        TGradientNode node = (TGradientNode) input[0].find(TGradientNode.class);
        T result = get(node);
        if(result==null){
            result = activation.get();
            put(result, node);
        }
        return result;
    }
    private synchronized T get(TGradientNode node){//function and source
        if(TENSORS.containsKey(node.gid())){
            if(TENSORS.get(node.gid()).containsKey(node)){
                return TENSORS.get(node.gid()).get(node);
            }
        }
        return null;
    }
    private synchronized void put(T t, TGradientNode node){
        if(node.isCachable()) {
            TreeMap<TGradientNode, T> variables = null;
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
