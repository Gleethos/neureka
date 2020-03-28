package neureka.calculus.environment;

import neureka.Tsr;
import neureka.autograd.GraphLock;
import neureka.autograd.GraphNode;
import neureka.calculus.Function;

import java.util.*;
import java.util.function.Supplier;

public class Cache
{
    private static Cache _cache = new Cache();

    private Cache(){ }

    public static Cache instance(){
        Cache c = _cache;
        _cache = null;
        return c;
    }


    private final Map<String, Function> FUNCTIONS = Collections.synchronizedMap(new WeakHashMap<>());

    public synchronized Map<String, Function> FUNCTIONS(){
        return this.FUNCTIONS;
    }

    private final Map<GraphLock, TreeMap<Long, Tsr>> PROCESSING = Collections.synchronizedMap(new TreeMap<>((a, b)->((int)(a.hashCode()-b.hashCode()))));

    public synchronized void free(GraphLock lock)//Tsr[] input
    {
        PROCESSING.remove(lock);
        lock.release();
    }

    public synchronized Tsr preprocess(Tsr[] inputs, Function function, Supplier<Tsr> activation, int d, int j)
    {
        if (!function.doesAD()) {
            return activation.get();//TODO make caching possible!!, (without graph nodes!) REMEMBER: !doAD => NO GRAPH NODES
        }
        boolean locked = true;//input tensors might all have graph nodes but are left from previous computation. (=>need to locked again!)
        Tsr untracked = null;
        for (Tsr t : inputs){
            GraphNode node = (GraphNode) t.find(GraphNode.class);
            if (node != null) {
                untracked=t;
                locked = (locked)&&node.lock().isLocked();
            }
        }
        if(untracked==null || !locked){//If graph tracking (nodes) has not yet been initialized!
            return Function.Setup.commit(null, inputs, function, activation);
        }
        GraphLock lock = ((GraphNode)untracked.find(GraphNode.class)).lock();
        for (Tsr t : inputs){
            if (t.has(GraphNode.class)){
                ((GraphNode)t.find(GraphNode.class)).obtainLocking(lock);
            } else {
               new GraphNode(function, lock, ()->t);
                //t.add(rg);
            }
        }
        GraphNode node = (GraphNode) inputs[0].find(GraphNode.class);
        Tsr result = null;
        if (function != null && function.id() != OperationType.instance("<").id()&&function.id()!=OperationType.instance(">").id()){
            result = _get(node, d, j);
        }
        if(result == null){
            result = activation.get();
            _put(result, node, d, j);
        }
        //add references/child to graph node?
        return result;
    }

    private synchronized Tsr _get(GraphNode node, int d, int j)
    {
        long key = node.nid()+_keyed(d)*31+_keyed(j);
        if (PROCESSING.containsKey(node.lock())){
            if (PROCESSING.get(node.lock()).containsKey(key)) {
                return PROCESSING.get(node.lock()).get(key);
            }
        }
        return null;
    }

    private synchronized void _put(Tsr t, GraphNode node, int d, int j)
    {
        long key = node.nid()+_keyed(d)*31+_keyed(j);
        if (node.isCachable()) {
            TreeMap<Long, Tsr> variables;
            if (!PROCESSING.containsKey(node.lock())) {
                variables = new TreeMap<>((a, b) -> (a.hashCode() - b.hashCode()));
                PROCESSING.put(node.lock(), variables);
            } else {
                variables = PROCESSING.get(node.lock());
            }
            variables.put(key, t);
        }
    }

    private int _keyed(int number){
        return (number>=0)?number+1:number;
    }

}
