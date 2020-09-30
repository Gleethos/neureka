package neureka.calculus.frontend;

import neureka.Tsr;
import neureka.autograd.GraphLock;
import neureka.autograd.GraphNode;
import neureka.calculus.Function;
import neureka.calculus.backend.operations.OperationType;

import java.util.*;
import java.util.function.Supplier;

public class Cache
{
    private static Cache _cache = new Cache();

    private Cache(){ }

    public static Cache instance()
    {
        Cache c = _cache;
        _cache = null;
        return c;
    }

    private final Map<String, Function> FUNCTIONS = Collections.synchronizedMap(new WeakHashMap<>());

    public synchronized Map<String, Function> FUNCTIONS(){
        return this.FUNCTIONS;
    }

    private final Map<GraphLock, TreeMap<Long, Tsr<Object>>> PROCESSING = Collections.synchronizedMap(new TreeMap<>((a, b)->((int)(a.hashCode()-b.hashCode()))));

    public synchronized void free( GraphLock lock )
    {
        PROCESSING.remove( lock );
        lock.release();
    }

    public synchronized Tsr<Object> preprocess( Tsr<Object>[] inputs, Function function, Supplier<Tsr<Object>> activation, int d, int j )
    {
        if ( !function.doesAD() ) {
            return activation.get();//TODO make caching possible!!, (without graph nodes!) REMEMBER: !doAD => NO GRAPH NODES
        }
        boolean locked = true;//input tensors might all have graph nodes but are left from previous computation. (=>need to locked again!)
        Tsr<Object> untracked = null;
        for ( Tsr<Object> t : inputs ) {
            GraphNode<Object> node = t.find(GraphNode.class);
            if ( node != null ) {
                untracked=t;
                locked = (locked)&&node.lock().isLocked();
            }
        }
        if( untracked==null || !locked ){ // If graph tracking (nodes) has not yet been initialized!
            return Function.Setup.commit(null, inputs, function, activation);
        }
        GraphLock lock =  untracked.find(GraphNode.class).lock();
        for ( Tsr<Object> t : inputs ){
            if ( t.has(GraphNode.class) ) t.find(GraphNode.class).obtainLocking(lock);
            else new GraphNode( function, lock, ()->t );
        }
        GraphNode<Object> node = inputs[0].find(GraphNode.class);
        Tsr<Object> result = null;

        if ( !function.getOperation().isInline() ) result = _get(inputs, d, j);

        if( result == null ){
            result = activation.get();
            _put( result, node, d, j );
        }
        // add references / child to graph node?
        return result;
    }

    private synchronized Tsr<Object> _get(Tsr<Object>[] tsrs, int d, int j)
    {
        GraphLock lock = null;
        long key = 0;
        for( int i = 0; i < tsrs.length; i++ ) {
            GraphNode<Object> node = tsrs[i].find( GraphNode.class );
            lock = node.lock();
            key += ( (i+1) * node.nid() ) + _keyed(d) * 31 + _keyed(j);
        }
        if ( PROCESSING.containsKey(lock) ) {
            if (PROCESSING.get(lock).containsKey(key)) return PROCESSING.get(lock).get(key);
        }
        return null;
    }

    private synchronized void _put( Tsr<Object> t, GraphNode node, int d, int j )
    {
        long key = node.nid() + _keyed(d) * 31 + _keyed(j);
        if ( node.isCachable() ) {
            TreeMap<Long, Tsr<Object>> variables;
            if ( PROCESSING.containsKey(node.lock()) ) variables = PROCESSING.get(node.lock());
            else {
                variables = new TreeMap<>((a, b) -> (a.hashCode() - b.hashCode()));
                PROCESSING.put( node.lock(), variables );
            }
            variables.put( key, t );
        }
    }

    private int _keyed( int number ){
        return ( number>=0 ) ? number + 1 : number;
    }

}
