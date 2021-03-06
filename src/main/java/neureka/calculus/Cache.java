/*
MIT License

Copyright (c) 2019 Gleethos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*/

package neureka.calculus;

import neureka.Tsr;
import neureka.autograd.GraphLock;
import neureka.autograd.GraphNode;
import neureka.backend.api.OperationContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.function.Supplier;

/**
 *  This class is part of a given {@link OperationContext} instance
 *  responsible for caching {@link Function} references based on
 *  their {@link String} representation generated by {@link Object#toString()}
 *  as well as caching of results for active functions.
 */
public final class Cache
{
    private static Cache _cache = new Cache();
    private Logger _log = LoggerFactory.getLogger( Cache.class );

    public Cache() {
        _log.debug("New instance of class 'Cache' created for function result caching.");
    }

    public static Cache instance()
    {
        Cache c = _cache;
        _cache = null;
        return c;
    }

    private final Map<String, Function> functionCache = Collections.synchronizedMap( new WeakHashMap<>() );

    public synchronized Map<String, Function> functions() {
        return this.functionCache;
    }

    private final Map<GraphLock, TreeMap<Long, Tsr<Object>>> functionResultCache = Collections.synchronizedMap( new TreeMap<>( (a, b )-> a.hashCode() - b.hashCode() ) );

    public synchronized void free( GraphLock lock )
    {
        functionResultCache.remove( lock );
        lock.release();
    }

    public synchronized Tsr<Object> preprocess(
            Tsr<Object>[] inputs,
            Function function,
            Supplier<Tsr<Object>> activation,
            int d, int j
    ) {
        if ( !function.isDoingAD() ) {
            return activation.get(); // TODO make caching possible!!, (without graph nodes!) REMEMBER: !doAD => NO GRAPH NODES
        }
        boolean allLocked = true; // Input tensors might all have graph nodes which are left from previous computation.
        // ( => needs to be locked again! )
        Tsr<?> untracked = null;
        for ( Tsr<?> t : inputs ) {
            GraphNode<Object> node = t.find( GraphNode.class );
            if ( node != null ) {
                untracked = t;
                allLocked = node.getLock().isLocked() && allLocked;
            }
        }
        if ( untracked == null || !allLocked ) { // If graph tracking (nodes) has not yet been initialized!
            return Function.Setup.commit( null, inputs, function, activation );
        }
        GraphLock lock =  untracked.find( GraphNode.class ).getLock();
        for ( Tsr<Object> t : inputs ) {
            if ( t.has( GraphNode.class ) ) t.find( GraphNode.class ).obtainLocking( lock );
            else new GraphNode( function, lock, () -> t );
        }
        GraphNode<Object> node = inputs[ 0 ].find( GraphNode.class );
        Tsr<Object> result = null;

        if ( !function.getOperation().isInline() ) result = _get( inputs, d, j );

        if ( result == null ) {
            result = activation.get();
            _put( result, node, d, j );
        }
        // add references / child to graph node?
        return result;
    }

    private synchronized Tsr<Object> _get( Tsr<Object>[] tsrs, int d, int j )
    {
        GraphLock lock = tsrs[ 0 ].find( GraphNode.class ).getLock();
        long key = _keyOf( tsrs, d, j );
        if ( key != 0 && functionResultCache.containsKey( lock ) && functionResultCache.get( lock ).containsKey( key ) ) {
                _log.debug(
                        "Result cache hit occurred! Function lock : '{}'; Key : '{}';", lock, key
                );
                return functionResultCache.get( lock ).get( key );
        }
        return null;
    }

    private synchronized void _put( Tsr<Object> t, GraphNode<Object> node, int d, int j )
    {
        GraphNode[] nodes = node.getParents();
        Tsr[] tsrs = null;
        if ( nodes != null ) {
            tsrs = new Tsr[ nodes.length ];
            for ( int i=0; i<nodes.length; i++ ) tsrs[i] = nodes[i].getPayload();
        }
        long key = _keyOf( tsrs, d, j );
        if ( node.isCacheable() && key != 0 ) {
            TreeMap<Long, Tsr<Object>> variables;
            if ( functionResultCache.containsKey( node.getLock() ) ) variables = functionResultCache.get( node.getLock() );
            else {
                variables = new TreeMap<>((a, b) -> (a.hashCode() - b.hashCode()));
                functionResultCache.put( node.getLock(), variables );
            }
            variables.put( key, t );
        }
    }

    private long _keyOf( Tsr[] tsrs, int d, int j )
    {
        long key = 0;
        if ( tsrs == null ) return 0;
        for( int i = 0; i < tsrs.length; i++ ) {
            if ( tsrs[ i ] == null ) return 0; // Tensor has probably been garbage collected!
            key += ( ( i + 1 ) * tsrs[ i ].hashCode() ) + _keyed( d ) * 31 + _keyed( j );
        }
        return key;
    }

    private int _keyed( int number ) {
        return ( number>=0 ) ? number + 1 : number;
    }

}
