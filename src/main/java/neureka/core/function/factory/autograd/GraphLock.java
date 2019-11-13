package neureka.core.function.factory.autograd;
import neureka.core.Tsr;
import neureka.core.function.Function;

public class GraphLock
{
    /**
     *  GraphLock is a component of tensors which lends it's identity
     *  as TreeMap key for function result caching and also in order to deny other functions
     *  access to tensors which are involved in the computation graph rendered by the 'lock owner',
     *  namely any Function object!
     * */

    /**
     *  Owner of the lock of a graph:
     */
    private Function _owner;

    /**
     *  Lock status (is locked if the graph is currently processing)
     */
    private boolean _is_locked = true;

    /**
     * CONSTRUCTOR
     * @param owner => The function which currently processes the graph of nodes of which this lock is referenced by.
     * @param sources
     */
    public GraphLock(Function owner, Tsr[] sources){
        this._owner = owner;
    }

    /**
     * @return => Returns true if the graph is locked
     */
    public boolean isLocked(){
        return _is_locked;
    }

    /**
     *  Releases this lock and permits nodes of this graph
     *  to be used for further processing.
     */
    public void release(){
        _is_locked = false;
    }

    /**
     * @return A description based on the identity of this lock and its owner (a function)!
     */
    @Override
    public String toString(){
        return "GID:"+Integer.toHexString(this.hashCode())+":f"+ _owner.toString()+"";
    }

}
