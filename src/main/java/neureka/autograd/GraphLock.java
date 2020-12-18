package neureka.autograd;
import lombok.Getter;
import lombok.ToString;
import lombok.experimental.Accessors;
import neureka.Tsr;
import neureka.calculus.Function;

/**
 *  GraphLock is a component of tensors which lends it's identity
 *  as TreeMap key for function result caching and also in order to deny other functions
 *  access to tensors which are involved in the computation graph rendered by the 'lock owner',
 *  namely any Function object!
 * */
@Accessors( prefix = {"_"} )
public class GraphLock
{

    /**
     *  Owner of the lock of a graph:
     */
    private Function _owner;

    /**
     *  Lock status (is locked if the graph is currently processing)
     *
     *  @return Returns true if the graph is locked
     */
    @Getter
    private boolean _isLocked = true;

    /**
     * CONSTRUCTOR
     * @param owner The function which currently processes the graph of nodes of which this lock is referenced by.
     * @param sources
     */
    public GraphLock( Function owner, Tsr[] sources ) {
        this._owner = owner;
    }

    /**
     *  Releases this lock and permits nodes of this graph
     *  to be used for further processing.
     */
    public void release() {
        _isLocked = false;
    }

    /**
     * @return A description based on the identity of this lock and its owner (a function)!
     */
    @Override
    public String toString() {
        return "GID:"+Integer.toHexString(this.hashCode())+":f"+ _owner.toString()+"";
    }

}
