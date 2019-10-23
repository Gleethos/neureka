package neureka.core.function.factory.autograd;
import neureka.core.Tsr;
import neureka.core.function.IFunction;

public class GraphLock {

    /**
     *  GraphLock is a component of tensors which lends it's identity
     *  as TreeMap key for function result caching and also in order to deny other functions
     *  access to tensors which are involved in the computation graph rendered by the 'lock owner',
     *  namely any IFunction object!
     * */
    private IFunction _owner;

    private boolean _is_locked = true;

    public GraphLock(IFunction owner, Tsr[] source){
        this._owner = owner;
    }

    public boolean isLocked(){
        return _is_locked;
    }

    public void release(){
        _is_locked = false;
    }

    @Override
    public String toString(){
        return "GID["+this.hashCode()+"]:( "+ _owner.toString()+" )";
    }

}
