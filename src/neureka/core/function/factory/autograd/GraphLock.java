package neureka.core.function.factory.autograd;
import neureka.core.T;
import neureka.core.function.IFunction;

public class GraphLock {

    /**
     *  GraphLock is a component of Tensors which lends it's identity
     *  as TreeMap key for function result caching and also in order to deny other functions
     *  access to tensors which are involved in the computation graph rendered by the 'lock _owner',
     *  namely a IFunction object!
     * */
    private IFunction _owner;

    public GraphLock(IFunction owner, T[] source){
        this._owner = owner;
    }

    public long key(){
        return this.hashCode();
    }

    @Override
    public String toString(){
        return "GID["+this.hashCode()+"]:( "+ _owner.toString()+" )";
    }

}
