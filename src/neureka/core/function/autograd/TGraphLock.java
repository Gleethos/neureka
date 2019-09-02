package neureka.core.function.autograd;
import neureka.core.T;
import neureka.core.function.TFunction;

public class TGraphLock {

    /**
     *  TGraphLock is a component of Tensors which lends it's identity
     *  as TreeMap key for function result caching and also in order to deny other functions
     *  access to tensors which are involved in the computation graph rendered by the 'lock _owner',
     *  namely a TFunction object!
     * */
    private TFunction _owner;

    public TGraphLock(TFunction owner, T[] source){
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
