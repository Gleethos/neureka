package neureka.core.function;
import neureka.core.function.TFunction;

public class TLock {

    /**
     *  TLock is a component of Tensors which lends it's identity as
     *  as TreeMap key for function result caching and also in order to deny other functions
     *  access to tensors which are involved in the computation graph rendered by the 'lock owner',
     *  namely a TFunction object!
     * */
    private TFunction owner;

    public TLock(TFunction owner){
        this.owner = owner;
    }

    public int key(){
        return this.hashCode();
    }

    public TFunction owner(){
        return this.owner;
    }

    @Override
    public String toString(){
        return this.hashCode()+"";
    }

}
