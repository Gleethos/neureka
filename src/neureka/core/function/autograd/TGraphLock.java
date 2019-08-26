package neureka.core.function.autograd;
import neureka.core.T;
import neureka.core.function.TFunction;

public class TGraphLock {

    /**
     *  TGraphLock is a component of Tensors which lends it's identity
     *  as TreeMap key for function result caching and also in order to deny other functions
     *  access to tensors which are involved in the computation graph rendered by the 'lock owner',
     *  namely a TFunction object!
     * */
    private TFunction owner;
    //private long key;

    public TGraphLock(TFunction owner, T[] source){
        //long key = 1;
        //for(T t : source){
        //    key*=t.hashCode();//TODO: add version!
        //}
        //key+=owner.hashCode();
        this.owner = owner;
        //this.key = key;
    }

    public long key(){
       // return this.key;
        return this.hashCode();
    }

    public TFunction owner(){
        return this.owner;
    }

    @Override
    public String toString(){
        return "GID["+this.hashCode()+"]:( "+owner.toString()+" )";
    }

}
