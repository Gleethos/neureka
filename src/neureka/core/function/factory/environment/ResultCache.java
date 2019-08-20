package neureka.core.function.factory.environment;

import neureka.core.T;
import neureka.core.function.TFunction;
import neureka.core.function.TLock;

import java.util.TreeMap;
import java.util.function.Supplier;

public class ResultCache
{
    private final TreeMap<TLock, TreeMap<TFunction, T>> TENSORS = new TreeMap<>((a, b)->((int)(a.key()-b.key())));

    public synchronized TreeMap<TLock, TreeMap<TFunction, T>> TENSORS(){
        return this.TENSORS;
    }

    public synchronized void free(T[] input){
        for(T t : input){
            TENSORS.remove(t.find(TLock.class));
            t.remove(TLock.class);
        }
    }
    public synchronized T handle(T[] input, TFunction function, Supplier<T> activation){
        TLock lock = (TLock) input[0].find(TLock.class);
        T result = (!function.isFlat())? get(lock, function):null;
        if(result==null){
            result = activation.get();
            if(!function.isFlat()&&lock!=null) {
                put(result, lock, function);
            }
        }
        return result;
    }
    private synchronized T get(TLock lock, TFunction function){//function and source
        if(TENSORS.containsKey(lock)){
            if(TENSORS.get(lock).containsKey(function)){
                return TENSORS.get(lock).get(function);
            }
        }
        return null;
    }
    private synchronized void put(T t, TLock lock, TFunction function){
        TreeMap<TFunction, T> variables = null;
        if(!TENSORS.containsKey(lock)){
            variables = new TreeMap<>((a, b)->a.hashCode()-b.hashCode());
            TENSORS.put(lock, variables);
        }else{
            variables = TENSORS.get(lock);
        }
        variables.put(function, t);
    }
}
