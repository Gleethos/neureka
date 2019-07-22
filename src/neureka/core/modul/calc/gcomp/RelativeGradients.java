package neureka.core.modul.calc.gcomp;

import neureka.core.T;

import java.util.HashMap;
import java.util.Set;
import java.util.function.BiConsumer;

public class RelativeGradients {

    private HashMap<T, T> map = new HashMap<T, T>();

    public RelativeGradients(){}
    public RelativeGradients(T derivative){
        map.put(derivative, T.factory.copyOf(derivative));
    }
    public void put(T key, T value){
        map.put(key, value);
    }
    public T get(T key){
        return map.get(key);
    }
    public boolean has(T key){
        return map.containsKey(key);
    }
    public Set<T> sources(){
        return map.keySet();
    }
    public void forEach(BiConsumer<T, T> action){ map.forEach(action); }

}
