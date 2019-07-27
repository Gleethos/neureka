package neureka.core.module.calc.gcomp;

import neureka.core.T;
import neureka.core.module.calc.fcomp.Function;

import java.util.HashMap;
import java.util.Set;
import java.util.function.BiConsumer;

public class GradientNode {

    private HashMap<T, T> map = new HashMap<T, T>();
    private int mode = 0;
    private Function function = null;

    public GradientNode(int m, Function f){
        this.mode = m;
        this.function = f;
    }

    public int mode(){
        return this.mode;
    }
    public Function function(){
        return this.function;
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
