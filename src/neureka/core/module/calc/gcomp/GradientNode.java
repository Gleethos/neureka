package neureka.core.module.calc.gcomp;

import neureka.core.T;
import neureka.core.module.calc.fcomp.Function;

import java.util.Collection;
import java.util.HashMap;
import java.util.Set;
import java.util.function.BiConsumer;

public class GradientNode {

    private HashMap<T, T> map = new HashMap<T, T>();
    private int mode = 0;
    private Function function = null;
    private T[] source = null;

    public GradientNode(int m, Function f, T[] src){
        this.mode = m;
        this.function = f;
        this.source = src;
    }

    public void trimTree(){
        if(source==null){
            return;
        }
        for(T src : source){
            if(src.has(GradientNode.class)){
                GradientNode node = (GradientNode) src.find(GradientNode.class);
                node.trimTree();
                Collection<T> values = node.map.values();
                values.forEach((t)->{
                    if(map.containsValue(t)){

                    }
                });
            }
        }
        source = null;
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
    public void forEach(BiConsumer<T, T> action){
        map.forEach(action);
    }

}
