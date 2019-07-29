package neureka.core.autograd;

import neureka.core.T;
import neureka.core.function.Function;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Set;
import java.util.function.BiConsumer;

public class GradientNode {

    /**
     *  These functions describe the meaning of 'mode'
     * */
    private boolean usesAD(){return (this.mode !=0);}
    private boolean usesForwardAD(){ return (this.mode >0); }
    private boolean usesReverseAD(){ return (this.mode <0); }
    /**
     *  modes:    |
     *  ----------+----------------------------------+-
     *  mode == 0 | no Auto-Differentiation          |
     *  ----------+----------------------------------+-
     *  mode > 0  | forward Auto-Differentiation     |
     *  ----------+----------------------------------+-
     *  mode < 0  | backward Auto-Differentiation    |
     *  ----------+----------------------------------+-
     * */

    private HashMap<T, T> map = new HashMap<T, T>();
    private int mode = 0;
    private Function function = null;
    private T[] source = null;

    public GradientNode(int m, Function f, T[] src){
        this.mode = m;
        this.function = f;
        this.source = src;
    }

    public void trimTree(T target){
        if(source==null){
            return;
        }
        if(target!=null){
            ArrayList<T> blacklist = new ArrayList<>();
            this.forEach((g, t)->{
                if(t.equals(target)){
                    blacklist.add(g);
                }
            });
            blacklist.forEach((b)->{
                if(true||this.function.id()!=18){
                    b.delete();
                }
                this.map.remove(b);
            });
            for(T src : this.source){
                if(src.has(GradientNode.class)){
                    GradientNode node = (GradientNode) src.find(GradientNode.class);
                    node.trimTree(target);
                }
            }
        }else{
            for(T src : this.source){
                if(src.has(GradientNode.class)){
                    GradientNode node = (GradientNode) src.find(GradientNode.class);
                    this.forEach((g, t)->node.trimTree(t));
                }
            }
        }
        this.source = null;
    }

    public void backward(T error){
        if(this.usesAD()){
            if(this.usesForwardAD()){
                this.forEach((g, t)->t.backward(T.factory.multiplication(error, g)));
            }else if(this.usesReverseAD()){
                this.forEach((g, t)->{
                    if(this.function.id()==18){// x operation required for chainrule!
                        t.backward(T.factory.convolution(error, g));
                    }else{
                        t.backward(T.factory.multiplication(error, g));
                    }
                });
            }
        }
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
