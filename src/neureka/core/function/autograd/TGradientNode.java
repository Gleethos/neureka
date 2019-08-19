package neureka.core.function.autograd;

import neureka.core.T;
import neureka.core.function.TFunction;

import java.util.*;
import java.util.function.BiConsumer;

public class TGradientNode{

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

    /**
     * Recorded Function and Source tensors
     * */
    private TFunction function = null;
    private T[] source = null;

    /**
     * Keys are targets and values are gradients with respect to that target
     * (Note: values can be null if the recorded function is of type 'reshape')
     * */
    private TreeMap<T, T> targets_gradients = new TreeMap<>((a, b)->a.hashCode()-b.hashCode());

    /**
     * Forward or Backward AD ?
     * */
    private int mode = 0;


    public TGradientNode(int m, TFunction f, T[] src){
        this.mode = m;
        this.function = f;
        this.source = src;
    }

    public void trimTree(T target){// Find and remove redundant targets:
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
                this.targets_gradients.remove(b);
            });
            for(T src : this.source){
                if(src.has(TGradientNode.class)){
                    TGradientNode node = (TGradientNode) src.find(TGradientNode.class);
                    node.trimTree(target);
                }
            }
        }else{
            for(T src : this.source){
                if(src.has(TGradientNode.class)){
                    TGradientNode node = (TGradientNode) src.find(TGradientNode.class);
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

    public TFunction function(){
        return this.function;
    }

    public void put(T key, T value){
        targets_gradients.put(key, value);
    }
    public T get(T key){
        return targets_gradients.get(key);
    }
    public boolean has(T key){
        return targets_gradients.containsKey(key);
    }
    public Set<T> sources(){
        return targets_gradients.keySet();
    }
    public void forEach(BiConsumer<T, T> action){
        targets_gradients.forEach(action);
    }

}
