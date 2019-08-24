package neureka.core.autograd;

import neureka.core.T;
import neureka.core.function.TFunction;
import neureka.core.function.TLock;

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
    private TreeMap<T, T> targets_gradients;

    /**
     * Forward or Backward AD ?
     * */
    private int mode = 0;

    private TLock gid = null;

    public TLock gid(){
        return gid;
    }

    public long nid(){
        long nid = 1;
        if(this.source!=null){
            for(T t : this.source){
                nid*=t.hashCode();
            }
        }
        if(this.function!=null){
            nid+=this.function.hashCode();
        }
        return nid;
    }

    public boolean isCachable(){
        return (this.nid()!=1);
    }

    public TGradientNode(T value, TFunction f, T[] src, TLock gid){
        this.mode = (src!=null)?modeOf(src, f):(value.rqsGradient())?1:0;
        this.function = f;
        this.source = src;
        this.gid = gid;
    }


    private static int modeOf(T[] source, TFunction function){
        /**
         *  Evaluate auto-grad mode:
         * */
        int mode = 0;
        int[] srcModes = new int[source.length];
        int m = 0;
        for(int Ii = 0; Ii< source.length; Ii++){
            if(source[Ii].has(TGradientNode.class)){
                TGradientNode node = (TGradientNode) source[Ii].find(TGradientNode.class);
                srcModes[Ii] = node.mode();
            }else if(source[Ii].rqsGradient()){
                srcModes[Ii] = 1;
            }
            m += (srcModes[Ii]!=0)?1:0;
        }
        if(m==1 && (function.type()!="x" && function.type()!=",")){//Convolution and reshaping prohibit forward AD
            for(int Ii = 0; Ii< source.length; Ii++){
                mode += (srcModes[Ii]<0)?1:srcModes[Ii];
            }
        }else{
            mode = -m;
        }
        return mode;
    }

    public void trimTree(T target){// Find and remove redundant targets:
        if(source==null){
            return;
        }
        if(target!=null){
            ArrayList<T> blacklist = new ArrayList<>();
            this.forEach((g, t)->{
                if(t==target){
                    blacklist.add(g);
                }
            });
            blacklist.forEach((b)->{
                if(this.function.id()!=18){
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
        /**
         * sources can be deleted because unused graph nodes are already trimmed off the tree (targets remain!)
         * */
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
        //TODO add delete!!(if targets (and derivatives) are used!)
        //ONLY USER VARIABLES REMAIN!!!
    }

    public int mode(){
        return this.mode;
    }

    public TFunction function(){
        return this.function;
    }

    public void put(T key, T value){
        if(targets_gradients==null){
            targets_gradients = new TreeMap<>((a, b)->a.hashCode()-b.hashCode());
        }
        targets_gradients.put(key, value);
    }
    public T get(T key){
        if(targets_gradients==null){
           return null;
        }
        return targets_gradients.get(key);
    }
    public boolean has(T key){
        if(targets_gradients==null){
            return false;
        }
        return targets_gradients.containsKey(key);
    }
    public Set<T> sources(){
        return targets_gradients.keySet();
    }

    public TreeMap<T, T> getMap(){
        return targets_gradients;
    }

    public int size(){
        return (this.targets_gradients!=null)?this.targets_gradients.size():0;
    }

    public void forEach(BiConsumer<T, T> action){
        if(targets_gradients==null){
            return;
        }
        targets_gradients.forEach(action);
    }

}
