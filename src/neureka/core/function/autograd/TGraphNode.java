package neureka.core.function.autograd;

import neureka.core.T;
import neureka.core.function.TFunction;

import java.util.*;
import java.util.function.BiConsumer;

public class TGraphNode {

    /**
     *  These functions describe the meaning of '_mode'
     * */
    public boolean usesAD(){return (_mode !=0);}
    public boolean usesForwardAD(){ return (_mode >0); }
    public boolean usesReverseAD(){ return (_mode <0); }
    /**
     *  modes:    |
     *  ----------+----------------------------------+-
     *  _mode == 0 | no Auto-Differentiation          |
     *  ----------+----------------------------------+-
     *  _mode > 0  | forward Auto-Differentiation     |
     *  ----------+----------------------------------+-
     *  _mode < 0  | backward Auto-Differentiation    |
     *  ----------+----------------------------------+-
     * */

    /**
     * Recorded Function and Source tensors
     * */
    private TFunction _function = null;
    private T[] _source = null;

    /**
     * Keys are targets and _values are gradients with respect to that target
     * (Note: _values can be null if the recorded _function is of type 'reshape')
     * */
    private TreeMap<T, T> targets_gradients;

    /**
     * Forward or Backward AD ?
     * */
    private int _mode = 0;

    private TGraphLock gid = null;

    public TGraphLock gid(){
        return gid;
    }

    public long nid(){
        long nid = 1;
        if(_source !=null){
            for(T t : _source){
                nid*=t.hashCode();
            }
        }
        if(_function !=null){
            nid+=_function.hashCode();
        }
        return nid;
    }

    public boolean isCachable(){
        return (this.nid()!=1);
    }

    public boolean isOrigin(){
        return (_source ==null);
    }

    public TGraphNode(T value, TFunction f, T[] src, TGraphLock gid){
        _mode = (src!=null)?modeOf(src, f):(value.rqsGradient())?1:0;
        _function = f;
        _source = src;
        this.gid = gid;
    }


    private static int modeOf(T[] source, TFunction function){
        /**
         *  Evaluate auto-grad _mode:
         * */
        int mode = 0;
        int[] srcModes = new int[source.length];
        int m = 0;
        for(int Ii = 0; Ii< source.length; Ii++){
            if(source[Ii].has(TGraphNode.class)){
                TGraphNode node = (TGraphNode) source[Ii].find(TGraphNode.class);
                srcModes[Ii] = (source[Ii].rqsGradient())?1:node.mode();
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
        if(_source ==null || mode()==0){
            return;
        }
        boolean dive = (target==null || mode()<0);

        if(!dive){
            TreeMap<T, T> blacklist = new TreeMap<>((a, b)->a.hashCode()-b.hashCode());
            this.forEach((t, g)->{
                if(t==target){
                    blacklist.put(g, t);
                }
            });
            blacklist.forEach((b, t)->{
                if(!b.has(TGraphNode.class) || !((TGraphNode)b.find(TGraphNode.class)).isOrigin()){
                    this.targets_gradients.remove(t);
                    b.delete();
                }
            });
            for(T src : _source){
                if(src.has(TGraphNode.class)){
                    TGraphNode node = (TGraphNode) src.find(TGraphNode.class);
                    node.trimTree(target);
                }
            }
        }else{
            for(T src : _source){
                if(src.has(TGraphNode.class)){
                    TGraphNode node = (TGraphNode) src.find(TGraphNode.class);
                    this.forEach((t, g)->{
                        if(this.mode()>0 || g==src){
                            node.trimTree(t);
                        }
                    });
                }
            }
        }
        /**
         * sources can be deleted because unused graph nodes are already trimmed off the tree (targets remain!)
         * */
        _source = null;
    }

    public void backward(T error){
        if(this.usesAD()){
            if(this.usesForwardAD()){
                this.forEach((g, t)->t.backward(T.factory.multiplication(error, g)));
            }else if(this.usesReverseAD()){
                this.forEach((g, t)->{
                    if(_function.id()==18){// x operation required for chainrule!
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
        return _mode;
    }

    public TFunction function(){
        return _function;
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
