package neureka.core.function.factory.autograd;

import neureka.core.T;
import neureka.core.function.IFunction;
import neureka.core.function.factory.Function;

import java.util.*;
import java.util.function.BiConsumer;

/**
 *
 */
public class GraphNode {

    /**
     *  This gradient node is involved in auto-differentiation.
     * @return boolean
     */
    public boolean usesAD(){
        return (_mode !=0);
    }

    /**
     *  This node propagates forward.
     * @return boolean
     */
    public boolean usesForwardAD(){
        return (_mode >0);
    }

    /**
     * This node propagates backward.
     * @return boolean
     */
    public boolean usesReverseAD(){
        return (_mode <0);
    }

    /**
     *   modes:   |
     *  ----------+----------------------------------+-
     *  _mode == 0 | no Auto-Differentiation         |
     *  ----------+----------------------------------+-
     *  _mode > 0  | forward Auto-Differentiation    |
     *  ----------+----------------------------------+-
     *  _mode < 0  | backward Auto-Differentiation   |
     *  ----------+----------------------------------+-
     *
     * @var int _mode
     * */
    private int _mode;

    /**
     * Recorded Function.
     *
     * @var IFunction _function
     * */
    private IFunction _function;

    /**
     * Input tensors. ('Parents' of the tensor of this node)
     * */
    private T[] _input;

    /**
     * Keys are targets and values are gradients with respect to that target
     * Note: values can be null if the recorded function is of type 'reshape'!
     * Why? => because reshape operation does not need variables for backward pass!
     * */
    private TreeMap<T, T> _targets_gradients;

    /**
     * "Lock object" for graph identity. (result caching)
     */
    private GraphLock _lock;

    //==================================================================================================================
    /**
     * @return GraphLock
     */
    public GraphLock lock(){
        return _lock;
    }

    /**
     * Node-ID
     * @return long
     */
    public long nid(){
        long nid = 1;
        if(_input !=null){
            for(T t : _input){
                nid*=t.hashCode();
            }
        }
        if(_function !=null){
            nid+=_function.hashCode();
        }
        return nid;
    }

    /**
     * Some nodes are not cachable! Namely: leave tensors! They are not results of
     * any function operation.
     * @return boolean
     */
    public boolean isCachable(){
        return (this.nid()!=1);
    }

    /**
     * This node (and the corresponding tensor) was not created by a function! (it's a leave tensor)
     * @return boolean
     */
    public boolean isOrigin(){
        return (_input ==null && _function==null);
    }

    /**
     * @param value
     * @param f
     * @param src
     * @param lock
     */
    public GraphNode(T value, IFunction f, T[] src, GraphLock lock){
        _mode = (src!=null)? _modeOf(src, f):(value.rqsGradient())?1:0;
        _function = f;
        _input = src;
        _lock = lock;
    }

    /**
     * @param source
     * @param function
     * @return int
     */
    private static int _modeOf(T[] source, IFunction function){
        /**
         *  Evaluate auto-grad mode:
         * */
        int mode = 0;
        int[] srcModes = new int[source.length];
        int m = 0;
        for(int Ii = 0; Ii< source.length; Ii++){
            if(source[Ii].has(GraphNode.class)){
                GraphNode node = (GraphNode) source[Ii].find(GraphNode.class);
                srcModes[Ii] = (source[Ii].rqsGradient())?1:node.mode();
            }else if(source[Ii].rqsGradient()){
                srcModes[Ii] = 1;
            }
            m += (srcModes[Ii]!=0)?1:0;
        }
        if(m==1 && ("x,".replace(function.type(), "")=="x,")){//Convolution and reshaping prohibit forward AD
            for(int Ii = 0; Ii< source.length; Ii++){
                mode += (srcModes[Ii]<0)?1:srcModes[Ii];
            }
        }else{
            mode = -m;
        }
        mode = ("<>".replace(function.type(), "")=="<>")?mode:0;
        return mode;
    }

    /**
     * @param target
     * @return void
     */
    public void trimTree(T target){// Find and remove redundant targets:
        if(_input ==null || mode()==0){
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
                if(!b.has(GraphNode.class) || !((GraphNode)b.find(GraphNode.class)).isOrigin()){
                    _targets_gradients.remove(t);
                    b.delete();
                }
            });
            for(T src : _input){
                if(src.has(GraphNode.class)){
                    GraphNode node = (GraphNode) src.find(GraphNode.class);
                    node.trimTree(target);
                }
            }
        }else{
            for(T src : _input){
                if(src.has(GraphNode.class)){
                    GraphNode node = (GraphNode) src.find(GraphNode.class);
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
        _input = null;
    }

    /**
     * @param error
     * @return void
     */
    public void backward(T error){
        if(this.usesAD()){
            if(this.usesForwardAD()){
                this.forEach((t, g)->t.backward(new T(new T[]{error, g}, "I[0]*I[1]", false)));
            }else if(this.usesReverseAD()){
                this.forEach((t, g)->{
                    if(_function.id()==18){// x operation required for chainrule!
                        //t.backward(new T(new T[]{error, g, new T(t.shape(), 0)}, "I[0]>>I[1]>>I[2]", false));
                        t.backward(Function.setup.commit(new T[]{error, g, new T(t.shape(), 0)}, "I[0]>>I[1]>>I[2]", false));
                    }else{
                        //t.backward(new T(new T[]{error, g}, "I[0]*I[1]", false));
                        t.backward(Function.setup.commit(new T[]{error, g}, "I[0]*I[1]", false));
                    }
                });
            }
        }
    }



    /**
     * @return int
     */
    public int mode(){
        return _mode;
    }

    /**
     *
     * @return IFunction
     */
    public IFunction function(){
        return _function;
    }

    /**
     * @param key
     * @param value
     */
    public void put(T key, T value){
        if(_targets_gradients ==null){
            _targets_gradients = new TreeMap<>((a, b)->a.hashCode()-b.hashCode());
        }
        _targets_gradients.put(key, value);
    }

    /**
     * @param key
     * @return T
     */
    public T get(T key){
        if(_targets_gradients ==null){
           return null;
        }
        return _targets_gradients.get(key);
    }

    /**
     *
     * @param key
     * @return boolean
     */
    public boolean has(T key){
        if(_targets_gradients ==null){
            return false;
        }
        return _targets_gradients.containsKey(key);
    }
    public Set<T> sources(){
        return _targets_gradients.keySet();
    }

    /**
     * @return TreeMap<T, T>
     */
    public TreeMap<T, T> getMap(){
        return _targets_gradients;
    }

    /**
     * @return int
     */
    public int size(){
        return (_targets_gradients !=null)?this._targets_gradients.size():0;
    }

    /**
     * @param action
     * @return void
     */
    public void forEach(BiConsumer<T, T> action){
        if(_targets_gradients ==null){
            return;
        }
        _targets_gradients.forEach(action);
    }

}
