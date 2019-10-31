package neureka.core.function.factory.autograd;

import neureka.core.Tsr;
import neureka.core.function.IFunction;
import neureka.core.function.factory.Function;
import neureka.core.function.factory.assembly.FunctionBuilder;

import java.util.*;
import java.util.function.BiConsumer;

/**
 *
 */
public class GraphNode
{

    private static IFunction MUL = FunctionBuilder.build("(I[0]*I[1])", false);
    private static IFunction ADD = FunctionBuilder.build("(I[0]+I[1])", false);


    /**
     *  This gradient64 node is involved in auto-differentiation.
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
    private GraphNode[] _parents;

    /**
     * The value of this graph node!
     * This node belongs to a tensor during creation but may lose
     * it during memory cleanup : trimTree(Tsr target) ! -> payload might be deleted!
     */
    private Tsr _payload;

    /**
     * Keys are targets and values are gradients with respect to that target
     * Note: values can be null if the recorded function is of type 'reshape'!
     * Why? => because reshape operation does not need variables for backward pass!
     * */
    private TreeMap<Tsr, Tsr> _targets_derivatives;

    /**
     * "Lock object" for graph identity. (result caching)
     */
    private GraphLock _lock;

    ///**
    // * How often the tensor of this graph node has been used as input to a function!
    // * */
    //private int _referenced_count;

    private boolean _is_used_as_derivative = false;

    private List<GraphNode> _children;

    //==================================================================================================================
    /**
     * @return GraphLock
     */
    public GraphLock lock(){
        return _lock;
    }

    /**
     * @param newLock
     */
    public synchronized void optainLocking(GraphLock newLock){
        _lock = newLock;
    }

    /**
     * @param newChild which references it's input namely the parent (this) has...
     */
    private synchronized void _attachChild(GraphNode newChild){
        if(_children==null){
            _children = new ArrayList<>();
        }
        _children.add(newChild);
    }

    /**
     * @return the playload of this graph-node.
     */
    public Tsr getPayload(){
        return _payload;
    }

    /**
     * Node-ID
     * @return long
     */
    public long nid(){
        long nid = 1;
        if(_parents !=null){
            for(GraphNode n : _parents){
                nid*=n.getPayload().hashCode();
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
        return (_parents ==null && _function==null);
    }

    /**
     * @return if the tensor to which this graph node is attached has been deleted!
     */
    public boolean isVirtual(){
        return _payload==null;
    }

    /**
     * @param output
     * @param f
     * @param inputs
     * @param lock
     */
    public GraphNode(Tsr output, IFunction f, Tsr[] inputs, GraphLock lock)
    {
        _mode = (inputs!=null)? _modeOf(inputs, f):(output.rqsGradient())?1:0;
        _function = f;
        _lock = lock;
        _payload = output;
        if(inputs!=null){
            _parents = new GraphNode[inputs.length];
            for(int i=0; i<inputs.length; i++){
                _parents[i] = (GraphNode)inputs[i].find(GraphNode.class);
                if(_parents[i]==null){
                    throw new IllegalStateException("[GraphNode]:(constructor): Input tensors of a new graph-node must contain origin graph-nodes!");
                } else {
                    _parents[i]._attachChild(this);
                }
            }
        }else {
           _parents = null;
        }
        output.add(this);
        _connect(this, output, inputs, f);
    }

    private void _connect(GraphNode node, Tsr output, Tsr[] inputs, IFunction function)
    {
        /** Returning if the above cannot form an AD computation graph! :
         * */
        if(function==null || !function.isFlat()) return; // Origin nodes cannot be connected!!

        for(Tsr t : inputs){
            if(t.equals(output)) return;
        }
        //--------------------------------------------------------------------------------------
        if(node.usesAD() && function.isFlat())
        {
            /**  Preparing for back propagation:  * */
            if(node.usesForwardAD())
            {
                int i = 0;
                for(Tsr input : inputs){
                    GraphNode src_node = ((GraphNode) input.find(GraphNode.class));
                    if(src_node.function()!=null && src_node.function().id()==IFunction.TYPES.LOOKUP.get("x")){
                        Tsr d = function.derive(inputs, i);//TODO: is this ever used? / visited? - yes but why?
                        node.put(input, d);// Sources created by x-mul are reverse-mode cases!
                    }else{
                        if(src_node.usesAD()){
                            Tsr d = function.derive(inputs, i);
                            if(src_node.size()==0 && node.size()==0){
                                node.put(inputs[i], d);
                            } else {
                            /**  Chain rule (forward) for every _gradient w.r.t. leaves (reverseAD or user leaves):* */
                                src_node.forEach(
                                    (t, g)->{
                                        if(node.has(t)){
                                            Tsr dg = node.get(t);
                                            node.put(t, ADD.activate(new Tsr[]{dg, MUL.activate(new Tsr[]{d, g})}));
                                        }else{
                                            node.put(t, MUL.activate(new Tsr[]{d, g}));
                                        }//TODO: flag within src tsrs that grant that the tensor has been created by function constructor!
                                    });
                            }
                        }
                    }
                    i++;
                }
            }
            else if(node.usesReverseAD())
            {
                int i = 0;
                for(Tsr input : inputs){
                    GraphNode src_node = ((GraphNode) input.find(GraphNode.class));
                    if(src_node.mode()!=0 || input.rqsGradient()){
                        Tsr d = function.derive(inputs, i);
                        node.put(input, d);// Add gradients with respect to every source tensor!
                    }
                    i++;
                }
            }
        }
        //--------------------------------------------------------------------------------------
    }


    /**
     * @param inputs
     * @param function
     * @return int
     */
    private static int _modeOf(Tsr[] inputs, IFunction function){
        /**
         *  Evaluate auto-grad mode:
         * */
        int result_mode = 0;
        int[] modes = new int[inputs.length];
        int input_mode = 0;
        for(int Ii = 0; Ii< inputs.length; Ii++){
            GraphNode node = (GraphNode) inputs[Ii].find(GraphNode.class);
            modes[Ii] = (inputs[Ii].rqsGradient())?1:node.mode();
            input_mode += (modes[Ii]!=0)?1:0;
        }
        if(input_mode==1 && ("x,".replace(function.type(), "")=="x,")){//Convolution and reshaping prohibit forward AD
            for(int Ii = 0; Ii< inputs.length; Ii++){
                result_mode += (modes[Ii]==0)?0:(modes[Ii]<0)?1:modes[Ii]+1;
            }
        }else{
            result_mode = -input_mode;
        }
        result_mode = ("<>".replace(function.type(), "")=="<>")?result_mode:0;
        return result_mode;
    }

    /**
     * @param target
     * @return void
     */
    public void trimTree(Tsr target)
    {// Find and remove redundant gradients: ... todo: and maybe forward ad nodes?!?!?!?!?
        if(_parents==null || mode()==0){
            return;
        }
        boolean dive = (target==null || mode()<0);
        if(!dive){
            TreeMap<Tsr, Tsr> blacklist = new TreeMap<>((a, b)->a.hashCode()-b.hashCode());
            this.forEach((t, g)->{
                if(t==target){
                    blacklist.put(g, t);
                }
            });
            blacklist.forEach((b, t)->{
                if(!b.has(GraphNode.class) || !((GraphNode)b.find(GraphNode.class)).isOrigin()){
                    _targets_derivatives.remove(t);
                    //TODO: get graph node and remove tensor reference! (this creates a virtual graph node (without payload!))
                    ((GraphNode)b.find(GraphNode.class))._payload = null;
                    b.delete();
                }
            });
            for(GraphNode node : _parents){
                node.trimTree(target);
            }
        }else{
            for(GraphNode node : _parents){
                node._deletionDive(_mode);
                this.forEach((t, g)->{
                    if(this.mode()>0 || g==node.getPayload()){
                        node.trimTree(t);
                    }
                });
            }
        }
        /** sources can be deleted because unused graph nodes are already trimmed off the tree (targets remain!)
         * */
        //TODO: find target through inputs... delete forward mode AD node tensors!
        //_parents = null;//This might not be necessary...
    }

    /**
     * The following properties must be true to allow payload deletion:
     * - The node is not a origin node! (Node supplied by user/from outside the locked graph)
     * - The node is part of a chain of forward-AD nodes (mode>0)
     * - The mode value of the node is smaller then the largest of anoher within a chain of forward-AD
     * =>(The largest mode value is owned by 'the most recent derivative w.r.t some origin node')
     *
     * @param child_mode is used to assess if the payload in this node is useful for backpropagation!
     */
    private void _deletionDive(int child_mode){
        if(_mode>0 && child_mode>_mode && !this.isOrigin()){
            _payload.remove(GraphNode.class);
            if(!_is_used_as_derivative){
                _payload.delete();
            }
            _payload = null;
        }
        if(_parents!=null){
            for(GraphNode n : _parents){
                n._deletionDive(_mode);
            }
        }
    }

    /**
     * @param error
     * @return void
     */
    public void backward(Tsr error){
        if(this.usesAD()){
            if(this.usesForwardAD()){
                this.forEach((t, g)->t.backward(new Tsr(new Tsr[]{error, g}, "I[0]*I[1]", false)));
            }else if(this.usesReverseAD()){
                this.forEach((t, g)->{
                    if(_function.id()==Function.TYPES.LOOKUP.get("x")){// x operation required for chain-rule!
                        t.backward(Function.setup.commit(new Tsr[]{error, g, new Tsr(t.shape(), 0)}, "I[0]>>I[1]>>I[2]", false));
                    }else{
                        t.backward(Function.setup.commit(new Tsr[]{error, g}, "I[0]*I[1]", false));
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
     * @return IFunction
     */
    public IFunction function(){
        return _function;
    }

    /**
     * @param target
     * @param derivative
     */
    public void put(Tsr target, Tsr derivative){
        if(_targets_derivatives ==null){
            _targets_derivatives = new TreeMap<>((a, b)->a.hashCode()-b.hashCode());
        }
        _targets_derivatives.put(target, derivative);
        if(derivative.has(GraphNode.class)){
            ((GraphNode)derivative.find(GraphNode.class))._is_used_as_derivative = true;
        }
    }

    /**
     * @param key
     * @return Tsr
     */
    public Tsr get(Tsr key){
        if(_targets_derivatives ==null){
           return null;
        }
        return _targets_derivatives.get(key);
    }

    /**
     *
     * @param target
     * @return boolean
     */
    public boolean has(Tsr target){
        if(_targets_derivatives ==null){
            return false;
        }
        return _targets_derivatives.containsKey(target);
    }

    /**
     * @return Map<Tsr, Tsr>
     */
    public Map<Tsr, Tsr> getMap(){
        return _targets_derivatives;
    }

    /**
     * @return int
     */
    public int size(){
        return (_targets_derivatives !=null)?this._targets_derivatives.size():0;
    }

    /**
     * @param action
     * @return void
     */
    public void forEach(BiConsumer<Tsr, Tsr> action){
        if(_targets_derivatives ==null){
            return;
        }
        _targets_derivatives.forEach(action);
    }

}
