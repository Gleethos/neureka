package neureka.function.factory.autograd;

import neureka.Tsr;
import neureka.acceleration.openCL.utility.WeakTensorReference;
import neureka.function.Function;
import neureka.function.factory.assembly.FunctionBuilder;

import java.lang.ref.WeakReference;
import java.util.*;
import java.util.function.BiConsumer;

/**
 *
 */
public class GraphNode
{
    private static Function MUL = FunctionBuilder.build("(I[0]*I[1])", false);
    private static Function ADD = FunctionBuilder.build("(I[0]+I[1])", false);
    private static Function CONV = FunctionBuilder.build("I[0]>>I[1]>>I[2]", false);

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
     * Recorded AbstractFunction.
     *
     * @var Function _function
     * */
    private Function _function;

    /**
     * Input tensors. ('Parents' of the tensor of this node)
     * */
    private GraphNode[] _parents;

    /**
     * The value of this graph node!
     * This node belongs to a tensor during creation but may lose
     * it during memory cleanup : _targetedCleanup(Tsr target) ! -> payload might be deleted!
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

    /**
     *  The chain-rule states that the derivative of f(x) = h(g(x)) with respect to x is: g'(x) * h'(g(x))
     *  An example would be:
     *  f(x) = ((x*y)*z)
     *  f'(x) = (1*y) * (1*z) = z*y
     *  The values z,y or z*y must not be deleted as they are needed for back-propagation!
     */
    private boolean _is_used_as_derivative = false;

    /**
     *
     */
    private List<WeakReference<GraphNode>> _children;

    /**
     *
     */
    private long _nid = -1;
    //==================================================================================================================
    /**
     * Unique object which locks the payload to the current computation graph.
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
        if(_children==null)_children = new ArrayList<>();
        WeakTensorReference<GraphNode> ref = new WeakTensorReference<GraphNode>(newChild, null);
        _children.add(ref);
    }

    /**
     * @return the playload of this graph-node.
     */
    public Tsr getPayload(){
        return _payload;
    }

    /**
     *
     * @return long Node-ID (Used for caching to avoid redundant computation within one computation graph)
     */
    public long nid(){
        return _nid;
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
    public boolean isLeave(){
        return (_parents ==null && _function==null);
    }

    /**
     * @return if the tensor to which this graph node is attached has been deleted!
     */
    public boolean isVirtual(){
        return _payload==null;
    }

    public boolean isUsedAsDerivative(){
        return _is_used_as_derivative;
    }

    /**
     * @param output
     * @param function
     * @param inputs
     * @param lock
     */
    public GraphNode(Tsr output, Function function, Tsr[] inputs, GraphLock lock)
    {
        if(output==null) throw new RuntimeException("[GraphNode]:(constructor): Payload must no be null!");
        _mode = (inputs!=null)? _modeOf(inputs, function):(output.rqsGradient())?1:0;
        _function = function;
        _lock = lock;
        _payload = output;
        if(inputs!=null){
            _parents = new GraphNode[inputs.length];
            for(int i=0; i<inputs.length; i++){
                _parents[i] = (GraphNode)inputs[i].find(GraphNode.class);
                if(_parents[i]==null){
                    throw new IllegalStateException("[GraphNode]:(constructor): Input tensors of a new graph-node must contain leave graph-nodes!");
                } else {
                    _parents[i]._attachChild(this);
                }
            }
        }else {
           _parents = null;
        }
        output.add(this);
        if(_nid==-1){
            long nid = 1;
            if(_parents !=null){
                for(GraphNode n : _parents){
                    nid*=n.getPayload().hashCode();
                }
            }
            if(_function !=null){
                nid+=_function.hashCode();
            }
            _nid = nid;
        }
        _connect(this, output, inputs, function);
    }

    private void _connect(GraphNode node, Tsr output, Tsr[] inputs, Function function)
    {
        /** Returning if the above cannot form an AD computation graph! :
         * */
        if(function==null || !function.isFlat()) return; // Leave nodes cannot be connected!!

        for(Tsr t : inputs) if(t.equals(output)) return;
        //--------------------------------------------------------------------------------------
        if(node.usesAD() && function.isFlat())
        {
            /**  Preparing for back propagation:  * */
            if(node.usesForwardAD())
            {
                int i = 0;
                for(Tsr input : inputs){
                    GraphNode src_node = ((GraphNode) input.find(GraphNode.class));
                    if(src_node.function()!=null && src_node.function().id()== Function.TYPES.LOOKUP.get("x")){
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
     * Evaluate auto-grad mode:
     * @param inputs
     * @param function
     * @return int
     */
    private static int _modeOf(Tsr[] inputs, Function function)
    {
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

    public void redundantGradientCleanup()
    {
        if(_parents==null || mode()==0) return;//Gradient cleanup not needed in this case!
        for(GraphNode node : _parents){
            node._payloadDeletionDive(_mode);
            this.forEach((t, d)->{
                if( this.mode()>0 || d==node.getPayload() ) node._targetedCleanup(t);
            });
        }
    }

    /**
     * @param target
     * @return void
     */
    private void _targetedCleanup(Tsr target)
    {// Find and remove redundant gradients sharing the same target: ...
        if(target==null) throw new IllegalStateException("[GraphNode][_targetedCleanup]: target tensor must not be null!");
        if(_parents==null || mode()==0) return;//Gradient cleanup not needed in this case!
        boolean cleanForwardAD = (mode()>0);
        if(cleanForwardAD)
        {
            TreeMap<Tsr, Tsr> blacklist = new TreeMap<>((a, b)->a.hashCode()-b.hashCode());
            this.forEach((t, d)->{ if(t==target) blacklist.put(d, t); });
            blacklist.forEach((b, t)->{
                if(!b.has(GraphNode.class) || !((GraphNode)b.find(GraphNode.class)).isLeave()){
                    _targets_derivatives.remove(t);
                    //TODO: get graph node and remove tensor reference! (this creates a virtual graph node (without payload!))
                    ((GraphNode)b.find(GraphNode.class))._payload = null;
                    b.delete();
                }
            });
            for(GraphNode node : _parents) node._targetedCleanup(target);
        }else{
            redundantGradientCleanup();
        }
        /** sources can be deleted because unused graph nodes are already trimmed off the tree (targets remain!)
         * */
        //TODO: find target through inputs... delete forward mode AD node tensors!
        //_parents = null;//This might not be necessary...
    }

    /**
     * The following properties must be true to allow payload deletion:
     * - The node is not a leave node! (Node supplied by user/from outside the locked graph)
     * - The node is not a tip node! (Output node... ->($) )
     * - The node is part of a chain of forward-AD nodes (mode>0)
     * - The mode value of the node is smaller then the largest of another within a chain of forward-AD ($)
     * =>(The largest mode value is owned by 'the most recent derivative w.r.t some leave node')
     *
     * @param child_mode is used to assess if the payload in this node is useful for backpropagation!
     */
    private void _payloadDeletionDive(int child_mode)
    {
        if(_mode>0 && child_mode>_mode && !this.isLeave() && _payload!=null){//If _payload==null return maybe?? (because graph already clean?)
            _payload.remove(GraphNode.class);
            if(!_is_used_as_derivative) _payload.delete();
            _payload = null;
        }
        if(_parents!=null){
            for(GraphNode n : _parents) n._payloadDeletionDive(_mode);
        }
    }

    /**
     * This method is called when a tensor is deleted and belongs to a computation graph.
     * All parents of this tensor will be checked if deletions is possible.
     * This is usually the case when the branch lineage is not tied to any other children!
     * @param child
     */
    public void extinguishLineageBy(GraphNode child)
    {
        boolean childrenAreDead = true;
        if(child==null){
            throw new IllegalStateException("[GraphNode][extinguishLineageBy]: Error! Child is null!");
        } else if(this!=child){
            boolean contains = false;
            int index = 0;
            for(int i=0; i<_children.size(); i++){
                if(_children.get(i)!=null){
                    if(_children.get(i).get().equals(child)){
                        contains = true;
                        index = i;
                    }
                }
            }
            if(!contains){
                throw new IllegalStateException("[GraphNode][extinguishLineageBy]: Error! Child is not recognized by parent!");
            }
            _children.set(index, null);
            for(int i=0; i<_children.size(); i++){
                childrenAreDead = (_children.get(i)==null) && childrenAreDead;
            }
        }
        if(childrenAreDead && !this.isLeave()){
            if(_payload!=null && !_is_used_as_derivative){
                _payload.remove(GraphNode.class);
                if(child!=this){
                    _payload.delete();
                }
            }
            if(_parents!=null){
                for(GraphNode parent : _parents){
                    parent.extinguishLineageBy(this);
                }
            }
            _function = null;
            _lock = null;
            _parents = null;
            _targets_derivatives = null;
            _children = null;
        }
    }

    /**
     * @param error
     * @return void
     */
    public void backward(Tsr error){
        if(this.usesAD()){
            if(_payload==null) throw new RuntimeException();
            if(this.usesForwardAD()){//Using forward-AD derivatives for reverse-mode AD!:
                this.forEach((t, d)->t.backward(MUL.activate(new Tsr[]{error, d})));
            }else if(this.usesReverseAD()){//Standard reverse mode-AD:
                this.forEach((t, d)->{
                    if(_function.id()==Function.TYPES.LOOKUP.get("x")){// x operation requires individual operation!
                        t.backward(CONV.activate(new Tsr[]{error, d, new Tsr(t.shape(), 0)}));
                    } else {//Normal elementwise backpropagation:
                        MUL.activate(new Tsr[]{error, d});
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
     * @return Function
     */
    public Function function(){
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
        if(_targets_derivatives ==null) return null;
        return _targets_derivatives.get(key);
    }

    /**
     *
     * @param target
     * @return boolean
     */
    public boolean has(Tsr target){
        if(_targets_derivatives ==null) return false;
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
        if(_targets_derivatives ==null) return;
        _targets_derivatives.forEach(action);
    }

    public String type(){
        String type = "";
        if(this.isLeave()) type+="LEAVE";
        else type += "BRANCH";
        if(_payload==null) type = type+" DELETED";
        else if(_payload.rqsGradient()) type += " RQS GRADIENT";
        return type;
    }

    @Override
    public String toString(){
        return "]> LOCK: "+lock()+" |> GRAPH:\n]\n"
                    +_toString("]    0", true)
                +"\n]\n]|END|>";
    }

    private String _toString(String deep, boolean isLast){//int depth){
        String delimiter = ((isLast)?("    "):("|   "));
        String arrow = ((char)187)+""+((_parents!=null)?(String.valueOf(_parents.length)):"0")+((char)187);
        String asString = deep+
            arrow+"("+this.type()+"): [NID:"+Long.toHexString(nid())+"]:<(  "
                +"f"+((_function==null)?"(NONE)":_function)+" => "+((_payload==null)?"NULL":_payload.toString("cs"))+"  )>";
        deep = deep.substring(0, deep.length()-1);
        if(_parents!=null){
            asString += "\n"+deep+((isLast)?"   \\\n":"|  \\\n");
            for(int i=0; i<_parents.length; i++){
                boolean last = (i==_parents.length-1);
                asString += ((i!=0)?deep+delimiter+"|\n":"");
                asString+=(_parents[i]._toString(deep+delimiter+i, last)+"\n");
            }
            asString = asString.substring(0, asString.length()-1);
        }
        return asString;
    }


}
