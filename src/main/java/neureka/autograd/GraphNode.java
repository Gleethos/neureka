package neureka.autograd;

import neureka.Neureka;
import neureka.Tsr;
import neureka.acceleration.opencl.utility.WeakTensorReference;
import neureka.calculus.Function;
import neureka.calculus.factory.assembly.FunctionBuilder;

import java.lang.ref.WeakReference;
import java.util.*;
import java.util.List;
import java.util.function.BiConsumer;
import java.util.function.Supplier;

/**
 *
 */
public class GraphNode
{
    private static Function MUL = FunctionBuilder.build("(I[0]*I[1])", false);
    private static Function ADD = FunctionBuilder.build("(I[0]+I[1])", false);
    private static Function INV_X = FunctionBuilder.build("I[0]x>>I[1]x>>I[2]", false);

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
     * This node propagates _backward.
     * @return boolean
     */
    public boolean usesReverseAD(){
        return (_mode <0);
    }

    /**
     *   mode state meaning:
     *  -----------+----------------------------------+-
     *  _mode == 0 |  no Auto-Differentiation         |
     *  -----------+----------------------------------+-
     *  _mode > 0  |  forward Auto-Differentiation    |
     *  -----------+----------------------------------+-
     *  _mode < 0  |  _backward Auto-Differentiation  |
     *  -----------+----------------------------------+-
     *
     * */
    private int _mode;

    /**
     * This flag is used for a performance optimization feature namely 'Just In Time Propagation'.
     * This feature accumulated errors and continues propagation
     * as soon as they are needed. (At the end of 'backward()' or when the tensor is used again).
     * If the flag Neureka.Settings.AD._retainPendingErrorForJITProp is set to true
     * then error values will accumulate whenever it makes sense.
     * This technique however uses more memory but will
     * improve performance for some networks substantially.
     *
     * All nodes between a Pending-Error and those requiring gradients will
     * be marked with '_relies_on_JIPProp=true'!
     */
    public  boolean reliesOnJustInTimeProp(){
        return _relies_on_JIPProp;
    }
    private boolean _relies_on_JIPProp = false;


    /**
     *
     */
    public PendingError getAndRemovePendingError(){
        PendingError pe = _pending_error;
        _pending_error = null;
        return pe;
    }
    private PendingError _pending_error = null;


    /**
     *  The chain-rule states that the derivative of f(x) = h(g(x)) with respect to x is: g'(x) * h'(g(x))
     *  An example would be:
     *  f(x) = ((x*y)*z)
     *  f'(x) = (1*y) * (1*z) = z*y
     *  The values z,y or z*y must not be deleted as they are needed for back-propagation!
     */
    public boolean isUsedAsDerivative(){
        return _is_used_as_derivative;
    }
    private boolean _is_used_as_derivative = false;


    /**
     * Recorded AbstractFunction.
     *
     * @var Function _function
     * */
    public Function getFunction(){
        return _function;
    }
    private Function _function;

    /**
     * Input tensors. ('Parents' of the tensor of this node)
     * */
    public GraphNode[] getParents(){
        return _parents;
    }
    private GraphNode[] _parents;

    /**
     * The value of this graph node!
     * This node belongs to a tensor during creation but may lose
     * it during memory cleanup : _targetedCleanup(Tsr target) -> payload might be deleted!
     *
     * @return the playload of this graph-node.
     */
    public Tsr getPayload(){
        return _payload;
    }
    private Tsr _payload;

    /**
     * Keys are targets and values are gradients with respect to that target
     * Note: values can be null if the recorded function is of type 'reshape'!
     * Why? => because reshape operation does not need variables for _backward pass!
     * */
    private TreeMap<GraphNode, Tsr> _targets_derivatives;

    /**
     * "Lock object" for graph identity. (result caching)
     * Unique object which locks the payload to the current computation graph.
     * @return GraphLock
     */
    public GraphLock lock(){
        return _lock;
    }
    private GraphLock _lock;

    /**
     *
     */
    public List<WeakReference<GraphNode>> getChildren(){
        return  _children;
    }
    private List<WeakReference<GraphNode>> _children;

    /**
     * @return long AbstractSurfaceNode-ID (Used for caching to avoid redundant computation within one computation graph)
     */
    public long nid(){
        return _nid;
    }
    private long _nid = -1;

    //==================================================================================================================

    /**
     * @param newLock
     */
    public synchronized void obtainLocking(GraphLock newLock){
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

    public boolean isGraphLeave(){
        if(isLeave()){
            return true;
        }
        for(GraphNode p : _parents){
            if(p.lock()!=this.lock()){
                return true;
            }
        }
        return false;
    }

    /**
     * @return if the tensor to which this graph node is attached has been deleted!
     */
    public boolean isVirtual(){
        return _payload==null;
    }

    /**
     *
     * @param function Is the function that lead to the creation of this node.
     * @param context Can be either an array of tensors or a new lock (for leave node or fresh function locking)
     * @param payloadSupplier Provides the payload of this node.
     */
    public GraphNode(Function function, Object context, Supplier<Tsr> payloadSupplier){
        if(context instanceof GraphLock){
            _construct(payloadSupplier.get(), function, null, (GraphLock)context);
        } else if (context instanceof Tsr[]){
            Tsr[] inputs = (Tsr[])context;
            /* Applying JITProp and gradients */
            if (Neureka.Settings.AD.applyGradientWhenTensorIsUsed()) {
                for (Tsr t : inputs) {
                    if (t.has(JITProp.class)) {
                        JITProp jit = (JITProp) t.find(JITProp.class);
                        jit.execute();
                    }
                }
                for (Tsr t : inputs) t.remove(JITProp.class);
                for (Tsr t : inputs) t.applyGradient();
            }
            _construct(payloadSupplier.get(), function, inputs, ((GraphNode) inputs[0].find(GraphNode.class)).lock());
        }
    }

    private void _construct(Tsr output, Function function, Tsr[] inputs, GraphLock lock)
    {
        if(output==null) throw new RuntimeException("[GraphNode]:(constructor): Payload must no be null!");
        _mode = (inputs!=null)? _modeOf(inputs, function):(output.rqsGradient())?1:0;
        _function = function;
        _lock = lock;
        _payload = output;
        if(inputs!=null)
        {
            _parents = new GraphNode[inputs.length];
            for(int i=0; i<inputs.length; i++) {
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
            if(_parents !=null) {
                for(GraphNode n : _parents) {
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
                        node.put(src_node, d);// Sources created by x-mul are reverse-mode cases!
                    }else{
                        if(src_node.usesAD()){
                            Tsr d = function.derive(inputs, i);
                            if(src_node.size()==0 && node.size()==0){
                                node.put((GraphNode) inputs[i].find(GraphNode.class), d);
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
                        node.put(src_node, d);// Add gradients with respect to every source tensor!
                    }
                    i++;
                }
            }
        }
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
            if(node == null){
                throw new IllegalStateException("[GraphNode]:(constructor): Input tensors of a new graph-node must contain graph-nodes!");
            }
            modes[Ii] = (inputs[Ii].rqsGradient())?1:node.mode();
            input_mode += (modes[Ii]!=0)?1:0;
        }
        if(input_mode==1 && ("x,".replace(function.type(), "").equals("x,"))){//Convolution and reshaping prohibit forward AD
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
            if(!node.isGraphLeave()){
                node._payloadDeletionDive(_mode);
                this.forEach((t, d)->{
                    if( this.mode()>0 || d==node.getPayload() ) node._targetedCleanup(t);
                });
            }
        }
    }

    /**
     * @param target The target node of a derivative which is being traversed. Other derivatives targeting it will be removed!
     */
    private void _targetedCleanup(GraphNode target)
    {// Find and remove redundant gradients sharing the same target: ... remove target payload if it is not used!
        if(target==null) throw new IllegalStateException("[GraphNode][_targetedCleanup]: target node must not be null!");
        if(_parents==null || mode()==0) return;//Gradient cleanup not needed in this case!
        boolean cleanForwardAD = (mode()>0);
        if(cleanForwardAD)
        {
            TreeMap<Tsr, GraphNode> blacklist = new TreeMap<>((a, b)->a.hashCode()-b.hashCode());
            this.forEach((t, d)->{ if(t==target) blacklist.put(d, t); });
            blacklist.forEach((b, t)->{
                if(!b.has(GraphNode.class) || !((GraphNode)b.find(GraphNode.class)).isLeave()){
                    _targets_derivatives.remove(t);
                    //TODO: get graph node and remove tensor reference! (this creates a virtual graph node (without payload!))
                    ((GraphNode)b.find(GraphNode.class))._payload = null;
                    b.delete();
                }
            });
            // Recursive cleanup: (but only within the current graph!)
            for(GraphNode node : _parents) if(!node.isGraphLeave()) node._targetedCleanup(target);
        }else{
            redundantGradientCleanup();
        }
        /** sources can be deleted because unused graph nodes are already trimmed off the tree (targets remain!)
         * */
        //TODO: query target through inputs... delete forward mode AD node tensors!
        //_parents = null;//This might not be necessary...
    }

    /**
     * The following properties must be true to allow payload deletion:
     * - The node is not a leave node! (AbstractSurfaceNode supplied by user/from outside the locked graph)
     * - The node is not a tip node! (Output node... ->($) )
     * - The node is part of a chain of forward-AD nodes (mode>0)
     * - The mode value of the node is smaller then the largest of another within a chain of forward-AD ($)
     * =>(The largest mode value is owned by 'the most recent derivative w.r.t some leave node')
     *
     * @param child_mode is used to assess if the payload in this node is useful for backpropagation!
     */
    private void _payloadDeletionDive(int child_mode)
    {
        if(!this.isLeave() && _payload!=null){
            if(_mode>0 && child_mode>_mode){//If _payload==null return maybe?? (because graph already clean?)
                _payload.remove(GraphNode.class);
                if(!_is_used_as_derivative) _payload.delete();
                _payload = null;
            } else if(_mode<0){
                if(!Neureka.Settings.Debug.keepDerivativeTargetPayloads()){
                    _payload.remove(GraphNode.class);
                    if(!_is_used_as_derivative) _payload.delete();
                    _payload = null;
                }
            }
        }
        if(_parents!=null){
            for(GraphNode n : _parents) if(!n.isGraphLeave()) n._payloadDeletionDive(_mode);
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
     * This method is the entry-point for the back-propagation process.
     * It sets up a key/value map which stores nodes and their intermediate error accumulations.
     * Accumulations occurs inside the private '_backward' method which traverses the computation graph
     * recursively, halts when errors can be accumulated, adds a PendingError and returns to the method below!
     * Here all the nodes and error values will then be carried (propagated) to the gradients!
     * @param error The current error which is created by multiplying it with current derivatives and traversing it.
     */
    public void backward(Tsr error){
        Map<GraphNode, PendingError> pendingBackProp = new LinkedHashMap<GraphNode, PendingError>(32, 0.777f);//new TreeMap<>((a, b)->a.hashCode()-b.hashCode());

        Set<GraphNode> pendingNodes = new HashSet<>();

        _backward(error, pendingNodes, false);// Entry-point to private recursive back-propagation!
        if(Neureka.Settings.AD.retainPendingErrorForJITProp()){
            pendingNodes.forEach((n)->n._carryPendingBackPropToGradients(pendingNodes));
        } else {
            pendingNodes.forEach((n)->{
                if(!n._pending_error.isFullyAccumulated()) throw new IllegalStateException("[GraphNode][backward]: Pending error has not received expected accumulation.");
                n.backward(n._pending_error.getAccumulatedError());//Continue back-propagation recursively!
            });
        }
        _deleteDerivativesRecursively();//Cleanup after back-propagation!
    }
    /**
     * This method traverses the graph and applies errors to gradients.
     *
     * Note: JITProp is enabled when
     * this node is on the path between
     * a pending error and a tensor (rqsGradient==true) waiting
     * to receive it.
     * When _backward is called and JITProp is true then this means
     * the method has been called by a JITProp class (stored at rqsGradient==true tensors...)
     *
     * The method traverses the computation graph and stores nodes
     * and their intermediate error accumulations  in 'toBeBackpropagated'.
     * The method halts when errors are accumulated and returns.
     *
     * @param error It is originally supplied by the user but later on is modified by derivatives...
     */
    private void _backward(Tsr error, Set<GraphNode> pendingNodes, boolean allowPendingError)//Map<GraphNode, PendingError> pendingBackProp, boolean allowPendingError)
    {
        if(_payload.isOutsourced()) _payload.device().add(error);
        if (_payload.rqsGradient()) _payload.addToGradient(error);

        if(this.usesAD())
        {
            /** Checking JIT-Prop conditions and create Pending error if possible **/
            if(allowPendingError && !this.isLeave())
            {//==> We are NOT inside a 'Just-In-Time-Backprop' process (new pending error can be created)
                int ADPaths = _numberOfReverseModeADChildren();// Multiple children triggers creation of a pending error
                if(ADPaths>1){
                    if(_pending_error==null){
                        _pending_error = new PendingError(error, ADPaths-1);
                        pendingNodes.add(this);
                    } else {
                        _pending_error.accumulate(error);
                    }
                    return;//Backprop will be continued later! This node is being remembered in 'PendingError'
                    // NOTE: Multiple AD paths leading to one node in history will be accumulated first! (performance)
                    //This optimization is a light version of JITProp. JITProp builds on this!
                }
            }

            if(this.usesForwardAD()) {//Using forward-AD derivatives for reverse-mode AD!:
                this.forEach((t, d)->t._backward(MUL.activate(new Tsr[]{error, d}), pendingNodes, true));
            }else if(this.usesReverseAD()){//Standard reverse mode-AD:
                this.forEach((t, d)->{
                    if(_function.id()==Function.TYPES.LOOKUP.get("x")){// x operation requires inverse convolve operation!
                        t._backward(INV_X.activate(new Tsr[]{error, d, new Tsr(t.getPayload().shape(), 0)}), pendingNodes, true);
                    } else {//Normal elementwise backpropagation:
                        t._backward(MUL.activate(new Tsr[]{error, d}), pendingNodes, true);
                    }
                });
            }
        }
    }

    /**
     * This method is called only if JIT-propagation is enabled.
     * It carries pending errors to the tensors requiring gradients which will
     * later on be processed just in time.
     * The path is being marked with '_relies_on_JITProp' so that intermediate derivatives will
     * not be deleted.
     * @param pendingBackProp
     */
    private void _carryPendingBackPropToGradients(Set<GraphNode> pendingBackProp){//Map<GraphNode, PendingError> pendingBackProp){
        _relies_on_JIPProp = true;
        this.forEach((t, d)->t._carryPendingBackPropToGradients(pendingBackProp));
        if(this.isLeave() && _payload.rqsGradient()){
            JITProp jit = (JITProp) _payload.find(JITProp.class);
            if(jit==null) jit = new JITProp(pendingBackProp);
            else jit.addPending(pendingBackProp);
            _payload.add(jit);
        }
        return;
    }

    /**
     * This method is called only when JITProp is active.
     * If an error has accumulated inside a JITProp component and
     * the component is triggered to continue pending backward calls
     * then this happens through this method.
     * The node from where the pending error stems from
     * is being passed down the graph (back in 'time')
     * in order to mark this error source as 'done'
     * so that other JITProp components do not propagate
     * this 'source' node multiple times.
     *
     * @param error
     */
    public void backwardJIT(Tsr error)
    {
        _backwardJIT(error, this);
        _deleteDerivativesRecursively();//Cleanup after back-propagation!
    }
    private void _backwardJIT(Tsr error, GraphNode source){
        if(_payload.isOutsourced()) _payload.device().add(error);
        if (_payload.rqsGradient()) _payload.addToGradient(error);
        JITProp jit = (JITProp) _payload.find(JITProp.class);//Get JIT-Prop node.
        if(jit!=null)jit.noteFinished(source);//note pending errors and store them as 'done'
        if(_pending_error!=null && source!=this) {
            _pending_error.accumulate(error);
            /*
              A pending error has been found, so this means that this node
              is referenced by one or more JIT-Prop components.
              If among these components is the one that issued this very
              traverse we are in at this moment, then this pending error at this node will later on
              be continued to be propagated.
              Otherwise it makes sense to accumulate errors further and wait for JIT-Prop traversing!
             */
            _pending_error = null;
            return;//This node will continue its propagation via a JIT-Prop component!
        }
        if(this.usesAD()) {
            if(this.usesForwardAD()){//Using forward-AD derivatives for reverse-mode AD!:
                this.forEach((t, d)->t._backwardJIT(MUL.activate(new Tsr[]{error, d}), source));
            }else if(this.usesReverseAD()){//Standard reverse mode-AD:
                this.forEach((t, d)->{
                    if(_function.id()==Function.TYPES.LOOKUP.get("x")){// x operation requires inverse convolve operation!
                        t._backwardJIT(INV_X.activate(new Tsr[]{error, d, new Tsr(t.getPayload().shape(), 0)}), source);
                    } else {//Normal elementwise backpropagation:
                        t._backwardJIT(MUL.activate(new Tsr[]{error, d}), source);
                    }
                });
            }
        }
    }

    /**
     * This method is called after the backward call has been executed fully.
     * Derivatives are no longer used and will therefore be deleted when possible.
     * Deletion is forbidden if this node is flagged
     * as JITProp job. This means that the node is on the path between gradients
     * and pending error objects.
     * Only if JITProp is enabled (Neureka.Settings.AD...) this flag will
     * deviate from its default state, namely: true!
     */
    private void  _deleteDerivativesRecursively(){
        if(!Neureka.Settings.AD.retainGraphDerivativesAfterBackward()){
            if(!reliesOnJustInTimeProp()) _targets_derivatives = null;
            this.forEach((t, d)->t._deleteDerivativesRecursively());
        }
        return;
    }


    /**
     * Counts how many child nodes will later on provide error values for back-propagation!
     * @return
     */
    private int _numberOfReverseModeADChildren(){
        int count = 0;
        if(_children!=null){
            for(WeakReference weak : _children){
                if(weak!=null && weak.get()!=null){
                    GraphNode child = (GraphNode) weak.get();
                    if(child.usesReverseAD()){
                        count++;
                    }
                }
            }
        }
        return count;
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
     * @param target nodes are graph nodes which contain either tensors requiring errors for accumulation and/or more targets.
     * @param derivative tensors are used during back-propagation in order to distribute an error throughout the graph.
     */
    public void put(GraphNode target, Tsr derivative){
        if(_targets_derivatives ==null){
            _targets_derivatives = new TreeMap<>((a, b)->a.hashCode()-b.hashCode());
        }
        _targets_derivatives.put(target, derivative);
        if(derivative.has(GraphNode.class)){
            ((GraphNode)derivative.find(GraphNode.class))._is_used_as_derivative = true;
        }
    }

    /**
     * @param target
     * @return Tsr
     */
    public Tsr get(GraphNode target){
        if(_targets_derivatives ==null) return null;
        return _targets_derivatives.get(target);
    }

    /**
     *
     * @param target
     * @return boolean
     */
    public boolean has(GraphNode target){
        if(_targets_derivatives ==null) return false;
        return _targets_derivatives.containsKey(target);
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
    public void forEach(BiConsumer<GraphNode, Tsr> action){
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
        return toString("");
    }


    public String toString(String m){
        if(m.contains("g")){
            return "]> LOCK: "+lock()+" |> GRAPH:\n]\n" +_toString("]    0", true) +"\n]\n]|END|>";
        }
        if(m.contains("v")){
            return "("+this.type()+"): [NID:"+Long.toHexString(nid())+"]:<(  "
                    +"f"+((_function==null)?"(NONE)":_function)+" => "+((_payload==null)?"NULL":_payload.toString("cs"))+"  )>";

        } else {
            return "[NID:"+Long.toHexString(nid())+"]:( "+((_payload==null)?"NULL":_payload.toString("cs"))+" )";
        }

    }

    private String _toString(String deep, boolean isLast){//int depth){
        String delimiter = ((isLast)?("    "):("|   "));
        String arrow = ((char)187)+""+((_parents!=null)?(String.valueOf(_parents.length)):"0")+((char)187);
        String asString = deep+
            arrow+ toString("v");
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
