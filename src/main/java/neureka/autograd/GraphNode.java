package neureka.autograd;

import neureka.Component;
import neureka.Neureka;
import neureka.Tsr;
import neureka.acceleration.Device;
import neureka.acceleration.opencl.utility.WeakTensorReference;
import neureka.calculus.Function;
import neureka.calculus.environment.ExecutionCall;

import java.lang.ref.WeakReference;
import java.util.*;
import java.util.List;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.Supplier;

/**
 *  Instances of this class are components of tensors.
 *  GraphNodes form a computation graph during runtime which is traversed during backpropagation.
 *  Both parent and child references are use.
 *  Parents are the GraphNodes of the tensors from which the tensor of the current node was formed,
 *  whereas children are the nodes (also) produced by said current node.
 *  Children are weakly referenced so that abandoned / detached
 *  graph branches (child nodes) can be garbage collected...
 *  ...whereas parents are strongly referenced in order to enable traversal.
 */
public class GraphNode implements Component<Tsr>
{
    private final static Function MUL =  Function.Detached.MUL;
    private final static Function ADD = Function.Detached.ADD;

    /**
     * This gradient node is involved in auto-differentiation.
     *
     * @return boolean
     */
    public boolean usesAD() {
        return (_mode != 0);
    }

    /**
     * This node propagates forward.
     *
     * @return boolean
     */
    public boolean usesForwardAD() {
        return (_mode > 0);
    }

    /**
     * This node propagates _backward.
     *
     * @return boolean
     */
    public boolean usesReverseAD() {
        return (_mode < 0);
    }

    /**
     * mode state meaning:
     * -----------+----------------------------------+-
     * _mode == 0 |  no Auto-Differentiation         |
     * -----------+----------------------------------+-
     * _mode > 0  |  forward Auto-Differentiation    |
     * -----------+----------------------------------+-
     * _mode < 0  |  backward Auto-Differentiation   |
     * -----------+----------------------------------+-
     */
    private int _mode;

    /**
     *  This flag records the support evaluation of the forward-AD availability analysis
     *  done in the corresponding OperationTypeImplementation lambda
     *  for a given ExecutionCall instance.
     *
     *  The difference between this flag and the "usesForwardAD()" truth value
     *  is that the latter one can be false while the prior is true!
     *  ( However the reverse is not possible! )
     *  The reason is as follows:
     *  If a GraphNode has multiple parent nodes which require auto-differentiation,
     *  then said node will not be able to perform forward-AD even though it might very well
     *  be possible given an ExecutionCall whose state allows for such...
     */
    private boolean _allows_forward;

    /**
     * This flag is used for a performance optimization feature namely 'Just In Time Propagation'.
     * This feature accumulates errors and continues propagation
     * as soon as they are needed. (At the end of 'backward()' or when the tensor is used again).
     * If the flag Neureka.instance().settings().AutoDiff()._retainPendingErrorForJITProp is set to true
     * then error values will accumulate whenever it makes sense.
     * This technique however uses more memory but will
     * improve performance for some networks substantially.
     * <p>
     * All nodes between a Pending-Error and those requiring gradients will
     * be marked with '_relies_on_JIPProp=true'!
     */
    public boolean reliesOnJustInTimeProp() {
        return _relies_on_JIPProp;
    }

    private boolean _relies_on_JIPProp = false;


    /**
     * This method is called by the JITProp component.
     * A pending should only ever be retrieved from a GraphNode once because
     * afterwards the accumulated error is about to be backpropagated.
     * Therefore this method nulls the reference when returning the PendingError instance.
     * @return Returns an instance of the PendingError class containing a error accumulation.
     */
    public PendingError getAndRemovePendingError() {
        PendingError pe = _pending_error;
        _pending_error = null;
        return pe;
    }

    /**
     * Used by the Just-In-Time backprop component.
     */
    private PendingError _pending_error = null;


    /**
     * The chain-rule states that the derivative of f(x) = h(g(x)) with respect to x is: g'(x) * h'(g(x))
     * An example would be:
     * f(x) = ((x*y)*z)
     * f'(x) = (1*y) * (1*z) = z*y
     * The values z,y or z*y must not be deleted as they are needed for back-propagation!
     */
    public boolean isUsedAsDerivative() {
        return _is_used_as_derivative;
    }

    private boolean _is_used_as_derivative = false;


    /**
     * Recorded Function which produced this GrphNode.
     */
    public Function getFunction() {
        return _function;
    }

    private Function _function;

    /**
     * The GraphNodes of the input tensors. ('Parents' of the tensor of this node)
     * These are always the GraphNodes of the tensors from which the tensor payload of this
     * GraphNode has been formed.
     */
    public GraphNode[] getParents() {
        return _parents;
    }

    private GraphNode[] _parents;

    /**
     * The value of this graph node!
     * This node belongs to a tensor during creation.
     * The payload is referenced weakly and might be garbage collected.
     * When the tensor becomes phantom reachable the lambda defined
     * in this method will be executed.
     * It is stored inside the Cleaner within the device of the payload.
     * Cleaning means to null the targets_derivatives map.
     * Leaning however only occurs if the payload reference is still null.
     * If it is not null then this means that the payload
     * changed (happens during injection)
     *
     * @return the payload of this graph-node.
     */
    public Tsr getPayload() {
        return (_payload == null) ? null : _payload.get();
    }

    private void _setPayload( Tsr p ) {
        if ( p == null ) {
            _payload = null;
        } else {
            _payload = new WeakReference<>(p);
            p.device().cleaning(p, () -> {
                if (this.getPayload() == null) {
                    boolean allChildrenUseForwardAD = true;
                    if ( _children != null ) {
                        for ( WeakReference<GraphNode> childRef : _children ) {
                            GraphNode childNode = childRef.get();
                            if (childNode != null && childNode.usesReverseAD()) allChildrenUseForwardAD = false;
                        }
                    }
                    if (allChildrenUseForwardAD) _targets_derivatives = null;
                }
            });
        }
    }

    /**
     * This is the tensor owning this GraphNode component.
     * It is referenced weakly because it might not be needed anymore (Not referenced inside AD-Agent for example)
     * and can therefore be garbage collected.
     */
    private WeakReference<Tsr> _payload;

    @Override
    public void update(Tsr oldOwner, Tsr newOwner) {
        _setPayload(newOwner);
    }

    /**
     * Keys are targets and values are gradients with respect to that target
     * Note: values can be null if the recorded function is of type 'reshape'!
     * Why? => because reshape operation does not need variables for _backward pass!
     */
    private TreeMap<GraphNode, ADAgent> _targets_derivatives;

    /**
     * "Lock object" for graph identity. (result caching)
     * Unique object which locks the payload to the current computation graph.
     *
     * @return GraphLock
     */
    public GraphLock lock() {
        return _lock;
    }

    private GraphLock _lock;

    /**
     *
     */
    public List<WeakReference<GraphNode>> getChildren() {
        return _children;
    }

    private List<WeakReference<GraphNode>> _children;

    /**
     * @return long AbstractSurfaceNode-ID (Used for caching to avoid redundant computation within one computation graph)
     */
    public long nid() {
        return _nid;
    }

    private long _nid = -1;

    //==================================================================================================================

    /**
     * @param newLock The new lock of this GraphNode.
     */
    public synchronized void obtainLocking(GraphLock newLock) {
        _lock = newLock;
    }

    /**
     * @param newChild which references it's input namely the parent (this) has...
     */
    private synchronized void _attachChild(GraphNode newChild) {
        if ( _children == null ) _children = new ArrayList<>();
        WeakReference<GraphNode> ref = new WeakTensorReference<>( newChild, null );
        _children.add( ref );
    }

    /**
     * Some nodes are not cachable! Namely: leave tensors! They are not results of
     * any function operation.
     *
     * @return boolean
     */
    public boolean isCachable() {
        return ( this.nid() != 1 );
    }

    /**
     * This node (and the corresponding tensor) was not created by a function! (it's a leave tensor)
     *
     * @return boolean
     */
    public boolean isLeave() {
        return ( _parents == null && _function == null );
    }

    public boolean isGraphLeave() {
        if ( this.isLeave() ) return true;
        for ( GraphNode p : _parents ) {
            if ( p.lock() != this.lock() ) return true;
        }
        return false;
    }

    /**
     * @return if the tensor to which this graph node is attached has been deleted!
     */
    public boolean isVirtual() {
        return getPayload() == null;
    }

    /**
     * @param function        Is the function that lead to the creation of this node.
     * @param context         Can be either an array of tensors or a new lock (for leave node or fresh function locking)
     * @param payloadSupplier Provides the payload of this node.
     */
    public GraphNode( Function function, Object context, Supplier<Tsr> payloadSupplier )
    {
        if ( function == null ) throw new IllegalArgumentException("Passed constructor argument of type Function must not be null!");
        if ( context instanceof GraphLock ) { // Note function always null in this case:
            _construct( payloadSupplier.get(), function, null, (GraphLock) context );
        } else if ( context instanceof ExecutionCall ) {
            ExecutionCall call = (ExecutionCall) context;
            Tsr[] inputs = call.getTensors();
            /* Applying JITProp and gradients */
            Neureka.Settings.AutoGrad adSetting = Neureka.instance().settings().autograd();
            if ( adSetting.isApplyingGradientWhenTensorIsUsed() ) {
                for ( Tsr t : inputs ) {
                    if( !adSetting.isApplyingGradientWhenRequested() || t.gradientApplyRqd() ) {
                        t.forComponent( JITProp.class, JITProp::execute );
                        t.remove( JITProp.class );
                        t.applyGradient();
                        t.setGradientApplyRqd( false );
                    }
                }
            }
            GraphLock foundLock = null;
            for ( int i=0; i< inputs.length; i++ ) {
                GraphNode child = inputs[i].find( GraphNode.class );
                if ( child == null ) throw new IllegalStateException(
                        "Input tensor at index '"+i+"' did not return a GraphNode instance." +
                                "Input tensors of a new GraphNode must be part of the computation graph!"
                );
                if ( foundLock == null ) foundLock = child.lock();
                if ( foundLock != child.lock() ) {
                    throw new IllegalStateException(
                            "GraphNode instances found in input tensors do not share the same GraphLock instance.\n" +
                                    "The given input tensors of a new node must be part of the same locked computation graph!"
                    );
                }
            }
            _construct( payloadSupplier.get(), function, call, inputs[0].find(GraphNode.class).lock() );
        } else {
            throw new IllegalArgumentException(
                    "The passed context object for the GraphNode constructor is of type '"+context.getClass().getName()+"'.\n" +
                            "A given context must either be a GraphLock instance or an ExecutionCall."
            );
        }
    }

    private void _construct(Tsr output, Function function, ExecutionCall<Device> call, GraphLock lock)
    {
        Tsr[] inputs = ( call == null ) ? null : call.getTensors();
        if ( output == null ) throw new NullPointerException("The supplied payload Tsr must no be null!");
        if ( !function.doesAD() ) return; // Only functions with AutoDiff enabled create computation graph!
        _lock = lock;
        _setPayload( output );
        output.add( this );
        if ( inputs == null ) {
            _mode = ( output.rqsGradient() ) ? 1 : 0;
            _function = null;
            _parents = null;
        } else {
            _mode = _modeOf( call, function );
            _function = function;
            _parents = new GraphNode[inputs.length];
            for ( int i = 0; i < inputs.length; i++ ) {
                _parents[i] = inputs[i].find(GraphNode.class);
                if ( _parents[i] == null ) {
                    throw new IllegalStateException("Input tensors of a new graph-node must contain leave graph-nodes!");
                } else _parents[i]._attachChild(this);
            }
        }
        if ( _nid == -1 ) {
            long nid = 1;
            if ( _parents != null ) {
                for ( GraphNode n : _parents )
                    nid *= n.getPayload().hashCode(); //payload might be 0! Why? -> garbage collected!
            }
            if ( _function != null ) nid += _function.hashCode();
            _nid = nid;
        }
        /* Returning if the above cannot form an AutoDiff computation graph! : */
        if ( inputs == null || !function.isFlat() ) return; // Leave nodes have!
        for ( Tsr t : inputs ) if ( t.equals(output) ) return; // Output must be a unique tensor for AD!

        if ( this.usesAD() && function.isFlat() ) {
            /* Preparing for back propagation: */
            if ( this.usesForwardAD() ) {
                for ( int i = 0; i < inputs.length; i++ ) {
                    GraphNode srcNode = inputs[i].find( GraphNode.class );
                    if ( srcNode.usesAD() ) {
                        if (
                                srcNode.size() == 0 && this.size() == 0
                                    ||// Sources created by for example dot/mm or x-mul are reverse-mode cases!
                                !srcNode.isLeave() && !srcNode._allows_forward//Was : src_node.function().type().allowsForward( inputs )
                        ) {
                            this.put(
                                    srcNode,
                                    call.getADAgentFrom(
                                            function,
                                            //function.getADAgent(
                                            null,
                                            new ExecutionCall<>(
                                                    call.getDevice(),
                                                    call.getTensors(),
                                                    i,
                                                    call.getJ(),
                                                    call.getType()
                                            ),
                                            true
                                    )
                            );
                        } else {
                            /*  Chain rule (forward) for every derivative w.r.t. leaves (reverseAD or user leaves): */
                            int finalI = i;
                            srcNode.forEach(
                                function.derive( inputs, i ),
                                ( node, targetDerivative ) -> {
                                    if ( this.has( node ) ) {
                                        Tsr derivative = ADD.call(new Tsr[]{targetDerivative, (Tsr) this.get(node)});
                                        this.put(
                                                node,
                                                call.getADAgentFrom(
                                                        function,
                                                //function.getADAgent(
                                                        derivative,
                                                        new ExecutionCall<>(
                                                                call.getDevice(),
                                                                call.getTensors(),
                                                                finalI,
                                                                call.getJ(),
                                                                call.getType()
                                                        ),
                                                        true
                                                )
                                        );
                                    } else {
                                        this.put(
                                                node,
                                                call.getADAgentFrom(
                                                        function,
                                                        //function.getADAgent(
                                                        targetDerivative,
                                                        new ExecutionCall<>(
                                                                call.getDevice(),
                                                                call.getTensors(),
                                                                finalI,
                                                                call.getJ(),
                                                                call.getType()
                                                        ),
                                                        true
                                                )
                                        );
                                    }
                                    // TODO: flag within src tsrs that grant that the tensor
                                    // has been created by function constructor!
                                }
                            );
                        }
                    }
                }
            } else if ( this.usesReverseAD() ) {
                for ( int i = 0; i < inputs.length; i++ ) {
                    GraphNode srcNode = inputs[i].find(GraphNode.class);
                    if ( srcNode.usesAD() || inputs[i].rqsGradient() ) {
                        this.put(
                                srcNode,
                                call.getADAgentFrom(
                                        function,
                                        //function.getADAgent(
                                        null,
                                        new ExecutionCall<Device>(
                                                call.getDevice(),
                                                call.getTensors(),
                                                i,
                                                call.getJ(),
                                                call.getType()
                                        ),
                                        false
                                )
                        );
                    }
                }
            }
        }
    }

    /**
     * Evaluate auto-grad/auto-differentiation mode:
     * A positive value means that the AD-procedure will be forward mode AD,
     * whereas a negative value is backward mode AD.
     * If the resulting mode equals 0 then this means that
     * no auto differentiation is needed.
     *
     * @param call The call containing inputs for the function which created the payload tensor of this GraphNode.
     * @param function The function which produced the payload tensor of this GraphNode.
     * @return int The mode of this GraphNode!
     */
    private int _modeOf( ExecutionCall<Device> call, Function function )
    {
        Tsr[] inputs = call.getTensors();
        int result_mode = 0;
        int[] modes = new int[inputs.length];
        int input_mode = 0;
        for ( int i = 0; i < inputs.length; i++ ) {
            GraphNode node = inputs[i].find( GraphNode.class ); // Not null checked in constructor!
            modes[i] = ( inputs[i].rqsGradient() ) ? 1 : node.mode();
            input_mode += ( modes[i] != 0) ? 1 : 0;
        }
        _allows_forward = call.allowsForward();
        if ( input_mode == 1 && _allows_forward ) { // Convolution and reshaping prohibit forward AutoDiff
            for ( int i = 0; i < inputs.length; i++ ) {
                result_mode += ( modes[i] == 0 ) ? 0 : ( modes[i] < 0 ) ? 1 : modes[i] + 1;
            }
        } else { // Reverse mode auto-differentiation :
            result_mode = -input_mode;
        }
        result_mode = ("<>".replace(function.type().identifier(), "").equals("<>")) ? result_mode : 0;
        return result_mode;
    }

    /**
     * This short method simply migrates the error to the device of
     * the payload tensor and possibly also applies the error to
     * the payload if its 'requires gradient' flag is set to true.
     *
     * @param e This is an error value passed to this method ba a backward traversal.
     */
    private void _migrateAndOrApplyError( Tsr e, Consumer<Tsr> also ) {
        Tsr payload = getPayload();
        if ( payload == null ) return; // Garbage collected!
        if ( payload.isOutsourced() ) payload.device().add(e);
        if ( payload.rqsGradient() ) payload.addToGradient(e);
        if ( also!=null ) also.accept(payload);
    }


    /**
     * This method is the entry-point for the back-propagation process.
     * It sets up a key/value map which stores nodes and their intermediate error accumulations.
     * Accumulations occurs inside the private '_backward' method which traverses the computation graph
     * recursively, halts when errors can be accumulated, adds a PendingError and returns to the method below!
     * Here all the nodes and error values will then be carried (propagated) to the gradients!
     *
     * @param error The current error which is created by multiplying it with current size and traversing it.
     */
    public void backward( Tsr error ) {
        Set<GraphNode> pendingNodes = new HashSet<>();
        _backward( error, pendingNodes, false );// Entry-point to private recursive back-propagation!
        if ( Neureka.instance().settings().autograd().isRetainingPendingErrorForJITProp() ) {
            pendingNodes.forEach( n -> n._carryPendingBackPropToGradients( pendingNodes ) );
        } else {
            pendingNodes.forEach( n -> {
                if ( !n._pending_error.isFullyAccumulated() )
                    throw new IllegalStateException("Pending error has not received expected accumulation.");
                n.backward( n._pending_error.getAccumulatedError() );//Continue back-propagation recursively!
            });
        }
        _deleteDerivativesRecursively();//Cleanup after back-propagation!
    }

    /**
     * This method traverses the computation graph and applies errors to gradients.
     * Errors might be accumulated temporarily or possibly longer for 'Just In Time propagation'.
     * JITProp is enabled in the global Neureka class.
     * It will traverse the path between a pending error and a tensor (rqsGradient==true)
     * containing the JITProp component which is triggered as soon as new gradients are needed or requested (applied).
     * This traverse however does not occur through the method below.
     * Instead the 'backwardJIT' method is called by the JITProp component if present.
     * Intermediate error accumulations are stored in the '_pending_error' variable.
     * The method halts when an error can be accumulated and returns.
     * This graph node however is not forgotten but being noted in the 'pendingNodes' Set.
     *
     * @param error A tensor which traverses the computation graph according to the rules of reverse mode AutoDiff.
     */
    private void _backward( Tsr error, Set<GraphNode> pendingNodes, boolean allowPendingError )
    {
        _migrateAndOrApplyError( error, null );
        if ( this.usesAD() ) {
            /* Checking JIT-Prop conditions and create Pending error if possible */
            if ( allowPendingError && !this.isLeave() ) {//==> We are NOT inside a 'Just-In-Time-Backprop' process (new pending error can be created)
                int numOfADPaths = _numberOfReverseModeADChildren();// Multiple children triggers creation of a pending error
                if ( numOfADPaths > 1 ) {
                    if ( _pending_error == null ) {
                        _pending_error = new PendingError( error, numOfADPaths - 1 );
                        pendingNodes.add( this );
                    } else _pending_error.accumulate( error );
                    return;
                    /* Backprop will be continued later! This node is being remembered in 'PendingError'
                       NOTE: Multiple AutoDiff paths leading to one node in history will be accumulated first! (performance)
                             This optimization is a light version of JITProp. JITProp builds on this!
                    */
                }
            }
            // Using forward-AutoDiff size for reverse-mode AutoDiff!
            if ( this.usesForwardAD() ) this.forEachForward( error, ( t, d ) -> t._backward(d, pendingNodes, true) );
            else if ( this.usesReverseAD() ) this.forEachBackward( error, ( t, e ) -> t._backward(e, pendingNodes, true) );
            // Standard reverse mode-AutoDiff!
        }
    }

    /**
     * This method is called only if JIT-propagation is enabled.
     * It carries pending errors to the tensors requiring gradients which will
     * later on be processed just in time.
     * The path is being marked with '_relies_on_JITProp' so that intermediate size will
     * not be deleted.
     *
     * @param pendingBackProp
     */
    private void _carryPendingBackPropToGradients( Set<GraphNode> pendingBackProp ) {
        _relies_on_JIPProp = true; //:=> Shall be traversed at a later point in time...
        this.forEachTarget( t -> t._carryPendingBackPropToGradients( pendingBackProp ) );
        if ( this.isLeave() && getPayload().rqsGradient() ) {
            JITProp jit = getPayload().find(JITProp.class);
            if ( jit == null ) jit = new JITProp( pendingBackProp );
            else jit.addPending( pendingBackProp );
            getPayload().add( jit );
        }
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
    public void backwardJIT(Tsr error) {
        _backwardJIT( error, this );
        _deleteDerivativesRecursively();// Cleanup after back-propagation!
    }

    private void _backwardJIT(Tsr error, GraphNode source) {
        _relies_on_JIPProp = false; // JITProp is currently being handled in this method. Afterwards it is not relying on it anymore!
        _migrateAndOrApplyError( error, payload -> {
            JITProp jit = payload.find( JITProp.class );//Get JIT-Prop node.
            if ( jit != null ) {
                jit.noteFinished( source );//note pending errors and store them as 'done'
                if ( jit.isDone() ) payload.remove( JITProp.class );
            }
        });
        if ( _pending_error != null && source != this ) {
            _pending_error.accumulate( error );
            /*
              A pending error has been found, so this means that this node
              is referenced by one or more JIT-Prop components.
              If among these components is the one that issued this very
              traverse we are in at this moment, then this pending error at this node will later on
              be continued to be propagated.
              Otherwise it makes sense to accumulate errors further and wait for JIT-Prop traversing!
             */
            return;// This node will continue its propagation via a JIT-Prop component later!
        }
        if ( this.usesAD() ) {
            // Using forward-AutoDiff size for reverse-mode AutoDiff!
            if ( this.usesForwardAD() ) this.forEachForward( error, ( t, e ) -> t._backwardJIT( e, source ) );
            else if ( this.usesReverseAD() ) this.forEachBackward( error, ( t, e ) -> t._backwardJIT( e, source ) );
            // Standard reverse mode-AutoDiff!
        }
    }

    /**
     * This method is called after the backward call has been executed fully.
     * Derivatives are no longer used and will therefore be deleted when possible.
     * Deletion is forbidden if this node is flagged
     * as JITProp job. This means that the node is on the path between gradients
     * and pending error objects.
     * Only if JITProp is enabled (Neureka.instance().settings().AutoDiff()...) this flag will
     * deviate from its default state, namely: true!
     */
    private void _deleteDerivativesRecursively() {
        if ( !Neureka.instance().settings().debug().isKeepingDerivativeTargetPayloads() ) {//<=- This flag is almost always false. (Used for testing)
            if ( !this.reliesOnJustInTimeProp() ) _targets_derivatives = null;
            if ( !this.isGraphLeave() ) forEachTarget( GraphNode::_deleteDerivativesRecursively );
        }
    }

    /**
     * Counts how many child nodes will later on provide error values for back-propagation!
     *
     * @return The number of child nodes using reverse-mode auto-differentiation.
     */
    private int _numberOfReverseModeADChildren() {
        int count = 0;
        if ( _children != null ) {
            for ( WeakReference<GraphNode> weak : _children ) {
                if ( weak != null && weak.get() != null ) {
                    GraphNode child = weak.get(); //TODO: make test which asserts that Detached Function does not trigger this!
                    if ( child!=null && child.usesReverseAD() ) count++;
                }
            }
        }
        return count;
    }

    /**
     * @return int
     */
    public int mode() {
        return _mode;
    }

    /**
     * @return Function
     */
    public Function function() {
        return _function;
    }

    /**
     * @param target nodes are graph nodes which contain either tensors requiring errors for accumulation and/or more targets.
     * @param agent ADAgent's are used during back-propagation in order to distribute an error throughout the graph.
     */
    public void put(GraphNode target, ADAgent agent) {
        if ( _targets_derivatives == null ) _targets_derivatives = new TreeMap<>((a, b) -> a.hashCode() - b.hashCode());

        _targets_derivatives.put( target, agent );

        Tsr d = agent.derivative();
        if ( d != null && d.has(GraphNode.class) ) d.find( GraphNode.class )._is_used_as_derivative = true;
    }

    /**
     * This method returns what is needed for AD, usually a derivative of AD-Agent.
     *
     * @param target
     * @return Tsr
     */
    public Object get( GraphNode target ) {
        if ( _targets_derivatives == null ) return null;
        return _targets_derivatives.get( target );
    }

    /**
     * This method checks if a given graph node is an AD target of this node.
     * This would mean that this node contains an AD-action for the given GraphNode (target).
     *
     * @param target
     * @return boolean
     */
    public boolean has( GraphNode target ) {
        if ( _targets_derivatives == null ) return false;
        return _targets_derivatives.containsKey( target );
    }

    /**
     * This is the number of AD-actions stored inside this node.
     * It can be interpreted as the 'number of AD paths'.
     *
     * @return int
     */
    public int size() {
        return ( _targets_derivatives != null ) ? this._targets_derivatives.size() : 0;
    }

    /**
     * @param action
     */
    public void forEachDerivative(BiConsumer<GraphNode, ADAgent> action) {
        if ( _targets_derivatives == null ) return;
        _targets_derivatives.forEach( action );
    }

    /**
     * @param action A lambda action providing derivative and target node as parameter.
     */
    public void forEachBackward( Tsr error, BiConsumer<GraphNode, Tsr> action ) {
        if ( _targets_derivatives == null ) return;
        _targets_derivatives.forEach( ( t, o ) -> {
            if ( !o.isForward() ) action.accept( t, o.backward( t, error ) );
        });
    }

    /**
     * @param action
     */
    public void forEachForward( Tsr error, BiConsumer<GraphNode, Tsr> action ) {
        if ( _targets_derivatives == null ) return;
        _targets_derivatives.forEach( ( t, o ) -> {
            //if ( o.isForward() ) action.accept( t, MUL.call( new Tsr[]{error, o.derivative()} ) );
            if ( o.isForward() ) action.accept( t, o.forward(t, error) );
        });
    }

    /**
     * @param action
     */
    public void forEachTarget(Consumer<GraphNode> action) {
        if ( _targets_derivatives == null ) return;
        _targets_derivatives.forEach( ( t, o ) -> action.accept( t ) );
    }

    /**
     * @param derivative
     * @param action
     */
    public void forEach( Tsr derivative, BiConsumer<GraphNode, Tsr> action ) {
        if ( _targets_derivatives == null ) return;
        _targets_derivatives.forEach( ( t, o ) -> action.accept( t, MUL.call( new Tsr[]{derivative, o.derivative()} ) ) );
    }


    /**
     * @return Checks if this node stores target / AD-action (usually derivatives) pairs.
     */
    public boolean hasDerivatives() {
        return ( _targets_derivatives != null ) && _targets_derivatives.size() > 0;
    }

    /**
     * @return Returns the type of the node as descriptive String in capital letters.
     */
    public String type() {
        String type = "";
        if ( this.isLeave() ) type += "LEAVE";
        else type += "BRANCH";
        if ( getPayload() == null ) type = type + " DELETED";
        else if ( getPayload().rqsGradient() ) type += " RQS GRADIENT";
        return type;
    }

    @Override
    public String toString() {
        return toString("");
    }

    /**
     * @param m Stands for 'mode' and is expected to contain certain letters which are used as settings.
     * @return Returns a String representation of this node.
     */
    public String toString( String m ) {
        if ( m.contains("g") ) {
            return "]> LOCK: " + lock() + " |> GRAPH:\n]\n" + _toString("]    0", true) + "\n]\n]|END|>";
        }
        if ( m.contains("v") ) {
            return "(" + this.type() + "): [NID:" + Long.toHexString(nid()) + "]:<(  "
                    + "f" + ((_function == null) ? "(NONE)" : _function) + " => " + ((getPayload() == null) ? "NULL" : getPayload().toString("cs")) + "  )>";
        } else {
            return "[NID:" + Long.toHexString(nid()) + "]:( " + ((getPayload() == null) ? "NULL" : getPayload().toString("cs")) + " )";
        }

    }

    /**
     * A private recursive method used by its public counterpart ( 'toString(String m)' )
     * in order to build a indented multi-line tree-like
     * String representation of the entire computation graph
     * starting at the node from where this method is called.
     *
     * @param deep The current depth / indentation
     * @param isLast Tells if this is the last parent node of this child.
     * @return A indented multi-line tree-like String representation of the computation graph.
     */
    private String _toString( String deep, boolean isLast ) {
        String delimiter = ((isLast) ? ("    ") : ("|   "));
        String arrow = ((char) 187) + "" + ((_parents != null) ? (String.valueOf(_parents.length)) : "0") + ((char) 187);
        StringBuilder asString = new StringBuilder(deep + arrow + toString("v"));
        deep = deep.substring(0, deep.length() - 1);
        if ( _parents != null ) {
            asString.append("\n").append( deep ).append( (isLast) ? "   \\\n" : "|  \\\n" );
            for ( int i = 0; i < _parents.length; i++ ) {
                boolean last = (i == _parents.length - 1);
                asString.append( ( i != 0 ) ? deep + delimiter + "|\n" : "" );
                asString.append( _parents[i]._toString(deep + delimiter + i, last) ).append("\n");
            }
            asString = new StringBuilder( asString.substring(0, asString.length() - 1) );
        }
        return asString.toString();
    }


}
