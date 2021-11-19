/*
MIT License

Copyright (c) 2019 Gleethos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _____                 _     _   _           _
   / ____|               | |   | \ | |         | |
  | |  __ _ __ __ _ _ __ | |__ |  \| | ___   __| | ___
  | | |_ | '__/ _` | '_ \| '_ \| . ` |/ _ \ / _` |/ _ \
  | |__| | | | (_| | |_) | | | | |\  | (_) | (_| |  __/
   \_____|_|  \__,_| .__/|_| |_|_| \_|\___/ \__,_|\___|
                   | |
                   |_|

    This class defines the nodes which form the computation graph used to track operations performed on tensors,
    or more precisely :
    instances of the 'Tsr' class!

*/

package neureka.autograd;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.common.composition.Component;
import neureka.devices.Device;
import neureka.devices.opencl.utility.WeakTensorReference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.ref.WeakReference;
import java.util.*;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.stream.Collectors;

/**
 *  Instances of the {@link GraphNode} class are components of tensors ({@link Tsr} instances)
 *  which model and record computations / operations between them.
 *  {@link GraphNode}s form a computation graph when operations are applied to tensors.
 *  This graph can then later on be used for traversal by an important algorithm implemented inside
 *  this class, namely: backpropagation.
 *  This algorithm is more generally known as reverse mode auto differentiation.
 *  The parent graph nodes of a given node are the nodes of the tensors
 *  from which the tensor of the current node was formed,
 *  whereas children are the nodes (also) produced by the computation modelled by said current node.
 *  Children are weakly referenced so that abandoned / detached
 *  graph branches (child nodes) can be garbage collected...
 *  ...whereas parents are strongly referenced in order to grant successful traversal.
 */
public class GraphNode<V> implements Component<Tsr<V>>
{
    private static Logger _LOG = LoggerFactory.getLogger(GraphNode.class);

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
     *  done in the corresponding OperationTypeImplementation method
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
     *  This flag records the support evaluation of the backward-AD availability analysis
     *  done in the corresponding OperationTypeImplementation method
     *  for a given ExecutionCall instance.
     *
     *  The difference between this flag and the "usesBackwardAD()" truth value
     *  is that the latter one can be false while the prior is true!
     *  ( However the reverse is not possible! )
     *  The reason is as follows:
     *  If for example a GraphNode has only one parent node which require auto-differentiation,
     *  then said node will most likely perform backward-AD even though it might very well
     *  be possible given an ExecutionCall whose state allows for such...
     */
    private boolean _allows_backward;

    /**
     * This flag is used for a performance optimization feature namely 'Just In Time Propagation'.
     * This feature accumulates errors and continues propagation
     * as soon as they are needed. (At the end of 'backward()' or when the tensor is used again).
     * If the flag  {@link Neureka.Settings.AutoGrad#isRetainingPendingErrorForJITProp()}  is set to true
     * then error values will accumulate whenever it makes sense.
     * This technique however uses more memory but will
     * improve performance for some networks substantially.
     * <p>
     * All nodes between a Pending-Error and those requiring gradients will
     * be marked with '_relies_on_JIPProp=true'!
     */
    private boolean _reliesOnJustInTimeProp = false;

    /**
     * Used by the Just-In-Time back-prop component.
     */
    private PendingError<V> _pendingError = null;

    /**
     * The chain-rule states that the derivative of f(x) = h(g(x)) with respect to x is: g'(x) * h'(g(x))
     * An example would be:
     * f(x) = ((x*y)*z)
     * f'(x) = (1*y) * (1*z) = z*y
     * The values z,y or z*y must not be deleted as they are needed for back-propagation!
     */
    private boolean _isUsedAsDerivative = false;

    /**
     * Recorded Function which produced this {@link GraphNode}.
     */
    private Function _function;

    /**
     * The GraphNodes of the input tensors. ('Parents' of the tensor of this node)
     * These are always the GraphNodes of the tensors from which the tensor payload of this
     * GraphNode has been formed.
     */
    private GraphNode<V>[] _parents;

    /**
     * This is the tensor owning this GraphNode component.
     * It is referenced weakly because it might not be needed anymore (Not referenced inside AD-Agent for example)
     * and can therefore be garbage collected.
     */
    private WeakReference<Tsr<V>> _payload;

    /**
     *  This variable holds a copy of the version of the payload tensor
     *  recorded when this GraphNode instance is instantiated.
     *  It must be treated as final and should never be modified.
     *  However it can be read freely in order to
     *  check that the version of the payload hasn't changed.
     */
    private int _payloadReferenceVersion = -1;

    /**
     * Keys are {@link GraphNode} targets and values are {@link ADAgent}s which most of the times
     * simply store derivatives as well as operation specific implementations
     * to propagate these derivatives with respect to mentioned  {@link GraphNode} targets.  <br>
     * Note: values can be null if the recorded function is of type 'reshape'!
     * Why? => because reshape operation does not need variables for _backward pass!
     */
    private TreeMap<GraphNode<V>, List<ADAgent>> _targetsToAgents;

    /**
     * "Lock object" for graph identity. (result caching)
     * Unique object which locks the payload to the current computation graph.
     */
    private GraphLock _lock;

    /**
     *  The children are {@link GraphNode} instances which represent computations
     *  involving the payload of this very {@link GraphNode} instance.
     */
    private List<WeakReference<GraphNode<V>>> _children;

    /**
     * long Node-ID (Used for caching to avoid redundant computation within one computation graph)
     */
    private long _nodeID = -1;


    /**
     * @param function        Is the function that lead to the creation of this node.
     * @param context         Can be either an array of tensors or a new lock (for leave node or fresh function locking)
     * @param payloadSupplier Provides the payload of this node.
     */
    public GraphNode( Function function, Object context, Supplier<Tsr<V>> payloadSupplier )
    {
        if ( function == null )
            throw new IllegalArgumentException(
                    "Passed constructor argument of type Function must not be null!"
            );
        if ( context instanceof GraphLock ) // Note function always null in this case:
            _construct( payloadSupplier.get(), function, null, (GraphLock) context );
        else if ( context instanceof ExecutionCall ) {
            ExecutionCall<Device<?>> call = (ExecutionCall<Device<?>>) context;
            Tsr<?>[] inputs = call.getTensors();
            /* Applying JITProp and gradients */
            Neureka.Settings.AutoGrad adSetting = Neureka.get().settings().autograd();
            if ( adSetting.isApplyingGradientWhenTensorIsUsed() ) {
                for ( Tsr<?> t : inputs ) {
                    if ( !adSetting.isApplyingGradientWhenRequested() || t.gradientApplyRequested() ) {
                        t.applyGradient(); // activates JITProp if present and removes it...
                        t.setGradientApplyRequested( false );
                    }
                }
            }
            GraphLock foundLock = null;
            for ( int i = 0; i < inputs.length; i++ ) {
                GraphNode<V> child = inputs[ i ].get( GraphNode.class );
                if ( child == null ) throw new IllegalStateException(
                        "Input tensor at index '" + i + "' did not return a GraphNode instance." +
                                "Input tensors of a new GraphNode must be part of the computation graph!"
                );
                if ( foundLock == null ) foundLock = child.getLock();
                if ( foundLock != child.getLock() ) {
                    throw new IllegalStateException(
                            "GraphNode instances found in input tensors do not share the same GraphLock instance.\n" +
                                    "The given input tensors of a new node must be part of the same locked computation graph!"
                    );
                }
                if ( function.getOperation().isInline() && child.usesAD() ) {
                    throw new IllegalStateException(
                            "Trying to apply inline operation '" + function.getOperation().getFunction() + "'\n" +
                            "on active autograd computation graph in non detached function.\n" +
                            "Please use detached functions instead! ( 'Function.create(\"" + function.getOperation().getFunction() + "(...)\", false)' )\n"
                    );
                }
            }
            _construct( payloadSupplier.get(), function, call, inputs[ 0 ].get( GraphNode.class ).getLock() );
        }
        else
            throw new IllegalArgumentException(
                    "The passed context object for the GraphNode constructor is of type '" + context.getClass().getName() + "'.\n" +
                            "A given context must either be a GraphLock instance or an ExecutionCall."
            );
    }

    /**
     * This method handles the construction of a GraphNode instance in more detail.
     *
     * @param output The container for the result of the execution, in a sense, its the output of this node / its payload!
     * @param function The function which produced this {@link GraphNode} instance.
     * @param call The {@link ExecutionCall} instance containing context information for the current execution.
     * @param lock An object whose identity will be used to reserve the {@link Tsr} instances of the current {@link ExecutionCall}.
     */
    private void _construct( Tsr<V> output, Function function, ExecutionCall<? extends Device<?>> call, GraphLock lock )
    {
        Tsr<Object>[] inputs = ( call == null ) ? null : (Tsr<Object>[]) call.getTensors();
        if ( output == null ) throw new NullPointerException( "The supplied payload Tsr must no be null!" );
        _payloadReferenceVersion = output.getVersion();
        if ( !function.isDoingAD() ) return; // Only functions with AutoDiff enabled create computation graph!
        _lock = lock;
        _setPayload( output );
        output.set( this );
        if ( inputs == null ) {
            _mode = ( output.rqsGradient() ) ? 1 : 0;
            _function = null;
            _parents = null;
        } else {
            _mode = _modeOf( call );
            _function = function;
            _parents = new GraphNode[ inputs.length ];
            for ( int i = 0; i < inputs.length; i++ ) {
                _parents[ i ] = inputs[ i ].get( GraphNode.class );
                if ( _parents[ i ] == null ) {
                    throw new IllegalStateException(
                            "Input tensors of a new graph-node must contain leave graph-nodes!"
                    );
                } else _parents[ i ]._attachChild(this);
            }
        }
        if ( _nodeID == -1 ) {
            long nid = 1;
            if ( _parents != null ) {
                for ( GraphNode<V> n : _parents )
                    nid *= n.getPayload().hashCode(); //payload might be 0! Why? -> garbage collected!
            }
            if ( _function != null ) nid += _function.hashCode();
            _nodeID = nid;
        }
        /* Returning if the above cannot form an AutoDiff computation graph! : */
        if ( inputs == null || !function.isFlat() ) return; // Leave nodes have!
        for ( Tsr<Object> t : inputs ) if ( t.equals(output) ) return; // Output must be a unique tensor for AD!

        if ( this.usesAD() && function.isFlat() ) {
            /* Preparing for back propagation: */
            if ( this.usesForwardAD() )
            {
                for ( int i = 0; i < inputs.length; i++ ) {
                    GraphNode<V> srcNode = inputs[ i ].get( GraphNode.class );
                    if ( srcNode.usesAD() ) {
                        if (
                                srcNode.size() == 0 && this.size() == 0
                                    ||// Sources created by for example dot/mm or x-mul are reverse-mode cases!
                                !srcNode.isLeave() && !srcNode._allows_forward
                        ) {
                            this.put(
                                    srcNode,
                                    call.getADAgentFrom(
                                            function,
                                            ExecutionCall.of(call.getTensors())
                                                            .andArgs(
                                                                    Arg.DerivIdx.of(i),
                                                                    Arg.VarIdx.of(call.getValOf(Arg.VarIdx.class))
                                                            )
                                                            .running(call.getOperation())
                                                            .on(call.getDevice()),
                                            true
                                    )
                            );
                        } else {
                            /*  Chain rule (forward) for every derivative w.r.t. leaves (reverseAD or user leaves): */
                            int finalI = i;
                            Tsr<V> localDerivative = (Tsr<V>) function.derive( inputs, i );
                            srcNode.forEachTargetAgentPair(
                                ( targetNode, localAgent ) ->
                                {
                                    // The agent multiplies the local derivative with its stored partial derivative...
                                    Tsr<?> targetDerivative = localAgent.forward( this, localDerivative );
                                    // ...this is now the new partial derivative with respect to the target node!
                                    this.put(
                                            targetNode,
                                            call.getADAgentFrom(
                                                    function,
                                                    ExecutionCall.of(call.getTensors())
                                                                    .andArgs(
                                                                            Arg.VarIdx.of(call.getValOf(Arg.VarIdx.class)),
                                                                            Arg.DerivIdx.of(finalI),
                                                                            Arg.Derivative.of(targetDerivative)
                                                                    )
                                                                    .running(call.getOperation())
                                                                    .on(call.getDevice()),
                                                    true
                                            )
                                    );
                                    // TODO: flag within src Tsr<ValType>s that grant that the tensor
                                    // has been created by function constructor!
                                }
                            );
                        }
                    }
                }
            } else if ( this.usesReverseAD() ) {
                for ( int i = 0; i < inputs.length; i++ ) {
                    GraphNode<V> srcNode = inputs[ i ].get( GraphNode.class );
                    if ( srcNode.usesAD() || inputs[ i ].rqsGradient() ) {
                        this.put(
                                srcNode,
                                call.getADAgentFrom(
                                        function,
                                        ExecutionCall.of(call.getTensors())
                                                        .andArgs(
                                                                Arg.DerivIdx.of(i),
                                                                Arg.VarIdx.of(call.getValOf(Arg.VarIdx.class))
                                                        )
                                                        .running(call.getOperation())
                                                        .on(call.getDevice()),
                                        false
                                )
                        );
                    }
                }
            }
        }
    }

    /**
     *  Evaluate auto-grad/auto-differentiation mode:
     *  A positive value means that the AD-procedure will be forward mode AD,
     *  whereas a negative value is backward mode AD.
     *  If the resulting mode equals 0 then this means that no auto differentiation is needed.
     *  This class tries to optimize the calculation of partial derivatives by forward propagating them
     *  for as long as only a single input for every computation graph node requires gradients
     *  and they all are differentiable!
     *
     *
     * @param call The call containing inputs for the function which created the payload tensor of this GraphNode.
     * @return int The mode of this GraphNode! ( m<0 : backward-AD, m>0 : forward-AD, m=0 : no-AD )
     */
    private int _modeOf( ExecutionCall<? extends Device<?>> call )
    {
        Tsr<V>[] inputs = (Tsr<V>[]) call.getTensors();
        int resultMode = 0;
        int[] modes = new int[ inputs.length ];
        int inputMode = 0;
        for ( int i = 0; i < inputs.length; i++ ) {
            GraphNode<V> node = inputs[ i ].get( GraphNode.class ); // Not null checked in constructor!
            modes[ i ] = ( inputs[ i ].rqsGradient() ) ? 1 : node.getMode();
            inputMode += ( modes[ i ] != 0) ? 1 : 0;
        }
        _allows_forward = call.allowsForward();
        _allows_backward = call.allowsBackward();
        if ( inputMode == 1 && _allows_forward ) { // Convolution and reshaping prohibit forward AutoDiff
            for ( int i = 0; i < inputs.length; i++ ) {
                resultMode += 
                        ( modes[ i ] == 0 ) 
                                ? 0 
                                : ( modes[ i ] < 0 ) ? 1 : modes[ i ] + 1;
            }
        } // Reverse mode auto-differentiation :
        else if ( _allows_backward ) resultMode = -inputMode;

        return resultMode;
    }

    /**
     * This short method simply migrates the error to the device of
     * the payload tensor and possibly also applies the error to
     * the payload if its 'requires gradient' flag is set to true.
     *
     * @param e This is an error value passed to this method ba a backward traversal.
     */
    private void _migrateAndOrApplyError( Tsr<V> e, Consumer<Tsr<V>> also ) {
        Tsr<V> payload = getPayload();
        if ( payload == null ) return; // Garbage collected!
        try {
            if ( payload.isOutsourced() ) payload.getDevice().store( e );
        } catch ( Exception exception ) {
            if ( payload.isUndefined() ) {
                throw new IllegalStateException(
                        "An undefined payload tensor has been detected inside the computation graph!\n" +
                        "This is most likely due to an error occurring during tensor identity transfer (Also see AbstractComponentOwner).\n" +
                        "One type of constructor in the 'Tsr' class enables passing a String expression for execution, " +
                        "whose resulting tensor needs to be merged into the newly created one..."
                );
            } else exception.printStackTrace();
        }
        if ( payload.rqsGradient() ) payload.addToGradient( e );
        if ( also != null ) also.accept( payload );
    }


    /**
     * This gradient node is involved in auto-differentiation.
     *
     * @return boolean
     */
    public boolean usesAD() { return ( _mode != 0 ); }

    /**
     * This node propagates forward.
     *
     * @return boolean
     */
    public boolean usesForwardAD() { return ( _mode > 0 ); }

    /**
     * This node propagates _backward.
     *
     * @return boolean
     */
    public boolean usesReverseAD() { return ( _mode < 0 ); }


    /**
     * Some nodes are not cacheable! Namely: leave tensors! They are not results of
     * any function operation.
     *
     * @return boolean
     */
    public boolean isCacheable() { return ( this.getNodeID() != 1 ); }

    /**
     * @param newLock The new lock of this GraphNode.
     */
    public synchronized void obtainLocking( GraphLock newLock ) {
        _lock = newLock;
    }

    /**
     * This node (and the corresponding tensor) was not created by a function! (it's a leave tensor)
     *
     * @return boolean
     */
    public boolean isLeave() { return _parents == null && _function == null; }

    public boolean isGraphLeave() {
        if ( this.isLeave() ) return true;
        for ( GraphNode<V> p : _parents ) {
            if ( p.getLock() != this.getLock() ) return true;
        }
        return false;
    }

    /**
     * @return if the tensor to which this graph node is attached has been deleted!
     */
    public boolean isVirtual() { return getPayload() == null; }

    /**
     * @param newChild which references it's input namely the parent (this) has...
     */
    private synchronized void _attachChild( GraphNode<V> newChild ) {
        if ( _children == null ) _children = new ArrayList<>();
        WeakReference<GraphNode<V>> ref = new WeakTensorReference<>( newChild, null );
        _children.add( ref );
    }

    /**
     *  The value of a graph node is the tensor to which it belongs (is a component of).  <br><br>
     *
     *  Warning: This method might return null because
     *           the payload is weakly referenced!
     *           Meaning that it might get garbage collected.
     *
     * @return The tensor payload of this graph-node.
     */
    public Tsr<V> getPayload() { return ( _payload == null ? null : _payload.get() ); }

    /**
     * @param p The {@link Tsr} ought to be set as payload / result of the
     *          computation modelled by this {@link GraphNode} instance.
     */
    private void _setPayload( Tsr<V> p ) {
        if ( p == null ) _payload = null;
        else {
            assert !p.isUndefined();
            _payload = new WeakReference<>( p );
            p.getDevice().cleaning( p, () -> {
                if ( this.getPayload() == null ) {
                    boolean allChildrenUseForwardAD = true;
                    if ( _children != null ) {
                        for ( WeakReference<GraphNode<V>> childRef : _children ) {
                            GraphNode<V> childNode = childRef.get();
                            if ( childNode != null && childNode.usesReverseAD() ) allChildrenUseForwardAD = false;
                        }
                    }
                    if ( allChildrenUseForwardAD ) _targetsToAgents = null;
                }
            });
        }
    }

    @Override
    public boolean update( OwnerChangeRequest<Tsr<V>> changeRequest ) {
        _setPayload( changeRequest.getNewOwner() );
        changeRequest.executeChange();
        return true;
    }

    /**
     * This method is called by the JITProp component.
     * A pending should only ever be retrieved from a GraphNode once because
     * afterwards the accumulated error is about to be backpropagated.
     * Therefore this method nulls the reference when returning the PendingError instance.
     * @return Returns an instance of the PendingError class containing a error accumulation.
     */
    public PendingError<V> getAndRemovePendingError() {
        PendingError<V> pe = _pendingError;
        _pendingError = null;
        return pe;
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
    public void backward( Tsr<V> error ) {
        Set<GraphNode<V>> pendingNodes = new HashSet<>();
        _backward( error, pendingNodes, false ); // Entry-point to private recursive back-propagation!
        if ( Neureka.get().settings().autograd().isRetainingPendingErrorForJITProp() ) {
            pendingNodes.forEach( n -> n._carryPendingBackPropToGradients( pendingNodes ) );
        } else {
            pendingNodes.forEach( n -> {
                if ( !n._pendingError.isFullyAccumulated() ) {
                    _LOG.error(
                            "Pending error in graph node '"+n.toString()+"' has not received expected accumulation. " +
                            ""
                    );
                }
                n.backward( n._pendingError.getAccumulatedError() ); // Continue back-propagation recursively!
            });
        }
        _deleteDerivativesRecursively(); // Cleanup after back-propagation!
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
    private void _backward(Tsr<V> error, Set<GraphNode<V>> pendingNodes, boolean allowPendingError )
    {
        _migrateAndOrApplyError( error, null );
        if ( this.usesAD() ) {
            /* Checking JIT-Prop conditions and create Pending error if possible */
            if ( allowPendingError && !this.isLeave() ) {//==> We are NOT inside a 'Just-In-Time-Backprop' process (new pending error can be created)
                int numOfADPaths = _numberOfReverseModeADChildren();// Multiple children triggers creation of a pending error
                if ( numOfADPaths > 1 ) {
                    if ( _pendingError == null ) {
                        _pendingError = new PendingError<>( error, numOfADPaths - 1 );
                        pendingNodes.add( this );
                    } else _pendingError.accumulate( error );
                    return;
                    /* Backprop will be continued later! This node is being remembered in 'PendingError'
                       NOTE: Multiple AutoDiff paths leading to one node in history will be accumulated first! (performance)
                             This optimization is a light version of JITProp. JITProp builds on this!
                    */
                }
            }
            // The following call ADAgents for reverse-mode AutoDiff!
            this.forEachBackward( error, ( t, e ) -> t._backward( e, pendingNodes, true ) );
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
    private void _carryPendingBackPropToGradients( Set<GraphNode<V>> pendingBackProp ) {
        _reliesOnJustInTimeProp = true; //:=> Shall be traversed at a later point in time...
        this.forEachTarget( t -> t._carryPendingBackPropToGradients( pendingBackProp ) );
        if ( this.isLeave() && getPayload().rqsGradient() ) {
            JITProp<V> jit = getPayload().get( JITProp.class );
            if ( jit == null ) jit = new JITProp<>( pendingBackProp );
            else jit.addPending( pendingBackProp );
            getPayload().set( jit );
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
     * @param error The error which ought to be back-propagated just-in-time.
     */
    public void backwardJIT( Tsr<V> error ) {
        _backwardJIT( error, this );
        _deleteDerivativesRecursively();// Cleanup after back-propagation!
    }

    private void _backwardJIT(Tsr<V> error, GraphNode<V> source ) {
        _reliesOnJustInTimeProp = false; // JITProp is currently being handled in this method. Afterwards it is not relying on it anymore!
        _migrateAndOrApplyError( error, payload -> {
            JITProp<V> jit = payload.get( JITProp.class );//Get JIT-Prop node.
            if ( jit != null ) {
                jit.noteFinished( source );//note pending errors and store them as 'done'
                if ( jit.isDone() ) payload.remove( JITProp.class );
            }
        });
        if ( _pendingError != null && source != this ) {
            _pendingError.accumulate( error );
            /*
              A pending error has been found, so this means that this node
              is referenced by one or more JIT-Prop components.
              If among these components is the one that issued this very
              traverse we are in at this moment, then this pending error at this node will later on
              be continued to be propagated.
              Otherwise it makes sense to accumulate errors further and wait for JIT-Prop traversing!
             */
            return; // This node will continue its propagation via a JIT-Prop component later!
        }
        if ( this.usesAD() ) {
            // The following call ADAgents for reverse-mode AutoDiff!
            this.forEachBackward( error, ( t, e ) -> t._backwardJIT( e, source ) );
            // JITProp reverse mode-AutoDiff!
        }
    }

    /**
     * This method is called after the backward call has been executed fully.
     * Derivatives are no longer used and will therefore be deleted when possible.
     * Deletion is forbidden if this node is flagged
     * as JITProp job. This means that the node is on the path between gradients
     * and pending error objects.
     * Only if JITProp is enabled (Neureka.instance().settings().autograd()...) this flag will
     * deviate from its default state, namely: true!
     */
    private void _deleteDerivativesRecursively() {
        if ( !Neureka.get().settings().debug().isKeepingDerivativeTargetPayloads() ) { // <=- This flag is almost always false. (Used for testing)
            if ( !this.isReliesOnJustInTimeProp() ) _targetsToAgents = null;
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
            for ( WeakReference<GraphNode<V>> weak : _children ) {
                if ( weak != null && weak.get() != null ) {
                    GraphNode<V> child = weak.get(); // TODO: make test which asserts that Detached Function does not trigger this!
                    if ( child != null && child.usesReverseAD() ) count++;
                }
            }
        }
        return count;
    }


    /**
     * @param target nodes are graph nodes which contain either tensors requiring errors for accumulation and/or more targets.
     * @param agent ADAgent's are used during back-propagation in order to distribute an error throughout the graph.
     */
    public void put( GraphNode<V> target, ADAgent agent ) {
        if ( _targetsToAgents == null ) _targetsToAgents = new TreeMap<>((a, b) -> a.hashCode() - b.hashCode());

        if ( _targetsToAgents.containsKey( target ) )
            _targetsToAgents.get( target ).add( agent );
        else
            _targetsToAgents.put( target, new ArrayList<>( Arrays.asList( agent ) ) );

        Tsr<?> d = agent.derivative();
        if ( d != null && d.has( GraphNode.class ) ) d.get( GraphNode.class )._isUsedAsDerivative = true;
    }

    /**
     * This method returns what is needed for AD, usually a derivative of AD-Agent.
     *
     * @param target The targeted derivation graph node reference.
     * @return Tsr&lt;ValType&gt;
     */
    public List<ADAgent> get( GraphNode<V> target ) {
        if ( _targetsToAgents == null ) return null;
        return _targetsToAgents.get( target );
    }

    /**
     * This method checks if a given graph node is an AD target of this node.
     * This would mean that this node contains an AD-action for the given GraphNode (target).
     *
     * @param target The targeted derivation graph node reference.
     * @return boolean
     */
    public boolean has( GraphNode<V> target ) {
        if ( _targetsToAgents == null ) return false;
        return _targetsToAgents.containsKey( target );
    }

    /**
     * This is the number of AD-actions stored inside this node.
     * It can be interpreted as the 'number of AD paths'.
     *
     * @return int
     */
    public int size() { return _targetsToAgents != null ? this._targetsToAgents.size() : 0; }

    /**
     * @param action The lambda performing an action on all targeted nodes and their agents.
     */
    public void forEachDerivative( BiConsumer<GraphNode<V>, ADAgent> action ) {
        if ( _targetsToAgents == null ) return;
        _targetsToAgents.forEach(
                ( t, agents ) -> agents.forEach( a -> action.accept( t, a ) )
        );
    }

    /**
     * @param action A lambda action providing derivative and target node as parameter.
     */
    public void forEachBackward( Tsr<V> error, BiConsumer<GraphNode<V>, Tsr<V>> action ) {
        if ( _targetsToAgents == null ) return;
        _targetsToAgents.forEach( (t, agents ) -> {
            for ( ADAgent a : agents ) action.accept( t, a.backward( t, error ) );
        });
    }

    /**
     * @param action
     */
    public void forEachTarget( Consumer<GraphNode<V>> action ) {
        if ( _targetsToAgents == null ) return;
        _targetsToAgents.forEach( (t, o ) -> action.accept( t ) );
    }

    /**
     * @param action The action which ought to be applied to each target {@link GraphNode} / {@link ADAgent} pair.
     */
    public void forEachTargetAgentPair( BiConsumer<GraphNode<V>, ADAgent> action ) {
        if ( _targetsToAgents == null ) return;
        _targetsToAgents
                .forEach(
                    ( targetNode, agents ) ->
                        agents.forEach(
                            a -> action.accept( targetNode, a )
                        )
                );
    }


    /**
     * @return Checks if this node stores target / AD-action (usually derivatives) pairs.
     */
    public boolean hasDerivatives() { return _targetsToAgents != null && _targetsToAgents.size() > 0; }

    /**
     *  This is the getter for an important {@link GraphNode} property which
     *  holds the auto-differentiation mode used by this instance to
     *  decide if a given error should be forward propagated
     *  backward propagated or not propagated at all.
     *  If the mode is greater than 0, then this means this {@link GraphNode}
     *  will perform forward propagation. In this case the mode number
     *  is also the cumulative number of forward propagation steps
     *  in the tree of source {@link GraphNode} instances.
     *  If the mode is below 0, then this means this instance will
     *  perform reverse mode differentiation (back-propagation).
     *  The absolute of a negative mode represents the number of
     *  referenced source nodes which have a mode state other than zero.
     *  This means that they directly or indirectly reference
     *  a {@link GraphNode} instance which represents a {@link Tsr} instance
     *  having the {@link Tsr#rqsGradient()} flag set to true!
     *                                                              <br>
     *  Mode state meaning:                                         <br>
     *  ----------------------------------------------------------- <br>
     *  |  mode equals 0  |  no Auto-Differentiation                <br>
     *  ----------------------------------------------------------- <br>
     *  |  mode greater 0  |  forward Auto-Differentiation          <br>
     *  ----------------------------------------------------------- <br>
     *  |  mode lesser 0  |  backward Auto-Differentiation          <br>
     *  ----------------------------------------------------------- <br><br>
     *
     * @return The differentiation mode represented as an integer which encodes 3 distinct states.
     */
    public int getMode() { return _mode; }

    /**
     * This flag is used for a performance optimization feature namely 'Just In Time Propagation'.
     * This feature accumulates errors and continues propagation
     * as soon as they are needed. (At the end of 'backward()' or when the tensor is used again).
     * If the flag {@link Neureka.Settings.AutoGrad#isRetainingPendingErrorForJITProp()} is set to true
     * then error values will accumulate whenever it makes sense.
     * This technique however uses more memory but will
     * improve performance for some networks substantially.
     * <p>
     * All nodes between a Pending-Error and those requiring gradients will
     * be marked with '_relies_on_JIPProp=true'!
     */
    public boolean isReliesOnJustInTimeProp() { return _reliesOnJustInTimeProp; }

    public PendingError<V> getPendingError() { return _pendingError; }

    public boolean isUsedAsDerivative() { return _isUsedAsDerivative; }

    public Function getFunction() { return _function; }

    public GraphNode<V>[] getParents() { return _parents; }

    public int getPayloadReferenceVersion() { return _payloadReferenceVersion; }

    public GraphLock getLock() { return _lock; }

    public List<WeakReference<GraphNode<V>>> getChildren() { return _children; }

    /**
     * @return The long Node-ID (Used for caching to avoid redundant computation within one computation graph)
     */
    public long getNodeID() { return _nodeID; }

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
    public String toString() { return toString( "" ); }

    /**
     * @param m Stands for 'mode' and is expected to contain certain letters which are used as settings.
     * @return Returns a String representation of this node.
     */
    public String toString( String m ) {
        Tsr<?> payload = getPayload();
        if ( m.equals("") ) {
            return this.getClass().getSimpleName()+"@"+Integer.toHexString(this.hashCode())+"[" +
                        "parents=[" + Arrays.stream(_parents).map(GraphNode::getPayload).map(t -> (t==null ? "?" : t.shape().stream().map(Object::toString).collect(Collectors.joining("x")))).collect(Collectors.joining(", ")) + "]," +
                        "function=" + this._function.toString() + "," +
                        "shape=" + (payload != null ? payload.shape().stream().map(Object::toString).collect(Collectors.joining("x")) : "?" ) +
                    "]";
        }
        if ( m.contains( "g" ) ) {
            String flags = m.replace( "g", "" );
            return "]> LOCK: " + getLock() + " |> GRAPH:\n]\n" + _toString( "]    0", true, flags ) + "\n]\n]|END|>";
        }
        String nid = ( m.contains( "n" ) ? "NID:" + Long.toHexString( getNodeID() ) : "NODE" );
        if ( m.contains( "v" ) ) {
            return "(" + this.type() + "): [" + nid + "]:<(  "
                    + "f" +
                    (
                            ( _function == null ) ? "(NONE)" : _function
                    )
                    + " => " + ( (payload == null ) ? "NULL" : payload.toString( "cs" ) ) + "  )>";
        } else
            return
                    "[" + nid + "]:( " + (
                            ( payload == null )
                                    ? "NULL"
                                    : payload.toString("cs")
                    ) + " )";
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
    private String _toString( String deep, boolean isLast, String flags ) {
        String delimiter = ( isLast ? ("    ") : ("|   ") );
        String arrow = ( (char) 187 ) + "" + ( _parents != null ? String.valueOf( _parents.length ) : "0" ) + ( (char) 187 );
        StringBuilder asString = new StringBuilder( deep + arrow + toString( flags ) );
        deep = deep.substring( 0, deep.length() - 1 );
        if ( _parents != null ) {
            asString.append( "\n" ).append( deep ).append( isLast ? "   \\\n" : "|  \\\n" );
            for ( int i = 0; i < _parents.length; i++ ) {
                boolean last = ( i == _parents.length - 1 );
                asString.append( i != 0 ? deep + delimiter + "|\n" : "" );
                asString.append( _parents[ i ]._toString(deep + delimiter + i, last, flags) ).append( "\n" );
            }
            asString = new StringBuilder( asString.substring( 0, asString.length() - 1 ) );
        }
        return asString.toString();
    }

}
