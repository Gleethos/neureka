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
    instances of the 'Tensor' class!

*/

package neureka.autograd;

import neureka.Neureka;
import neureka.Tensor;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Result;
import neureka.common.composition.Component;
import neureka.common.utility.LogUtil;
import neureka.devices.Device;
import neureka.dtype.DataType;
import neureka.math.Function;
import neureka.math.args.Arg;

import java.lang.ref.WeakReference;
import java.util.*;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 *  Instances of the {@link GraphNode} class are components of tensors ({@link Tensor} instances)
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
public class GraphNode<V> implements Component<Tensor<V>>
{
    /*
         mode state meaning:
       -+------------+----------------------------------+-
        | _mode == 0 |  no Auto-Differentiation         |
       -+------------+----------------------------------+-
        | _mode > 0  |  forward Auto-Differentiation    |
       -+------------+----------------------------------+-
        | _mode < 0  |  backward Auto-Differentiation   |
       -+------------+----------------------------------+-
    */
    private final int _mode;

    private final AutoDiffMode _adMode;

    private final Function _function;

    /**
     * The GraphNodes of the input tensors. ('Parents' of the tensor of this node)
     * These are always the GraphNodes of the tensors from which the tensor payload of this
     * GraphNode has been formed.
     */
    private final GraphNode<V>[] _parents;

    private final List<BackPropTargets<V>> _targetsToAgents;

    private final List<WeakReference<GraphNode<V>>> _children = new ArrayList<>(1);

    private final NodePayload<V> _nodePayload;

    private int _usedAsDerivative = 0;

    private boolean _reliesOnJustInTimeProp = false;

    private PendingError<V> _pendingError = null;




    /**
     * @param function        Is the function that lead to the creation of this node.
     * @param call            The execution call, or null if the node is not a result of an execution call (a leave).
     * @param payloadSupplier Provides the payload of this node.
     */
    public GraphNode( Function function, ExecutionCall<Device<?>> call, Supplier<Result> payloadSupplier )
    {
        _checkConstructorArgValidity( function, call );

        Result out = payloadSupplier.get();
        if ( out == null ) throw new NullPointerException( "The result must no be null!" );

        NodePayload<V> data    = new NodePayload<>( null );
        AutoDiffMode   adMode  = ( call != null ? call.autogradMode()                     : AutoDiffMode.NOT_SUPPORTED          );
        int            mode    = ( call != null ? GraphNodeUtility.modeOf( adMode, call ) : ( out.get().rqsGradient() ? 1 : 0 ) );
        GraphNode<V>[] parents = ( call != null ? new GraphNode[call.arity()]             : null                                );

        if ( function != null && function.isDoingAD() ) { // Only functions with AutoDiff enabled create computation graphs!
            data = new NodePayload<>( out.get() );
            ((Tensor<V>)out.get()).set(this);
            if ( call != null ) {
                Tensor<V>[] inputs = (Tensor<V>[]) call.inputs();
                for ( int i = 0; i < inputs.length; i++ ) {
                    parents[i] = inputs[i].getGraphNode().orElseThrow(()->new IllegalStateException("Input tensors of a new graph-node must contain leave graph-nodes!"));
                    parents[i]._attachChild(this);
                }
            }
        }
        _nodePayload     = data;
        _mode            = mode;
        _function        = ( call == null ? null : function );
        _adMode          = adMode;
        _parents         = parents;
        _targetsToAgents = _registerADActions( out, function, call );
    }

    private void _checkConstructorArgValidity(
        Function function,
        ExecutionCall<Device<?>> call
    ) {
        if ( function == null && call != null )
            throw new IllegalArgumentException( "Branch graph nodes require a function!" );

        if ( call != null ) {
            Tensor<?>[] inputs = call.inputs();
            /* Applying JITProp and gradients */
            Neureka.Settings.AutoGrad adSetting = Neureka.get().settings().autograd();
            if ( adSetting.isApplyingGradientWhenTensorIsUsed() ) {
                for ( Tensor<?> t : inputs ) {
                    if ( !adSetting.isApplyingGradientWhenRequested() || t.gradientApplyRequested() ) {
                        t.applyGradient(); // activates JITProp if present and removes it...
                        t.setGradientApplyRequested( false );
                    }
                }
            }
            _checkInputValidity( inputs, function );
        }
    }

    private void _checkInputValidity(Tensor<?>[] inputs, Function function )
    {
        for ( int i = 0; i < inputs.length; i++ ) {
            GraphNode<V> child = (GraphNode<V>) inputs[ i ].getGraphNode().orElse(null);
            if ( child == null )
                throw new IllegalStateException(
                        "Input tensor at index '" + i + "' did not return a GraphNode instance." +
                         "Input tensors of a new GraphNode must be part of the computation graph!"
                    );
            if ( function.getOperation().isInline() && child.usesAD() )
                throw new IllegalStateException(
                        "Trying to apply inline operation '" + function.getOperation().getIdentifier() + "'\n" +
                        "on active autograd computation graph in non detached function.\n" +
                        "Please use detached functions instead! ( 'Function.create(\"" + function.getOperation().getIdentifier() + "(...)\", false)' )\n"
                    );
        }
    }

    /**
     *  This method extracts {@link ADAction}s from the provided {@link Function}
     *  (and its underlying {@link neureka.backend.api.Operation}) to be stored associated with a
     *  particular target {@link GraphNode} node used as reference for back-prop traversal
     *  when doing back-prop/autograd later on...
     */
    private List<BackPropTargets<V>> _registerADActions(
        Result output, Function function, ExecutionCall<? extends Device<?>> call
    ) {
        if ( call == null || !function.isFlat() )
            return Collections.emptyList(); // Leave nodes don't need agents!

        BackPropTargetCollector<V> collector = new BackPropTargetCollector<>();

        Tensor<V>[] inputs = (Tensor<V>[]) call.inputs();
        /* Returning if the above cannot form an AutoDiff computation graph! : */
        for ( Tensor<V> t : inputs )
            if ( t == output.get() ) return collector.getTargets(); // Output must be a unique tensor for AD!

        if ( this.usesAD() && function.isFlat() ) {
            /* Preparing for back propagation: */
            if ( this.usesForwardAD() )
            {
                for ( int i = 0; i < inputs.length; i++ ) {
                    GraphNode<V> srcNode = inputs[ i ].getGraphNode().orElseThrow(IllegalStateException::new);
                    if ( srcNode.usesAD() && function.dependsOn(i) ) {
                        if (
                            srcNode.size() == 0
                               || // Sources created by for example by dot/mm or x-mul are reverse-mode cases!
                            !srcNode.isLeave() && !srcNode._adMode.allowsForward()
                        ) {
                            ADAction agent = output.getAgentSupplier().supplyADActionFor(function, call.withArgs(Arg.DerivIdx.of(i)) );
                            collector.put( i, srcNode, agent );
                            _informPartialDerivative(agent);
                        } else {
                            /*  Chain rule (forward) for every derivative w.r.t. leaves (reverseAD or user leaves): */
                            int finalI = i;
                            Tensor<V> localDerivative = function.derive( inputs, i );
                            srcNode._forEachTargetActionPair(
                                ( targets, localADAction ) ->
                                {
                                    // The agent multiplies the local derivative with its stored partial derivative...
                                    Tensor<?> targetDerivative = localADAction.act( new ADTarget<>(targets.index(), this, localDerivative) );
                                    // ...this is now the new partial derivative with respect to the target node!
                                    ADAction agent = output.getAgentSupplier()
                                                            .supplyADActionFor(
                                                                function,
                                                                call.withArgs(
                                                                    Arg.VarIdx.of(call.getValOf(Arg.VarIdx.class)),
                                                                    Arg.DerivIdx.of(finalI),
                                                                    Arg.Derivative.of(targetDerivative)
                                                                )
                                                            );
                                    collector.put( finalI, targets.node(), agent );
                                    _informPartialDerivative(agent);
                                }
                            );
                        }
                    }
                }
            }
            else if ( this.usesReverseAD() )
            {
                for ( int i = 0; i < inputs.length; i++ ) {
                    GraphNode<V> srcNode = inputs[ i ].getGraphNode().orElseThrow(IllegalStateException::new);
                    if ( ( srcNode.usesAD() || inputs[ i ].rqsGradient() )) {
                        if ( !function.dependsOn(i) )
                            throw new IllegalStateException(
                                    "The function '" + function + "' does not have an input for " +
                                    "for argument index '" + i + "'!\n" +
                                    "This is most likely due to a bug in the implementation of the function or the underlying operation(s)."
                                );

                        ADAction agent = output.getAgentSupplier().supplyADActionFor(
                                                        function,
                                                        call.withArgs(Arg.DerivIdx.of(i),Arg.VarIdx.of(call.getValOf(Arg.VarIdx.class)))
                                                    );
                        collector.put( i, srcNode, agent );
                        _informPartialDerivative(agent);
                    }
                }
            }
        }
        return collector.getTargets();
    }


    /**
     * This short method simply migrates the error to the device of
     * the payload tensor and possibly also applies the error to
     * the payload if its 'requires gradient' flag is set to true.
     *
     * @param e This is an error value passed to this method ba a backward traversal.
     */
    private void _migrateAndOrApplyError( Tensor<V> e, Consumer<Tensor<V>> also ) {
        this.getPayload().ifPresent( payload -> {
            // It was not garbage collected:
            try {
                if ( payload.isOutsourced() ) payload.getDevice().store( e );
            } catch ( Exception exception ) {
                if ( payload.isUndefined() )
                    throw new IllegalStateException(
                            "An undefined payload tensor has been detected inside the computation graph!\n" +
                            "This is most likely due to an error occurring during tensor identity transfer (Also see AbstractComponentOwner).\n" +
                            "One type of constructor in the 'Tensor' class enables passing a String expression for execution, " +
                            "whose resulting tensor needs to be merged into the newly created one..."
                        );
                else
                    exception.printStackTrace();
            }
            if ( payload.rqsGradient() ) payload.mut().addToGradient( e );
            if ( also != null ) also.accept( payload );
        });
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
     * This node (and the corresponding tensor) was not created by a function! (it's a leave tensor)
     *
     * @return boolean
     */
    public boolean isLeave() { return _parents == null && _function == null; }

    public boolean isGraphLeave() {
        if ( this.isLeave() ) return true;
        for ( GraphNode<V> p : _parents )
            if ( p != null ) return true;

        return false;
    }

    /**
     *  Note: This method will never return null even if the actual payload tensor was garbage collected.
     *  This is because the {@link GraphNode} will remember the shape of the tensor.
     *
     *  @return The shape of the payload tensor represented by this {@link GraphNode}.
     */
    public List<Integer> getPayloadShape() { return _nodePayload.getPayloadShape(); }

    /**
     * @param newChild which references it's input namely the parent (this) has...
     */
    private synchronized void _attachChild( GraphNode<V> newChild ) {
        Objects.requireNonNull( newChild );
        if ( _pendingError != null )
            throw new IllegalStateException(
                    "This node cannot be used for forward propagation because it is currently actively involved in backward propagation!\n" +
                    "Only after the pending error of this node has received all expected error contributions from its children, " +
                    "can it be used for forward propagation again."
                );
        _children.add( new WeakReference<>( newChild, null ) );
    }

    /**
     *  The value of a graph node is the tensor to which it belongs (is a component of).  <br><br>
     *  Meaning it is the tensor owning this {@link GraphNode} component.
     *  It is referenced weakly because it might not be needed any more (Not referenced inside AD-Agent for example)
     *  and can therefore be garbage collected.
     *  <p>
     *  Warning: This method might return null because
     *           the payload is weakly referenced!
     *           Meaning that it might get garbage collected.
     *
     * @return The tensor payload of this graph-node.
     */
    public Optional<Tensor<V>> getPayload() { return Optional.ofNullable(_nodePayload.getPayload()); }

    @Override
    public boolean update( OwnerChangeRequest<Tensor<V>> changeRequest ) {
        changeRequest.executeChange(); // This can be an 'add', 'remove' or 'transfer' of this component!
        return true;
    }

    /**
     * This method is called by the JITProp component.
     * A pending should only ever be retrieved from a GraphNode once because
     * afterward the accumulated error is about to be back-propagated.
     * Therefore, this method nulls the reference when returning the PendingError instance.
     * @return Returns an instance of the PendingError class containing a error accumulation.
     */
    public PendingError<V> getAndRemovePendingError() {
        PendingError<V> pending = _pendingError;
        _pendingError = null;
        return pending;
    }

    /**
     * This method is the entry-point for the back-propagation process.
     * It sets up a key/value map which stores nodes and their intermediate error accumulations.
     * Accumulation occurs inside the private '_backward' method which traverses the computation graph
     * recursively, halts when errors can be accumulated, adds a PendingError and returns to the method below!
     * Here all the nodes and error values will then be carried (propagated) to the gradients!
     *
     * @param error The current error which is created by multiplying it with current size and traversing it.
     */
    public void backward( Tensor<V> error ) {
        Set<GraphNode<V>> pendingNodes = new HashSet<>();
        _backward( error, pendingNodes, false ); // Entry-point to private recursive back-propagation!
        if ( Neureka.get().settings().autograd().isRetainingPendingErrorForJITProp() )
            pendingNodes.forEach( n -> n._carryPendingBackPropToGradients( pendingNodes ) );
        else {
            /*
                Now we have a set of pending graph nodes with their respective error accumulations.
                However, some of them may not be accumulated fully yet, and that is not necessarily a problem.
                One un-accumulated error may only be un-accumulated because another (fully accumulated) pending error
                is not yet back-propagated to the gradients of the tensor owning the un-accumulated error.
                So we first need to back-propagate all fully accumulated errors recursively
                and then check if there are still un-accumulated errors left:
            */
            long numberOfFullyAccumulated;
            do {
                for ( GraphNode<V> n : new ArrayList<>(pendingNodes) )
                    if ( n._pendingError.isFullyAccumulated() ) {
                        n.backward(n._pendingError.getAccumulatedError()); // Continue back-propagation recursively!
                        pendingNodes.remove( n );
                    }

                numberOfFullyAccumulated = pendingNodes.stream().filter( n -> n._pendingError.isFullyAccumulated() ).count();
            }
            while ( numberOfFullyAccumulated > 0 ); // Repeat until all fully accumulated errors are back-propagated!

            // Now we can check if there are still un-accumulated errors left (there shouldn't be any!):
            _verifyErrorAccumulation( pendingNodes );
        }
        _deleteDerivativesRecursively(); // Cleanup after back-propagation!
    }

    private void _verifyErrorAccumulation( Set<GraphNode<V>> pendingNodes )
    {
        List<GraphNode<V>> notFullyAccumulated = pendingNodes
                                                    .stream()
                                                    .filter( n -> !n._pendingError.isFullyAccumulated() )
                                                    .collect(Collectors.toList());

        if ( !notFullyAccumulated.isEmpty() ) {
            String explanation = "Not all graph nodes have received the expected amount of errors from their children!\n" +
                                 "This usually happens because the recorded computation graph has multiple roots, which " +
                                 "is most likely because not all of your model outputs are captured by a common loss function.\n";
            String problem =
                    "The following graph nodes have not received the expected amount of errors from their children:\n\n" +
                    notFullyAccumulated
                        .stream().map( n ->
                            "    " + n +";\n" +
                            "        Accumulation: " + n._pendingError.getReceived() + "/" + n._pendingError.getExpectedToBeReceived() + ";\n" +
                            "        Generation: " + n._pendingError.getGeneration() + ";\n" +
                            "        Involved operations '" +
                            Optional.ofNullable(n._function)
                                    .map( fun ->
                                            fun.getAllFunctions()
                                               .stream()
                                               .filter( f -> f.getOperation() != null )
                                               .map( f -> f.getOperation().getIdentifier() )
                                               .collect(Collectors.joining("', '"))
                                    )
                                    .orElse("[]")
                            + "';\n" +
                            "        Children: " + n._children.stream().map( c -> String.valueOf(c.get())).collect(Collectors.joining(", ")) + ";"
                        )
                        .collect(Collectors.joining("\n")) +
                        "\n";

            List<List<GraphNode<V>>> rootPaths = new ArrayList<>();
            List<List<GraphNode<V>>> longestPaths = new ArrayList<>();
            for ( GraphNode<V> node : notFullyAccumulated ) {
                List<GraphNode<V>> longPath = new ArrayList<>();
                node._findRootPathsRecursively( rootPaths, new ArrayList<>(), longPath, this );
                longestPaths.add( longPath );
            }
            if ( !rootPaths.isEmpty() )
                problem += "\nThe following paths from the current node to the root node have been found:\n" +
                          IntStream.range(0, rootPaths.size())
                            .mapToObj( i -> (1+i) + ".: " + rootPaths.get(i).stream().map(GraphNode::toString).collect(Collectors.joining(" -> ")) )
                            .collect(Collectors.joining("\n"))+"\n";
            else
                problem += "\nWe tried to find paths from the current node to root nodes but failed to do so.\n" +
                            "This is really strange and should not happen!\n" +
                            "Here a snapshot of the computation graph from the root graph node:" +
                            "\n" +
                            _fancyToString(112);

            if ( !longestPaths.isEmpty() )
                problem += "\nHere are the longest paths:\n" +
                            IntStream.range(0, longestPaths.size())
                                .mapToObj( i -> (1+i) + ".: " + longestPaths.get(i).stream().map(GraphNode::toString).collect(Collectors.joining(" -> ")) )
                                .collect(Collectors.joining("\n"))+"\n";

            throw new IllegalStateException( explanation + problem );
        }
    }

    private boolean _findRootPathsRecursively(
        List<List<GraphNode<V>>> paths,
        List<GraphNode<V>> current,
        List<GraphNode<V>> longestPath,
        GraphNode<V> correctRoot
    ) {
        /*
           This method is used to find all possible paths from the current node to "invalid" root nodes.
           Root nodes are nodes which have no children but are still part of the autograd graph.
        */
        current.add( this );

        if ( this == correctRoot ) {
            paths.add( new ArrayList<>(current) ); // We have found a path to a root node!
        }
        else if ( current.size() > longestPath.size() ) {
            longestPath.clear();// We are looking for paths to root nodes which are not the correct root node! (debugging)
            longestPath.addAll( current );
        }

        boolean foundRoot = false;
        if ( !_children.isEmpty() )
            for ( WeakReference<GraphNode<V>> child : new ArrayList<>(_children) ) {
                GraphNode<V> c = child.get();
                if ( c != null )
                    foundRoot = foundRoot || c._findRootPathsRecursively( paths, current, longestPath, correctRoot );
            }

        current.remove( this ); // Remove this node from the current path so that we can continue searching for other paths!

        return foundRoot;
    }

    /**
     * This method traverses the computation graph and applies errors to gradients.
     * Errors might be accumulated temporarily or possibly longer for 'Just In Time propagation'.
     * JITProp is enabled in the global Neureka class.
     * It will traverse the path between a pending error and a tensor (rqsGradient==true)
     * containing the JITProp component which is triggered as soon as new gradients are needed or requested (applied).
     * This traverse however does not occur through the method below.
     * Instead, the 'backwardJIT' method is called by the JITProp component if present.
     * Intermediate error accumulations are stored in the '_pending_error' variable.
     * The method halts when an error can be accumulated and returns.
     * This graph node however is not forgotten but being noted in the 'pendingNodes' Set.
     *
     * @param error A tensor which traverses the computation graph according to the rules of reverse mode AutoDiff.
     */
    private void _backward( Tensor<V> error, Set<GraphNode<V>> pendingNodes, boolean allowPendingError )
    {
        _migrateAndOrApplyError( error, null );
        if ( this.usesAD() ) {
            /* Checking JIT-Prop conditions and create Pending error if possible */
            if ( allowPendingError && !this.isLeave() ) {//==> We are NOT inside a 'Just-In-Time-Backprop' process (new pending error can be created)
                int numOfADPaths = _numberOfReverseModeADChildren();// Multiple children triggers creation of a pending error
                if ( numOfADPaths > 1 ) {
                    if ( _pendingError == null ) {
                        _pendingError = new PendingError<>( error, numOfADPaths, 1 );
                        pendingNodes.add( this );
                    }
                    else {
                        if ( _pendingError.isFullyAccumulated() )
                            _pendingError = new PendingError<>( _pendingError.getAccumulatedError(), numOfADPaths, _pendingError.getGeneration() + 1 );

                        _pendingError.accumulate( error );
                    }

                    return;
                    /* Back-prop will be continued later! This node is being remembered in 'PendingError'
                       NOTE: Multiple AutoDiff paths leading to one node in history will be accumulated first! (performance)
                             This optimization is a light version of JITProp. JITProp builds on this!
                    */
                }
            }
            // The following call ADActions for reverse-mode AutoDiff!
            _forEachBackRef( error, ( t, e ) -> t._backward( e, pendingNodes, true ) );
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
     * @param pendingBackProp The set of graph nodes where further propagation is pending.
     */
    private void _carryPendingBackPropToGradients( Set<GraphNode<V>> pendingBackProp ) {
        _reliesOnJustInTimeProp = true; //:=> Shall be traversed at a later point in time...
        this.forEachTarget( t -> t._carryPendingBackPropToGradients( pendingBackProp ) );
        this.getPayload().ifPresent( p -> {
            if ( this.isLeave() && p.rqsGradient() ) {
                JITProp<V> jit = p.get( JITProp.class );
                if ( jit == null ) jit = new JITProp<>( pendingBackProp );
                else jit.addPending( pendingBackProp );
                p.set( jit );
            }
        });
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
    public void backwardJIT( Tensor<V> error ) {
        _backwardJIT( error, this );
        _deleteDerivativesRecursively();// Cleanup after back-propagation!
    }

    private void _backwardJIT(Tensor<V> error, GraphNode<V> source ) {
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
              Otherwise, it makes sense to accumulate errors further and wait for JIT-Prop traversing!
             */
            return; // This node will continue its propagation via a JIT-Prop component later!
        }
        if ( this.usesAD() )
            // The following call ADActions for reverse-mode AutoDiff!
            _forEachBackRef( error, ( t, e ) -> t._backwardJIT( e, source ) );
            // JITProp reverse mode-AutoDiff!
    }

    /**
     * This method is called after the backward call has been executed fully.
     * Derivatives are no longer used and will therefore be deleted when possible.
     * Deletion is forbidden if this node is flagged
     * as JITProp job. This means that the node is on the path between gradients
     * and pending error objects.
     * Only if JITProp is enabled (Neureka.get().settings().autograd()...) this flag will
     * deviate from its default state, namely: true!
     */
    private void _deleteDerivativesRecursively() {
        if ( !Neureka.get().settings().debug().isKeepingDerivativeTargetPayloads() ) { // <=- This flag is almost always false. (Used for testing)
            if ( !this.isReliesOnJustInTimeProp() && !_targetsToAgents.isEmpty() ) _targetsToAgents.clear();
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
        if ( !_children.isEmpty() ) {
            for ( WeakReference<GraphNode<V>> weak : _children ) {
                if ( weak != null && weak.get() != null ) {
                    GraphNode<V> child = weak.get(); // TODO: make test which asserts that Detached Function does not trigger this!
                    if ( child != null && child.usesReverseAD() ) count++;
                }
            }
        }
        return count;
    }

    private void _informPartialDerivative( ADAction agent ) {
        agent.partialDerivative()
            .ifPresent( d ->  {
                if ( !d.has( GraphNode.class ) )
                    d.set(new GraphNode<>( null, null, () -> Result.of(d) ));
                d.getGraphNode().get()._usedAsDerivative++;
            });
    }

    /**
     * This method checks if a given graph node is an AD target of this node.
     * This would mean that this node contains an AD-action for the given GraphNode (target).
     *
     * @param target The targeted derivation graph node reference.
     * @return boolean
     */
    public boolean has( GraphNode<V> target ) {
        return _targetsToAgents.stream().anyMatch( ref -> ref.node() == target );
    }

    /**
     * This is the number of AD-actions stored inside this node.
     * It can be interpreted as the 'number of AD paths'.
     *
     * @return int
     */
    public int size() {
        return _targetsToAgents.size();
    }

    /**
     * @param action The lambda performing an action on all targeted nodes and their agents.
     */
    public void forEachDerivative( BiConsumer<GraphNode<V>, ADAction> action ) {
        if ( _targetsToAgents.isEmpty() ) return;
        new ArrayList<>(_targetsToAgents).forEach(
            ( ref ) -> ref.actions().forEach( a -> action.accept( ref.node(), a ) )
        );
    }

    /**
     * @param error The error which ought to be passed to the {@link ADAction}s.
     * @param action A lambda action providing derivative and target node as parameter.
     */
    private void _forEachBackRef( Tensor<V> error, BiConsumer<GraphNode<V>, Tensor<V>> action ) {
        if ( _targetsToAgents.isEmpty() ) return;
        error.getMut().setIsIntermediate( false );
        for ( BackPropTargets<V> ref : new ArrayList<>(_targetsToAgents) )
            for ( ADAction a : ref.actions() )
                action.accept( ref.node(), (Tensor<V>) a.act( new ADTarget<>(ref.index(), ref.node(), error) ));
    }

    /**
     * @param action An action which should be applied to the graph nodes of all the partial derivatives.
     */
    public void forEachTarget( Consumer<GraphNode<V>> action ) {
        if ( _targetsToAgents.isEmpty() ) return;
        new ArrayList<>(_targetsToAgents).forEach( ref -> action.accept( ref.node() ) );
    }

    /**
     * @param action The action which ought to be applied to each target {@link GraphNode} / {@link ADAction} pair.
     */
    private void _forEachTargetActionPair( BiConsumer<BackPropTargets<V>, ADAction> action ) {
        if ( _targetsToAgents.isEmpty() ) return;
        new ArrayList<>( _targetsToAgents )
                .forEach( ref  -> ref.actions().forEach(a -> action.accept( ref, a ) ) );
    }


    /**
     * @return Checks if this node stores target / AD-action (usually derivatives) pairs.
     */
    public boolean hasDerivatives() { return !_targetsToAgents.isEmpty(); }

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
     *  a {@link GraphNode} instance which represents a {@link Tensor} instance
     *  having the {@link Tensor#rqsGradient()} flag set to true!
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
     *
     * @return The truth value determining if this graph node relies on just in time propagation.
     */
    public boolean isReliesOnJustInTimeProp() { return _reliesOnJustInTimeProp; }

    /**
     * Used by the Just-In-Time back-prop component.
     */
    public Optional<PendingError<V>> getPendingError() { return Optional.ofNullable( _pendingError ); }

    /**
     * The chain-rule states that the derivative of f(x) = h(g(x)) with respect to x is: g'(x) * h'(g(x))
     * An example would be:
     * f(x) = ((x*y)*z)
     * f'(x) = (1*y) * (1*z) = z*y
     * The values z,y or z*y must not be deleted as they are needed for back-propagation!
     */
    public boolean isUsedAsDerivative() { return _usedAsDerivative > 0; }

    /**
     * Recorded Function which produced this {@link GraphNode}.
     */
    public Optional<Function> getFunction() { return Optional.ofNullable( _function ); }

    public List<GraphNode<V>> getParents() { return _parents == null ? Collections.emptyList() : Arrays.asList( _parents ); }

    /**
     *  This variable holds a copy of the version of the payload tensor
     *  recorded when this GraphNode instance is instantiated.
     *  It must be treated as final and should never be modified.
     *  However, it can be read freely in order to
     *  check that the version of the payload hasn't changed.
     */
    public int getPayloadReferenceVersion() { return _nodePayload.payloadReferenceVersion(); }

    public DataType<V> getPayloadDataType() { return _nodePayload.payloadDataType(); }

    /**
     *  The children are {@link GraphNode} instances which represent computations
     *  involving the payload of this very {@link GraphNode} instance.
     */
    public List<WeakReference<GraphNode<V>>> getChildren() {
        return Collections.unmodifiableList( _children );
    }

    public boolean canBeDeleted() {
        Tensor<V> payload = _nodePayload.getPayload();
        if ( payload == null ) return true;
        if ( !isUsedAsDerivative() ) return true;
        /*
            This node is a derivative of another node.
            Should we delete it? Usually not, but if the
            payload tensor is not used by any other node
            then we can delete it.
         */
        int aliveAncestors = _numberOfExistingAncestors();
        if ( aliveAncestors > 0 ) {
            /*
                This node has ancestors whose "backward()" methods could be called.
                In this case we must not delete this node, because it might be used!
             */
            return false;
        }
        else {
            /*
                If the number of ancestors is zero then this means that it can most likely be deleted
                because no ancestors means that theoretically this node is not used as a derivative in the computation...
                However, it is theoretically possible that this node is used as a derivative
                in an alternative computation graph which is not connected to this computation graph.
             */
            return _numberOfDerivativeUsages(payload) == _usedAsDerivative;
        }
    }

    private int _numberOfExistingAncestors() {
        return getChildren()
                .stream()
                .map( WeakReference::get )
                .filter( Objects::nonNull )
                .mapToInt( n -> {
                    int count = n.getPayload().map( p -> !p.isDeleted() ? 1 : 0 ).orElse( 0 );
                    return count + n._numberOfExistingAncestors();
                })
                .sum();
    }

    private long _numberOfDerivativeUsages( Tensor<V> derivative ) {
        return getChildren()
                .stream()
                .map( WeakReference::get )
                .filter( Objects::nonNull )
                .mapToLong( n -> {
                    long usages =
                            n._targetsToAgents
                            .stream()
                            .flatMap( t -> t.actions().stream() )
                            .map( a -> a.partialDerivative().orElse(null) )
                            .filter( Objects::nonNull )
                            .filter( d -> d == derivative )
                            .count();

                    return usages + n._numberOfDerivativeUsages( derivative );
                })
                .sum();
    }

    /**
     * @return Returns the type of the node as descriptive String in capital letters.
     */
    public String type() {
        String type = "";
        if ( this.isLeave() ) type += "LEAVE";
        else type += "BRANCH";
        type += this.getPayload()
                    .filter( p -> !p.isDeleted() )
                    .map( p -> p.rqsGradient() ? " RQS GRADIENT" : "" )
                    .orElse(" DELETED");
        return type;
    }

    @Override
    public String toString() { return toString( Print.SIMPLE ); }

    public enum Print { SIMPLE, COMPACT, FANCY }

    /**
     * @param mode The format of the string representation.
     * @return Returns a String representation of this node.
     */
    public String toString( Print mode ) {
        LogUtil.nullArgCheck( mode, "mode", Print.class );
        switch ( mode ) {
            case SIMPLE: return _simpleToString();
            case FANCY: return _fancyToString(1028);
            case COMPACT: return _compactToString();
        }
        throw new IllegalStateException();
    }

    private String _simpleToString() {
        return this.getClass().getSimpleName()+"@"+Integer.toHexString(hashCode())+"[" +
                            "parents=[" + ( _parentsToString() ) + "]," +
                            "function=" + (_function == null ? "?" : _function) + "," +
                            "shape=" + (getPayloadShape() != null ? getPayloadShape().stream().map(Object::toString).collect(Collectors.joining("x")) : "?" ) +
                        "]";
    }

    private String _parentsToString() {
        return
            Optional.ofNullable( _parents ).map( parents ->
                Arrays.stream(_parents)
                .map(GraphNode::getPayload)
                .map( p ->
                    p.filter( t -> !t.isDeleted() )
                    .map( t->t.shape().stream().map(Object::toString).collect(Collectors.joining("x")))
                    .orElse("?")
                )
                .collect(Collectors.joining(", "))
            )
            .orElse("?");
    }

    private String _fancyToString( int depthLimit ) {
        return "]> GRAPH:\n]\n" + _toString( "]    0", true, Print.COMPACT, depthLimit ) + "\n]\n]|END|>";
    }

    private String _compactToString() {
        Optional<Tensor<V>> payload = getPayload();
        String nid = this.getClass().getSimpleName();// + ( m.contains( "n" ) ? "#" + Long.toHexString( getNodeID() ) : "" );
        return " " + nid + "[ "
                + ( _function == null ? "" : _function + " => " )
                + (
                    payload.map( p -> p.toString(
                        settings -> settings
                                .setRowLimit( 3 )
                                .setIsScientific( true )
                                .setIsMultiline( false )
                                .setHasGradient( false )
                                .setCellSize( 1 )
                                .setHasValue( true )
                                .setHasRecursiveGraph( false )
                                .setHasDerivatives( false )
                                .setHasShape( true )
                                .setIsCellBound( false )
                                .setPostfix( "" )
                                .setPrefix( "" )
                                .setHasSlimNumbers( false )
                        )
                    ).orElse("?")
                ) +
                ", type='" + this.type() + "'" +
                "] ";
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
    private String _toString( String deep, boolean isLast, Print mode, int depthLimit ) {
        String delimiter = ( isLast ? ("    ") : ("|   ") );
        String arrow = ( (char) 187 ) + "" + ( _parents != null ? String.valueOf( _parents.length ) : "0" ) + ( (char) 187 );
        StringBuilder asString = new StringBuilder( deep + arrow + toString( mode ) );
        deep = deep.substring( 0, deep.length() - 1 );
        if ( _parents != null ) {
            asString.append( "\n" ).append( deep ).append( isLast ? "   \\\n" : "|  \\\n" );
            for ( int i = 0; i < _parents.length; i++ ) {
                boolean last = ( i == _parents.length - 1 );
                if ( deep.length() < depthLimit ) { // prevent stack overflow + blowing up the heap (although the heap will probably already have blown up before this point)
                    asString.append( i != 0 ? deep + delimiter + "|\n" : "" );
                    asString.append( _parents[ i ]._toString( deep + delimiter + i, last, mode, depthLimit ) ).append( "\n" );
                } else
                    asString.append( deep + delimiter + i + " ... (graph to deep for further traversal) " + (!last ? "\n" : " ") );
            }
            asString = new StringBuilder( asString.substring( 0, asString.length() - 1 ) );
        }
        return asString.toString();
    }

}
