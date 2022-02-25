package neureka.autograd;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.devices.Device;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.TreeMap;
import java.util.function.Supplier;

public class GraphNodeAssembler<V> {

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


    public int get_mode() {
        return _mode;
    }

    /**
     * @param _mode The mode of this GraphNode! ( m<0 : backward-AD, m>0 : forward-AD, m=0 : no-AD )
     */
    public GraphNodeAssembler<V> set_mode(int _mode) {
        this._mode = _mode;
        return this;
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
     */
    public void _modeOf( ExecutionCall<? extends Device<?>> call )
    {
        Tsr<V>[] inputs = (Tsr<V>[]) call.getTensors();
        int resultMode = 0;
        int[] modes = new int[ inputs.length ];
        int inputMode = 0;
        for ( int i = 0; i < inputs.length; i++ ) {
            GraphNode<V> node = inputs[ i ].getGraphNode(); // Not null checked in constructor!
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

        _mode = resultMode;
    }

    public boolean is_allows_forward() {
        return _allows_forward;
    }

    public GraphNodeAssembler<V> set_allows_forward(boolean _allows_forward) {
        this._allows_forward = _allows_forward;
        return this;
    }

    public boolean is_allows_backward() {
        return _allows_backward;
    }

    public GraphNodeAssembler<V> set_allows_backward(boolean _allows_backward) {
        this._allows_backward = _allows_backward;
        return this;
    }

    public boolean is_reliesOnJustInTimeProp() {
        return _reliesOnJustInTimeProp;
    }

    public GraphNodeAssembler<V> set_reliesOnJustInTimeProp(boolean _reliesOnJustInTimeProp) {
        this._reliesOnJustInTimeProp = _reliesOnJustInTimeProp;
        return this;
    }

    public PendingError<V> get_pendingError() {
        return _pendingError;
    }

    public GraphNodeAssembler<V> set_pendingError(PendingError<V> _pendingError) {
        this._pendingError = _pendingError;
        return this;
    }

    public boolean is_isUsedAsDerivative() {
        return _isUsedAsDerivative;
    }

    public GraphNodeAssembler<V> set_isUsedAsDerivative(boolean _isUsedAsDerivative) {
        this._isUsedAsDerivative = _isUsedAsDerivative;
        return this;
    }

    public Function get_function() {
        return _function;
    }

    public GraphNodeAssembler<V> set_function(Function _function) {
        this._function = _function;
        return this;
    }

    public GraphNode<V>[] get_parents() {
        return _parents;
    }

    public GraphNodeAssembler<V> set_parents(GraphNode<V>[] _parents) {
        this._parents = _parents;
        return this;
    }

    public WeakReference<Tsr<V>> get_payload() {
        return _payload;
    }

    public GraphNodeAssembler<V> set_payload(WeakReference<Tsr<V>> _payload) {
        this._payload = _payload;
        return this;
    }

    public int get_payloadReferenceVersion() {
        return _payloadReferenceVersion;
    }

    public GraphNodeAssembler<V> set_payloadReferenceVersion(int _payloadReferenceVersion) {
        this._payloadReferenceVersion = _payloadReferenceVersion;
        return this;
    }

    public TreeMap<GraphNode<V>, List<ADAgent>> get_targetsToAgents() {
        return _targetsToAgents;
    }

    public GraphNodeAssembler<V> set_targetsToAgents(TreeMap<GraphNode<V>, List<ADAgent>> _targetsToAgents) {
        this._targetsToAgents = _targetsToAgents;
        return this;
    }

    public GraphLock get_lock() {
        return _lock;
    }

    public GraphNodeAssembler<V> set_lock(GraphLock _lock) {
        this._lock = _lock;
        return this;
    }

    public List<WeakReference<GraphNode<V>>> get_children() {
        return _children;
    }

    public GraphNodeAssembler<V> set_children(List<WeakReference<GraphNode<V>>> _children) {
        this._children = _children;
        return this;
    }

    public long get_nodeID() {
        return _nodeID;
    }

    public GraphNodeAssembler<V> set_nodeID(long _nodeID) {
        this._nodeID = _nodeID;
        return this;
    }
}
