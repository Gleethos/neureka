package neureka.autograd;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.calculus.Function;
import neureka.devices.Device;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.ref.WeakReference;
import java.util.List;
import java.util.TreeMap;

public class GraphNodeAssembler<V> {

    private int _mode;

    private boolean _allows_forward;

    private boolean _allows_backward;

    private boolean _reliesOnJustInTimeProp = false;

    private PendingError<V> _pendingError = null;

    private boolean _isUsedAsDerivative = false;

    private Function _function;

    private GraphNode<V>[] _parents;

    private WeakReference<Tsr<V>> _payload;

    private int _payloadReferenceVersion = -1;

    private TreeMap<GraphNode<V>, List<ADAgent>> _targetsToAgents;

    private GraphLock _lock;

    private List<WeakReference<GraphNode<V>>> _children;

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
