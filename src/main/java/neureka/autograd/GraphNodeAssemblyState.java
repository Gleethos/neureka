package neureka.autograd;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.calculus.Function;
import neureka.devices.Device;

/**
 *  This class exists in order to allow for {@link GraphNode}s to be instantiated
 *  with final field variables by collecting them when defined
 *  within constructor methods...
 */
final class GraphNodeAssemblyState<V> {

    private int _mode;

    private boolean _allowsForward;

    private boolean _allowsBackward;

    private Function _function;

    private GraphNode<V>[] _parents;

    private int _payloadReferenceVersion = -1;

    private GraphLock _lock;

    private long _nodeID = -1;


    public int mode() { return _mode; }

    /**
     * @param mode The mode of this GraphNode! ( m<0 : backward-AD, m>0 : forward-AD, m=0 : no-AD )
     */
    public GraphNodeAssemblyState<V> setMode(int mode ) {
        _mode = mode;
        return this;
    }

    /**
     *  Evaluates and sets the auto-grad/auto-differentiation mode:
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
        _allowsForward = call.allowsForward();
        _allowsBackward = call.allowsBackward();
        if ( inputMode == 1 && _allowsForward) { // Convolution and reshaping prohibit forward AutoDiff
            for ( int i = 0; i < inputs.length; i++ ) {
                resultMode +=
                        ( modes[ i ] == 0 )
                                ? 0
                                : ( modes[ i ] < 0 ) ? 1 : modes[ i ] + 1;
            }
        } // Reverse mode auto-differentiation :
        else if (_allowsBackward) resultMode = -inputMode;

        _mode = resultMode;
    }

    public boolean isAllowsForward() { return _allowsForward; }

    public boolean isAllowsBackward() { return _allowsBackward; }

    public Function function() { return _function; }

    public GraphNodeAssemblyState<V> setFunction(Function function ) {
        _function = function;
        return this;
    }

    public GraphNode<V>[] parents() { return _parents; }

    public GraphNodeAssemblyState<V> setParents(GraphNode<V>[] parents ) {
        _parents = parents;
        return this;
    }

    public int payloadReferenceVersion() { return _payloadReferenceVersion; }

    public GraphNodeAssemblyState<V> setPayloadReferenceVersion(int payloadReferenceVersion) {
        _payloadReferenceVersion = payloadReferenceVersion;
        return this;
    }

    public GraphLock lock() { return _lock; }

    public GraphNodeAssemblyState<V> setLock(GraphLock lock) {
        _lock = lock;
        return this;
    }

    public long nodeID() { return _nodeID; }

    public GraphNodeAssemblyState<V> setNodeID(long nodeID) {
        _nodeID = nodeID;
        return this;
    }
}