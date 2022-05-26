package neureka.autograd;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.algorithms.fun.AutoDiffMode;
import neureka.calculus.Function;
import neureka.devices.Device;

import java.util.ArrayList;
import java.util.List;
import java.util.TreeMap;
import java.util.stream.Collectors;

/**
 *  This class exists in order to allow for {@link GraphNode}s to be instantiated
 *  with final field variables by collecting them when defined
 *  within constructor methods...
 */
final class GraphNodeAssemblyState<V> {

    private int _mode;

    private AutoDiffMode _adMode;

    private Function _function;

    private GraphNode<V>[] _parents;

    private GraphLock _lock;

    private TreeMap<GraphNode<V>, Value> _targetsToAgents;

    /**
     * @param target nodes are graph nodes which contain either tensors requiring errors for accumulation and/or more targets.
     * @param agent ADAgent's are used during back-propagation in order to distribute an error throughout the graph.
     */
    public void put( int index, GraphNode<V> target, ADAgent agent ) {
        if ( _targetsToAgents == null ) _targetsToAgents = new TreeMap<>((a, b) -> a.hashCode() - b.hashCode());

        if ( _targetsToAgents.containsKey( target ) )
            _targetsToAgents.get( target ).agents().add( agent );
        else
            _targetsToAgents.put( target, new Value(index, agent) );
    }

    public List<BackPropTargets<V>> getTargets() {
        if ( _targetsToAgents == null ) return null;
        else
            return _targetsToAgents.entrySet()
                                .stream()
                                .map( e -> new BackPropTargets<>( e.getValue().index(), e.getKey(), e.getValue().agents() ) )
                                .collect(Collectors.toList());
    }

    /**
     * This is the number of AD-actions stored inside this node.
     * It can be interpreted as the 'number of AD paths'.
     *
     * @return int
     */
    public int size() { return _targetsToAgents != null ? _targetsToAgents.size() : 0; }

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
    public void modeOf(ExecutionCall<? extends Device<?>> call )
    {
        Tsr<V>[] inputs = (Tsr<V>[]) call.inputs();
        int resultMode = 0;
        int[] modes = new int[ inputs.length ];
        int inputMode = 0;
        for ( int i = 0; i < inputs.length; i++ ) {
            GraphNode<V> node = inputs[ i ].getGraphNode(); // Not null checked in constructor!
            modes[ i ] = ( inputs[ i ].rqsGradient() ) ? 1 : node.getMode();
            inputMode += ( modes[ i ] != 0) ? 1 : 0;
        }
        _adMode = call.autogradMode();
        if ( inputMode == 1 && _adMode.allowsForward() ) { // Convolution and reshaping prohibit forward AutoDiff
            for ( int i = 0; i < inputs.length; i++ ) {
                resultMode +=
                        ( modes[ i ] == 0 )
                                ? 0
                                : ( modes[ i ] < 0 ) ? 1 : modes[ i ] + 1;
            }
        } // Reverse mode auto-differentiation :
        else if ( _adMode.allowsBackward() ) resultMode = -inputMode;

        _mode = resultMode;
    }

    public AutoDiffMode adMode() { return _adMode; }


    public Function function() { return _function; }

    public GraphNodeAssemblyState<V> setFunction(Function function ) {
        _function = function;
        return this;
    }

    public GraphNode<V>[] parents() { return _parents; }

    public void setParents(GraphNode<V>[] parents ) { _parents = parents; }

    public GraphLock lock() { return _lock; }

    public void setLock(GraphLock lock) { _lock = lock; }

    private static class Value {
        private final int _index;
        private final List<ADAgent> _agents = new ArrayList<>();

        private Value(int index, ADAgent agent) {
            _index = index;
            _agents.add(agent);
        }

        public int index() { return _index; }

        public List<ADAgent> agents() { return _agents; }
    }

}
