package neureka.autograd;

import java.util.ArrayList;
import java.util.List;
import java.util.TreeMap;
import java.util.stream.Collectors;

class BackPropTargetCollector<V> {

    private TreeMap<GraphNode<V>, Value> _targetsToAgents;

    /**
     * @param target nodes are graph nodes which contain either tensors requiring errors for accumulation and/or more targets.
     * @param agent ADAction's are used during back-propagation in order to distribute an error throughout the graph.
     */
    public void put( int index, GraphNode<V> target, ADAction agent ) {
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


    private static class Value {
        private final int _index;
        private final List<ADAction> _agents = new ArrayList<>();

        private Value(int index, ADAction agent) {
            _index = index;
            _agents.add(agent);
        }

        public int index() { return _index; }

        public List<ADAction> agents() { return _agents; }
    }

}
