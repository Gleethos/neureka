package neureka.autograd;

import java.util.Collections;
import java.util.List;

class BackPropTargets<V>
{
    private final GraphNode<V> _node;
    private final int _index;
    private final List<ADAgent> _agents;

    public BackPropTargets( int index, GraphNode<V> node, List<ADAgent> agents ) {
        _node = node;
        _index = index;
        _agents = Collections.unmodifiableList(agents);
    }

    public GraphNode<V> node() { return _node; }

    public int index() { return _index; }

    public List<ADAgent> agents() { return _agents; }
}
