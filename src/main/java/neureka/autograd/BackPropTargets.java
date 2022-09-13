package neureka.autograd;

import java.util.Collections;
import java.util.List;

final class BackPropTargets<V>
{
    private final GraphNode<V> _node;
    private final int _index;
    private final List<ADAction> _agents;

    public BackPropTargets( int index, GraphNode<V> node, List<ADAction> agents ) {
        _node = node;
        _index = index;
        _agents = Collections.unmodifiableList(agents);
    }

    public GraphNode<V> node() { return _node; }

    public int index() { return _index; }

    public List<ADAction> agents() { return _agents; }
}
