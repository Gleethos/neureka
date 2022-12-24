package neureka.autograd;

import java.util.Collections;
import java.util.List;

final class BackPropTargets<V>
{
    private final GraphNode<V> _node;
    private final int _index;
    private final List<ADAction> _actions;

    public BackPropTargets( int index, GraphNode<V> node, List<ADAction> actions ) {
        _node = node;
        _index = index;
        _actions = Collections.unmodifiableList(actions);
    }

    public GraphNode<V> node() { return _node; }

    public int index() { return _index; }

    public List<ADAction> actions() { return _actions; }
}
