package neureka.autograd;

import java.util.Collections;
import java.util.List;

public class BackPropBridge<V>
{
    private final GraphNode<V> _target;
    private final int _index;
    private final List<ADAgent> _agents;

    public BackPropBridge( int index, GraphNode<V> target, List<ADAgent> agents ) {
        _target = target;
        _index = index;
        _agents = Collections.unmodifiableList(agents);
    }

    public GraphNode<V> target() { return _target; }

    public int index() { return _index; }

    public List<ADAgent> agents() { return _agents; }
}
